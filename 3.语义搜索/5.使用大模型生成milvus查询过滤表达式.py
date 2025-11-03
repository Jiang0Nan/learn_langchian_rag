import os
import re
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from pymilvus.model.hybrid import  BGEM3EmbeddingFunction
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
# ===========================初始化
client = MilvusClient(uri="http://localhost:19530")
base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'
chat_model =  init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek")

embedding_model = BGEM3EmbeddingFunction(use_fp16=False, device="cpu", model_name=r"D:\files\models\bge-m3",return_sparse=False)

#=====================================创建对应的数据库表
fields = [
    FieldSchema(
        name="pk",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=100,
    ),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="age", dtype=DataType.INT64),
    FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="hobby", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_model.dim['dense']),
]
schema = CollectionSchema(fields=fields, description="User data embedding example")
collection_name = "user_data_collection"

if client.has_collection(collection_name):
    client.drop_collection(collection_name)
# Strong consistency waits for all loads to complete, adding latency with large datasets
# client.create_collection(
#     collection_name=collection_name, schema=schema, consistency_level="Strong"
# )
client.create_collection(collection_name=collection_name, schema=schema)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 128},
)
client.create_index(collection_name=collection_name, index_params=index_params)



#  ===========================构造数据并插入数据
data_to_insert = [
    {"name": "John", "age": 23, "city": "Shanghai", "hobby": "Drinking coffee"},
    {"name": "Alice", "age": 29, "city": "New York", "hobby": "Reading books"},
    {"name": "Bob", "age": 31, "city": "London", "hobby": "Playing chess"},
    {"name": "Eve", "age": 27, "city": "Paris", "hobby": "Painting"},
    {"name": "Charlie", "age": 35, "city": "Tokyo", "hobby": "Cycling"},
    {"name": "Grace", "age": 22, "city": "Berlin", "hobby": "Photography"},
    {"name": "David", "age": 40, "city": "Toronto", "hobby": "Watching movies"},
    {"name": "Helen", "age": 30, "city": "Sydney", "hobby": "Cooking"},
    {"name": "Frank", "age": 28, "city": "Beijing", "hobby": "Hiking"},
    {"name": "Ivy", "age": 26, "city": "Seoul", "hobby": "Dancing"},
    {"name": "Tom", "age": 33, "city": "Madrid", "hobby": "Writing"},
]



def get_embeddings(texts):
    return embedding_model(texts)


texts = [
    f"{item['name']} from {item['city']} is {item['age']} years old and likes {item['hobby']}."
    for item in data_to_insert
]
embeddings = get_embeddings(texts)

insert_data = []
dense = embeddings.get('dense')
# dense_embedding = [e.tolist() if hasattr(e, "tolist") else e for e in dense]

for item, embedding in zip(data_to_insert, dense):
    item_with_embedding = {
        "pk": str(uuid.uuid4()),
        "name": item["name"],
        "age": item["age"],
        "city": item["city"],
        "hobby": item["hobby"],
        "embedding":embedding,
    }
    insert_data.append(item_with_embedding)

client.insert(collection_name=collection_name, data=insert_data)

print(f"Collection '{collection_name}' has been created and data has been inserted.")


# ===================================查询打印3个样本

client.load_collection(collection_name=collection_name)

result = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "age", "city", "hobby"],
    limit=3,
)

for record in result:
    print(record)

# ========================过滤表达式文档
import docling
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
docs = [
    converter.convert(url)
    for url in [
        "https://milvus.io/docs/boolean.md",
        "https://milvus.io/docs/basic-operators.md",
        "https://milvus.io/docs/filtering-templating.md",
    ]
]

for doc in docs[:3]:
    print(doc.document.export_to_markdown())



# LLM 驱动的过滤器生成
import json
from IPython.display import display, Markdown

context = "\n".join([doc.document.export_to_markdown() for doc in docs])

prompt = """
You are an expert Milvus vector database engineer. Your task is to convert a user's natural language query into a valid Milvus filter expression, using the provided Milvus documentation as your knowledge base.

Follow these rules strictly:
1. Only use the provided documents as your source of knowledge.
2. Ensure the generated filter expression is syntactically correct.
3. If there isn't enough information in the documents to create an expression, state that directly.
4. Only return the final filter expression. Do not include any explanations or extra text.

---
**Milvus Documentation Context:**
{context}

---
**User Query:**
{user_query}

---
**Filter Expression:**
"""



def generate_filter_expr(user_query):
    """
    Generates a Milvus filter expression from a user query using GPT-4o-mini.
    """
    prompt.format(user_query=user_query,context=context)

    completion = chat_model.invoke( input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ]
    )
    cleaned = re.sub(r"<think>.*?</think>\n\n", "", completion.content, flags=re.DOTALL)
    return cleaned


user_query = "Find people older than 30 who live in London, Tokyo, or Toronto"

filter_expr = generate_filter_expr(user_query)

print(f"Generated filter expression: {filter_expr}")



# ============================测试生成的过滤器

match = re.search(r"```.*?```", filter_expr, flags=re.DOTALL)
if match:
    filter_expr=match.group(0)
clean_filter = (
    filter_expr.replace("```", "").replace('filter="', "").replace('"', "").strip()
)
print(f"Using filter: {clean_filter}")

query_embedding = get_embeddings(user_query)
search_results = client.search(
    collection_name="user_data_collection",
    data=[query_embedding.get('dense')],
    limit=10,
    filter=clean_filter,
    output_fields=["pk", "name", "age", "city", "hobby"],
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    },
)

print("Search results:")
for i, hits in enumerate(search_results):
    print(f"Query {i}:")
    for hit in hits:
        print(f"  - {hit}")
    print()


