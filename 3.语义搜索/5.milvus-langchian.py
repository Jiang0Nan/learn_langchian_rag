import os

from lear_langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import  db,utility,Collection

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

model = init_chat_model(
    model="deepseek-r1:7b-qwen-distill-q4_K_M", model_provider="ollama",
    base_url="http://localhost:11434")


model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True, "batch_size": 32}
embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\files\models\bge-m3",
    cache_folder=r"D:\files\models\bge-m3",
    model_kwargs=model_kwargs,  # 模型的参数
    encode_kwargs=encode_kwargs,  # encode的参数
    show_progress=True,
    # multi_process = True,#是否启用多进程并行编码，默认 False（注意：可能和某些环境冲突）
)

db_name = "milvus_test"

try :
    existing_db = db.list_database()
    if db_name in existing_db :
        print(f"Database-{db_name} already exists")
        db.using_database(db_name)

        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=db_name)
            collection.drop()
            print(f"Dropped collection {collection_name}")
        db.drop_database(db_name)
        print(f"Dropped database {db_name}")
    else:
        print(f"Creating database {db_name}")
        database = db.create_database(db_name)
        print(f"Created database {db_name}")
except Exception as e:
    print(f"出错了，message = {e}")
