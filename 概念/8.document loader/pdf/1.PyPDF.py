import asyncio

from lear_langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
file_path = r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf"

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'

model = init_chat_model(model=model_name, base_url=base_url, model_provider="ollama")

# 加载文件
loader = PyPDFLoader(file_path)
async def load_pdf_async(file_path):
    pages = []
    loader = PyPDFLoader(file_path)
    async for page in loader.alazy_load():
        pages.append(page)
    return pages



# 存入数据库
def init_inmemory_vector_store(page,embedding):
    """

    :param page: 需要嵌入的文本
    :param embedding: 嵌入模型
    :return:
    """
    return InMemoryVectorStore.from_documents(page,embedding)

# a = InMemoryVectorStore.from_documents()
# a.similarity_search()
# a.search(q,search_type="similarity_score_threshold",score_threshold=0.7,k=5)# "similarity","mmr", or "similarity_score_threshold".
# a.similarity_search_with_relevance_scores()#返回的是带相关性分数的结果。
# a.similarity_search_by_vector()#跳过了 query 转 embedding 的步骤



# 检索
def searh(q,k,vector):

    return vector.similarity_search(q,k)


if __name__ == '__main__':
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True, "return_dense": True, "return_sparse": True,
                     "return_colbert_vecs": True}
    hf = HuggingFaceEmbeddings(
        model_name=r"D:\files\models\bge-m3",
        cache_folder=r"D:\files\models\bge-m3",
        model_kwargs=model_kwargs,  # 文档编码参数
        encode_kwargs=encode_kwargs,  # 查询编码参数（覆盖文档的部分配置）
        show_progress=True,
    )

    pages = asyncio.run(load_pdf_async(file_path))
    vector = init_inmemory_vector_store(pages[0],embedding=hf)
    res = searh("成分是什么",5,vector)
    for re in res:
        print(re,end="\n===================\n")
