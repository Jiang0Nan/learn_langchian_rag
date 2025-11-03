import json
import os
import re
import time
import webbrowser
from pprint import pprint

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


from parctise.db.milvus_server import MilvusServer
import torch
from langchain_core.documents import Document
from keybert import KeyBERT
import jieba
from parctise.utils.tools import hash_str2int
# 可视化LangGraph
from IPython.display import Image, display


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"



def search(question:str,embedding_model,milvus,collection_name, output_fields):
    result = []
    ids = []
    if isinstance(embedding_model ,BGEM3EmbeddingFunction):
        embedded_q = embedding_model([question])
        milvus.load_collection(collection_name=collection_name, replica_number=1)

        hybrid_results = milvus.hybrid_search(
            query_dense_embedding=embedded_q["dense"][0].tolist(),
            query_sparse_embedding=embedded_q["sparse"][[0]],
            sparse_weight=0.7,
            dense_weight=0.3,
            limit=8,
            output_fields=output_fields,
            timeout=20,
            # expr :str = None
        )
        for i in hybrid_results:
            for j in i:
                if j.get('id') not in ids:
                    result.append(j)
                    ids.append(j.get('id'))

        dense_results = milvus.dense_search(
            query_dense_embedding=embedded_q['dense'][0],
            limit=8,
            output_fields=output_fields,
            timeout=20
        )
        for i in dense_results:
            for j in i:
                if j.get('id') not in ids:
                    result.append(j)
                    ids.append(j.get('id'))


        sparse_results = milvus.sparse_search(
            query_sparse_embedding=embedded_q["sparse"][[0]],
            limit=8,
            output_fields=output_fields,
            timeout=20
        )
        for i in sparse_results:
            for j in i:
                if j.get('id') not in ids:
                    result.append(j)
                    ids.append(j.get('id'))

        milvus.release_collection(collection_name=collection_name)
    return result

def get_extract_keywords(question):

    # 1. 初始化KeyBERT模型（中文建议用"paraphrase-multilingual-MiniLM-L12-v2"）
    kw_model = KeyBERT(model=r"D:\files\models\paraphrase-multilingual-MiniLM-L12-v2")

    # 2. 分词（KeyBERT中文需先分词，用空格分隔）
    text_cut = " ".join(jieba.lcut(question))

    # 4. 提取关键词（支持长短语，设置ngram_range=(1,2)表示提取1-2个词的短语）
    keywords = kw_model.extract_keywords(
        text_cut,
        keyphrase_ngram_range=(1, 2),  # 关键词长度：1-2个词
        top_n=3,  # 保留Top 3
        threshold=0.3  # 过滤分数低于0.3的关键词
    )

    return keywords

from langchain_core.runnables import Runnable, RunnablePassthrough


class CustomRetriever(Runnable):
    def __init__(self, docs):
        self.docs = docs  # 存储Document列表

    def invoke(self, query, **kwargs):
        # 接收查询（虽然这里用不到，但符合Runnable接口规范），返回文档列表
        return self.docs

def retrieval(question:str,embedding_model,compressor,milvus,collection_name):
    """负责吧问题转为数据库方便查询的语句"""
    # 处理不必要的虚词，符号，

    original_question = question
    # from parctise.utils.tools import add_space_between_eng_zh,rmWWW
    # question = add_space_between_eng_zh(question) # 中英文之间添加空格
    # question = rmWWW(question) # 去掉什么，等虚词
    #
    # tks = rag_tokenizer.tokenize(question).split()
    #
    # keyword = [t for t in tks if t ]
    # 生成的不太准确暂不启用，根据最后效果选择使用大模型
    keyword = get_extract_keywords(original_question)

    # todo 查找同义词（看效果在添加，一般使用词库，但是不够实时，因此看情况使用大模型）
    output_fields = ['id','doc_id','filename','content','page_ids','tokenizer_content',]

    # todo 可能存在查询为空的情况，可以调整阈值查询
    search_data = search(original_question,embedding_model,milvus,collection_name,output_fields)


    result = [
        Document(
            page_content=item['content'],  # 文档内容
            metadata={k: v for k, v in item.items() if k != 'content'}  # 其他字段作为元数据
        ) for item in search_data
    ]
    # todo 按照距离得分进行排序 (后面可以增加分区查询试试以及关键词筛选，)
    reranker_start_time = time.time()
    # 速度太慢 暂时弃用
    # reranker_data = [( question,i.get('content','')) for i in search_data]
    # reranker_start_time = time.time()
    # reranke_score = reranker.compute_score(reranker_data)
    # 转为runnable对象
    custom_retriever = CustomRetriever(docs=result)
    # langchian FlagReranker的快很多
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=custom_retriever
    )
    compressed_docs = compression_retriever.invoke(question)

    print(f"reranker 花费的时间为{time.time()-reranker_start_time}，检索结果长度为{len(result)}，排序后的结果{compressed_docs}")




    return compressed_docs


def kb_prompt(kbinfos, max_tokens, hash_id=False):
    """
    构建知识库提示词 - 将检索到的文档块格式化为LLM可理解的提示词

    Args:
        kbinfos: 检索结果信息，包含chunks和doc_aggs
        max_tokens: 最大token数量限制
        hash_id: 是否使用哈希ID（默认使用序号ID）

    Returns:
        list: 格式化后的知识库内容列表，用于构建LLM提示词
    """

    def draw_node(k, line):
        """
        绘制节点信息 - 格式化文档元数据

        Args:
            k: 字段名称
            line: 字段值

        Returns:
            str: 格式化后的节点信息
        """
        if not line:
            return ""
        # 将换行符替换为空格，保持单行格式
        return f"\n├── {k}: " + re.sub(r"\n+", " ", line, flags=re.DOTALL)

    # 重新构建格式化的知识库内容
    knowledges = []
    for i, ck in enumerate(kbinfos):
        # 构建块ID（序号或哈希值）
        cnt = "\nID: {}".format(i if not hash_id else hash_str2int(ck.metadata.get('entity').get( "id", ck.metadata.get('entity').get("doc_id")), 100))

        # 添加文档标题
        cnt += draw_node("Title", ck.metadata.get('entity').get( "filename", ck.metadata.get('entity').get("document_name")))

        # 添加URL（如果存在）
        cnt += draw_node("URL", ck.metadata.get('entity')['url']) if "url" in ck.metadata.get('entity') else ""

        # # 添加文档元数据字段
        # for k, v in docs.get(ck.get( "doc_id", "document_id"), {}).items():
        #     cnt += draw_node(k, v)

        # 添加内容部分
        cnt += "\n└── Content:\n"
        cnt += ck.metadata.get('entity').get( "content", ck.metadata.get('entity').get("content_with_weight"))
        cnt += ck.metadata.get('entity').get( "page_ids", " ")
        knowledges.append(cnt)

    return knowledges




@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    question = re.sub(r"（.*?）", "", last_query)
    retrieval_result = retrieval(question,embedding_model,compressor,milvus=milvus,collection_name=collection_name)
    knowledges = kb_prompt(retrieval_result, max_tokens)

    system_message = (
          f"""
    Role: You're a smart assistant. Your name is Miss R.
    Task: Summarize the information from knowledge bases and answer user's question.
    Requirements and restriction:
      - DO NOT make things up, especially for numbers.
      - If the information from knowledge is irrelevant with user's question, JUST SAY: Sorry, no relevant information provided.
      - Answer with markdown format text.
      - Answer in language of user's question.
      - DO NOT make things up, especially for numbers.

    ### Information from knowledge bases
    {knowledges}

    The above is information from knowledge bases.

    """

    )
    return system_message





class CustomAgentState(AgentState):
    user_id: str

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'
collection_name = 'new_milvus_collection_test'
embedding_model = None
milvus = None
chat_model = None
max_tokens = 4096
num_ctx = 4096
compressor = None
try:
    embedding_model = BGEM3EmbeddingFunction(model_name=r'D:\files\models\bge-m3',
                                             use_fp16=torch.cuda.is_available())
    reranker_model = HuggingFaceCrossEncoder(model_name=r"D:\files\models\bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    # DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
    checkpointer = InMemorySaver()
    milvus = MilvusServer(collection_name='new_milvus_collection_test')
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
    chat_model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
    )
    agent = create_agent(
            chat_model,
            tools=[],
            middleware=[prompt_with_context,
                        SummarizationMiddleware(
                            model="ollama:deepseek-r1:7b-qwen-distill-q4_K_M",
                            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
                            messages_to_keep=10,  # Keep last 20 messages after summary
                        )
                        ],
            checkpointer=checkpointer,
            state_schema=CustomAgentState,
        )
except Exception as e:
    print(f"初始化失败{e}")

def chat(question):
    try:
        # with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

        for token, metadata in agent.stream(
                {"messages": [{"role": "user", "content": question}],
                        "user_id": "user_123",
                 },

                {"configurable": {"thread_id": "1"}},
                stream_mode="messages",
        ):
            # print(f"node: {metadata['langgraph_node']}")

            # print("\n")
            if len(token.content_blocks) >0 and token.content_blocks[0].get('type','') == 'text':
                data = token.content_blocks[0]['text']
                yield f"data: {data}\n\n"
                time.sleep(0.05)
        yield "data: [DONE]\n\n"
        # agent.get_prompts()
        # display(Image(agent.get_graph().draw_mermaid_png()))
        img = agent.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(img)

    except Exception as e:
        print(f"回答失败{e}")
        yield f"data: [ERROR] {str(e)}\n\n"

# if __name__ == '__main__':
#
#     chat("在临床新技术实行分级分类准入管理制度中进行质量控制，项目组质控小组由什么人员组成？")
#     chat("刚刚我问了什么")