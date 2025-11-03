from lear_langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from lear_langchain.prompts import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

examples = [
    # 数学推理
    {
        "question": "小明的妈妈给了他10块钱去买文具，如果一支笔3块钱，小明最多能买几支笔？",
        "answer": "小明有10块钱，每支笔3块钱，所以他最多能买3支笔，因为3*3=9，剩下1块钱不够再买一支笔。因此答案是3支。"
    },
    {
        "question": "一个篮球队有12名球员，如果教练想分成两个小组进行训练，每组需要有多少人？",
        "answer": "篮球队总共有12名球员，分成两个小组，每组有12/2=6名球员。因此每组需要有6人。"
    },
    # 逻辑推理
    {
        "question": "如果所有的猫都怕水，而Tom是一只猫，请问Tom怕水吗？",
        "answer": "根据题意，所有的猫都怕水，因此作为一只猫的Tom也会怕水。所以答案是肯定的，Tom怕水。"
    },
    {
        "question": "在夏天，如果白天温度高于30度，夜晚就会很凉爽。今天白天温度是32度，请问今晚会凉爽吗？",
        "answer": "根据题意，只要白天温度高于30度，夜晚就会很凉爽。今天白天的温度是32度，超过了30度，因此今晚会凉爽。"
    },
    # 常识问题
    {
        "question": "地球绕太阳转一圈需要多久？",
        "answer": "地球绕太阳转一圈大约需要365天，也就是一年的时间。"
    },
    {
        "question": "水的沸点是多少摄氏度？",
        "answer": "水的沸点是100摄氏度。"
    },
    # 文化常识
    {
        "question": "中国的首都是哪里？",
        "answer": "中国的首都是北京。"
    },
    {
        "question": "世界上最长的河流是哪一条？",
        "answer": "世界上最长的河流是尼罗河。"
    },
]

# - example_selector ：负责为给定输入选择少数样本（以及它们返回的顺序）。它们实现了 BaseExampleSelector 接口。一个常见的例子是向量存储支持的 SemanticSimilarityExampleSelector
# - example_prompt ：通过其 format_messages 方法将每个示例转换为 1 条或多条消息。一个常见的示例是将每个示例转换为一条人工消息和一条人工智能消息响应，或者一条人工消息后跟一条函数调用消息。

to_vectorize = [" ".join(example.values()) for example in examples]

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

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'


model_ollama = ChatOllama(
    model = model_name,
    base_url=base_url,
    reasoning=True,#是否启用思考模式
    temperature = 0.8,
)
# 存储
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,#VectorStore 存储样例的数据库
    k=2,#选择的样例
    # example_keys:# Optional[list[str]] = None "可选，用于过滤示例，只考虑指定的 key
    # input_keys: # Optional[list[str]] = None    可选，指定输入变量 key，只根据这些变量做相似度搜
    # vectorstore_kwargs: #Optional[dict[str, Any]] = None 传递给 vectorstore.similarity_search 的额外参数，比如 score_threshold
    #
    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True,
    #     extra="forbid",
    # )
)
example_selector.select_examples({"input": "罗杰有五个网球，他又买了两盒网球，每盒有3个网球，请问他现在总共有多少个网球？"})

example = ChatPromptTemplate.from_messages([("human","{question}"),("ai","answer")])
few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt = example,
    input_variables = ["input"],
)
print(few_shot_chat_prompt.format(input="罗杰有五个网球，他又买了两盒网球，每盒有3个网球，请问他现在总共有多少个网球？"))
print(few_shot_chat_prompt.format(input="月亮每天什么时候出现"))

final_chat_prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是一个无所不能的人，无论什么问题都可以回答。"},
        few_shot_chat_prompt,
        {"role":"user","content":"{input}"}]
)
chain = (final_chat_prompt | model_ollama).invoke({"input":"月亮每天什么时候出现"})

