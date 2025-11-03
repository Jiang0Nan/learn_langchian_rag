from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 初始化 LLM
llm = init_chat_model(model="deepseek-chat", model_provider="deepseek")

# 构建 prompt（假设我们用最简单的模板）
prompt_template = ChatPromptTemplate.from_template("请将下列文本翻译成法语：今天天气真好")
chain = prompt_template | llm

# 加入对话历史
chat_history = InMemoryChatMessageHistory(messages=[
    HumanMessage(content="今天天气怎么样？"),
])

def get_history(session_id):
    return chat_history if session_id == "1" else InMemoryChatMessageHistory()

# 封装带对话历史的 chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
)

# 实际调用
res = chain_with_history.invoke(
    input={"text": "你好"},
    config={"configurable": {"session_id": "1"}},
)

print(res.content)

