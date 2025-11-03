from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import ConfigurableFieldSpec

# 模拟模型
llm = init_chat_model(model="deepseek-r1:7b-qwen-distill-q4_K_M", base_url="http://localhost:11434",model_provider="ollama")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是机器人 {bot_name}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm

# 用于保存历史
store = {}

def get_history(user_id, conversation_id,bot_name):
    key = f"{user_id}_{conversation_id}_{bot_name}"
    if key not in store:
        store[key] = InMemoryChatMessageHistory()
    return store[key]

# A 机器人，不共享字段
chain_a = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="user_id", annotation=str, default="u1"),
        ConfigurableFieldSpec(id="conversation_id", annotation=str, default="c1"),
        ConfigurableFieldSpec(id="bot_name", annotation=str, default="A"),
    ]
)

# B 机器人，不共享字段
chain_b = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="user_id", annotation=str, default="u1"),
        ConfigurableFieldSpec(id="conversation_id", annotation=str, default="c1"),
        ConfigurableFieldSpec(id="bot_name", annotation=str, default="B"),
    ]
)

# 测试：两个 chain 虽然 config 一样，但不会共享字段
config = {
    "configurable": {
        "user_id": "u1",
        "conversation_id": "c1",
        "bot_name": "A"
    }
}
config2 = {
    "configurable": {
        "user_id": "u1",
        "conversation_id": "c1",
        "bot_name": "B"
    }
}

print("=== A 说话 ===")
res_a = chain_a.invoke({"input": "你好","bot_name": "A"}, config=config)
print(res_a.content)

print("=== B 说话（没记住A说过什么）===")
res_b = chain_b.invoke({"input": "你记得我刚刚说什么了吗？","bot_name": "B"}, config=config2)
print(res_b.content)
