from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import ConfigurableFieldSpec

# 初始化模型
llm = init_chat_model(
    model="deepseek-r1:7b-qwen-distill-q4_K_M",
    base_url="http://localhost:11434",
    model_provider="ollama"
)

# prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是机器人 {bot_name}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm

# 历史存储
store = {}


def get_history(user_id, conversation_id, bot_name):
    key = f"{user_id}_{conversation_id}_{bot_name}"
    if key not in store:
        store[key] = InMemoryChatMessageHistory()
    return store[key]


# 🔹 消息裁剪 + 摘要生成
def trim_and_summarize(history: InMemoryChatMessageHistory, max_full_messages=3):
    """
    - 保留最近 max_full_messages 条完整消息
    - 旧消息生成摘要替代
    """
    messages = history.messages
    if len(messages) <= max_full_messages:
        return history

    old_messages = messages[:-max_full_messages]
    new_messages = messages[-max_full_messages:]

    # 将旧消息合并成摘要
    old_text = "\n".join([f"{m.type}: {m.content}" for m in old_messages])
    summary_prompt = f"""你是一名对话摘要助手，请将以下历史对话压缩成一句话或几句话，保留重要信息并删除冗余内容。
对话历史：
{old_text}
请用简洁自然的语言总结上面对话的核心内容："""

    summary_content = llm.invoke(summary_prompt)
        # llm([HumanMessage(content=summary_prompt)]).content

    # 用摘要替换旧消息
    summary_message = AIMessage(content=f"[历史摘要] {summary_content}")
    history.messages = [summary_message] + new_messages


# 包装 RunnableWithMessageHistory，自动裁剪
def make_chain(bot_name):
    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_history,
        input_messages_key="input",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, default="u1"),
            ConfigurableFieldSpec(id="conversation_id", annotation=str, default="c1"),
            ConfigurableFieldSpec(id="bot_name", annotation=str, default=bot_name),
        ]
    )


# 示例
chain_a = make_chain("A")

# 模拟对话
history_a = get_history("u1", "c1", "A")
for user_input in ["你好", "我今天心情不好", "工作压力很大", "有什么缓解方法？"]:
    trim_and_summarize(history_a, max_full_messages=2)  # 保留最近 2 条完整消息，其余摘要化
    res = chain_a.invoke({"input": user_input, "bot_name": "A"}, config={
        "configurable": {"user_id": "u1", "conversation_id": "c1", "bot_name": "A"}
    })
    print(res.content)

# 查看裁剪后历史
for msg in history_a.messages:
    print(f"{msg.type}: {msg.content}")
