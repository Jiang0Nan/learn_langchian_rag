from lear_langchain.chat_models import ChatOpenAI, init_chat_model
from lear_langchain.memory import ConversationSummaryMemory
from lear_langchain.prompts import PromptTemplate

# 1️⃣ 初始化聊天模型
chat_model = init_chat_model(model="deepseek-r1:7b-qwen-distill-q4_K_M", base_url="http://localhost:11434",model_provider="ollama")


# 2️⃣ 定义摘要模板
summary_prompt = PromptTemplate(
    input_variables=["summary", "new_message"],
    template=(
        "已有对话摘要：\n{summary}\n\n"
        "新对话内容：\n{new_message}\n\n"
        "请基于新消息更新摘要，保留关键信息并简明扼要："
    )
)

# 3️⃣ 初始化 ConversationSummaryMemory
memory = ConversationSummaryMemory(
    llm=chat_model,
    prompt=summary_prompt,
    memory_key="summary",  # 内存存储摘要的键
    input_key="input"
)


# 4️⃣ 模拟多轮对话
def chat(user_input):
    # 使用 memory 自动更新摘要并生成回答
    messages = memory.load_memory_variables({})  # 获取当前摘要
    summary = messages.get("summary", "")

    # 生成回答（可直接使用 summary + 新消息）
    response = chat_model([
        {"role": "system", "content": f"当前对话摘要：{summary}"},
        {"role": "user", "content": user_input}
    ])

    # 更新 memory（摘要自动更新）
    memory.save_context({"input": user_input}, {"output": response.content})

    return response.content


# 5️⃣ 模拟对话
print(chat("你好，我今天心情不好"))
print(chat("我最近工作压力很大"))
print(chat("有什么方法缓解压力吗？"))
print(chat("能推荐几个放松的活动吗？"))

# 6️⃣ 查看当前摘要
print("当前摘要：")
print(memory.load_memory_variables({})["summary"])
