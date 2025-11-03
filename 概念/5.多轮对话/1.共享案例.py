from langchain.chat_models import  init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.chat_history import InMemoryChatMessageHistory
base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'

model = init_chat_model(model=model_name,base_url=base_url,model_provider="ollama")

prompt_str = """
你是一名全科医学专家,请根据提供的患者信息`query`做出判断。回答必须专业、清晰。需要给出【诊断判断】【建议检查】【治疗建议】
例如：
     患者：45岁男性，持续咳嗽2周，有痰。
    你给出的诊断形如：【诊断判断】\n可能为急性支气管炎。\n\n【建议检查】\n胸部X光片，痰液检查。\n\n【治疗建议】\n多饮水，止咳药物。

用户提出的问题：
{query}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
chain = prompt | model

store={}
def get_session_history(
        user_id: str, conversation_id: str
):
   if user_id not in store:
        store[(user_id, conversation_id)] = InMemoryChatMessageHistory()
   return store[(user_id, conversation_id)]

with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history=get_session_history,
                input_messages_key="question",
                history_messages_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="user_id",
                        annotation=str,  # 类型
                        name="User ID",  # 可视化用的字段名（非必须）
                        description="Unique identifier for the user.",
                        default="guest",  # 默认值
                        is_shared=True  # 是否是共享的（多个 Chain 中复用）
                    ),
                    ConfigurableFieldSpec(
                        id="conversation_id",
                        annotation=str,
                        name="Conversation ID",
                        description="Unique identifier for the conversation.",
                        default="",
                        is_shared=True,
                    ),
                ],
            )

res = with_message_history.invoke(
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"user_id": "123", "conversation_id": "1"}}
            )

print(res)
