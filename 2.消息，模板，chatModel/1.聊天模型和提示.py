from langchain_core.messages import HumanMessage, SystemMessage
from  langchain_core.prompts import ChatPromptTemplate
from lear_langchain.chat_models import init_chat_model
import  os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
# if not os.environ.get("DEEPSEEK_API_KEY"):
#     os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
model = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek")

message = [
    SystemMessage("请将语言翻译成英文"),
    HumanMessage("今天天气真好")

]
#-------------直接获取模型回答结果（不是流式输出）-----------
# response = model.invoke(message)
# print(response)

#-------------直接获取模型回答结果（流式输出）-----------

for token in model.stream(message):
    print(token.content,end="|")


