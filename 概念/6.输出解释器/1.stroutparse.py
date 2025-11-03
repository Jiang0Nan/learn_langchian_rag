import os

from lear_langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool


@tool
def get_weather(location:str)->str:
    "Get the weather from a location."
    return "Sunny"


model_init = init_chat_model(
    model = 'deepseek-r1:7b-qwen-distill-q4_K_M',
    model_provider = "ollama",
    base_url = "http://localhost:11434",
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = [ "streaming"],#配置可变参数
)

if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
model = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek")
model_with_tool=  model.bind_tools([get_weather])

# 原本的输出
res = model_with_tool.invoke("What's the weather in San Francisco, CA?")
print(res)

#结构化输出
chain = model_with_tool | StrOutputParser()
res = chain.invoke("What's the weather in San Francisco, CA?")
print(res)

for chunk in chain.stream("What's the weather in San Francisco, CA?"):
    print(chunk, end="|")



