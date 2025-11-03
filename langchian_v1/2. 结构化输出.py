from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
class Weather(BaseModel):
    # 温度
    temperature : float
    # 经纬度
    location : str

def weather_tool(city:str)->str:
    """通过city获取天气信息"""
    return f"{city}的温度为12°，天气晴朗,不提供地理位置"



agent = create_agent(
    "deepseek:deepseek-chat",
    tools=[weather_tool],
    response_format=ToolStrategy(Weather))


result = agent.invoke({
    "messages":[{"role":"user","content":"北京的天气怎么样"}]

})

print(repr(result['structured_response']))