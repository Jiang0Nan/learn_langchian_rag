from lear_langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import  os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
model = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek")

class ResponseFormat(BaseModel):
    "始终使用这个工具来结构化你的回答给用户"
    answer:str = Field(description="用于回答用户提出的问题，内容需准确、完整" )
    followup_question:list = Field(description="从用户视角提出接下来可能关心的3个问题，需贴合主题,")

# 方法1
model_with_tool = model.bind_tools([ResponseFormat])

res = model_with_tool.invoke([HumanMessage("什么是AIGC")])
print(res.tool_calls[0]["args"])
print( ResponseFormat.model_validate(res.tool_calls[0]["args"]))

#  方法2
model_with_structure = model.with_structured_output(ResponseFormat)
res = model_with_structure.invoke("什么是机器学习")
print(res)
