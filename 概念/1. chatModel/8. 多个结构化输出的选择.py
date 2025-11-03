from typing import Union, Optional

from lear_langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

# =====================================================
# 官方建议，如果模型支持工具，推荐使用工具进行切换，具体参考https://python.langchain.com/docs/how_to/tool_calling/
# =====================================================
class ResponseFormat(BaseModel):
    "始终使用这个工具来结构化你的回答给用户"
    answer:str = Field(description="用于回答用户提出的问题，内容需准确、完整" )
    followup_question:list = Field(description="从用户视角提出接下来可能关心的3个问题，需贴合主题,")

json_schema= {
    "title":"joke",
    "description":"Joke to tell user",
    "type":"object",
    "properties":{
        "setup": {
            "type": "string",
            "description": "有关这个笑话的问题",
        },
        "punchline": {
            "type": "string",
            "description": "这个笑话的笑点",
        },
        "rating": {
            "type": "integer",
            "description": "这个笑话好笑程度，从1到10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}

class Jock(TypedDict):
    """Jock to user"""
    setup : Annotated[str,...,"the setup of the joke"]

    punchline : Annotated[str,...,"The punchline of the joke"]

    rating : Annotated[Optional[int],None,"the rating of the joke"]

class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


model_init = init_chat_model(
    model = 'deepseek-r1:7b-qwen-distill-q4_K_M',
    model_provider = "ollama",
    base_url = "http://localhost:11434",
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = [ "streaming"],#配置可变参数
    streaming = True,
)

class FinalResponse(BaseModel):
    final_output : Union[ResponseFormat,Jock]

model_init.with_structured_output(FinalResponse)
print(model_init.invoke("给我讲一个有关英文的笑话"))