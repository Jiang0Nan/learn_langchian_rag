from typing import Optional

from lear_langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated


class Jock(TypedDict):
    """Jock to user"""
    setup : Annotated[str,...,"the setup of the joke"]

    punchline : Annotated[str,...,"The punchline of the joke"]

    rating : Annotated[Optional[int],None,"the rating of the joke"]

model_init = init_chat_model(
    model = 'deepseek-r1:7b-qwen-distill-q4_K_M',
    model_provider = "ollama",
    base_url = "http://localhost:11434",
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = [ "streaming"],#配置可变参数
    streaming = True,
)

struct_llm = model_init.with_structured_output(Jock)

for i in struct_llm.stream("将一个关于上班的笑话"):
    print(i,end=" \n",flush=True)