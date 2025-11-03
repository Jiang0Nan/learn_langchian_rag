#流程为：工具创建->绑定>调用>执行
from lear_langchain.chat_models import init_chat_model
from langchain_core.tools import tool

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

# @tool进行声明
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = model.bind_tools([multiply])

res = llm_with_tools.invoke("human:2multiply3的结果是什么")
print(res)
# 该模型不支持tool