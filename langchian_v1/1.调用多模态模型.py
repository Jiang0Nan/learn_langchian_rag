# 1.初始化模型
# 2. 编码图像
# 3. 构建多模态信息
import base64
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"


def get_chat_model(model_name: str = 'deepseek-chat', provider: str = "deepseek"):
    try:
        chat_model = init_chat_model(
            f"{provider}:{model_name}",
            temperature=0.7,
        )
        return chat_model
    except Exception as e:
        print(f"初始化模型失败{e}")
        return


def encode_image_to_base64(image_url: str):
    with open(image_url, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


image = encode_image_to_base64(r'D:\projects\learn_langchain_rag\langchian_v1\新版本练习\res\img.png')
message = HumanMessage(
    content=[
        {"type": "text", "text": "请分析这张图片，并分析他们的关系"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    ]
)
model = get_chat_model("qwen2.5vl:3b","ollama")

# 注释这里的[]原因在于他不支持单个message
response = model.invoke([message])
response.pretty_repr()
print("==================")
response.pretty_print()
print("==================")
print(response)