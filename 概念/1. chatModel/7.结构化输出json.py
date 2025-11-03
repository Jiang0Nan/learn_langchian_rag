from lear_langchain.chat_models import init_chat_model

json_schema= {
    "title":"joke",
    "description":"Joke to tell user",
    "type":"object",
    "properties":{
        "setup": {
            "type": "string",
            "description": "有关这个笑话的设置",
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
model_init = init_chat_model(
    model = 'deepseek-r1:7b-qwen-distill-q4_K_M',
    model_provider = "ollama",
    base_url = "http://localhost:11434",
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = [ "streaming"],#配置可变参数
    streaming = True,
)

struct_llm = model_init.with_structured_output(json_schema)
print(struct_llm.invoke("开一个关于上班的玩笑"))