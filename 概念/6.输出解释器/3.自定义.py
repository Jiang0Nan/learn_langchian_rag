from lear_langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator


class Joke(BaseModel):
    setup : str = Field(description="这个笑话的问题")
    punchline: str = Field(description="这个笑话的回答")

    # 验证逻辑
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, value:dict)->dict:
        setup = value['setup']
        if setup and setup[-1] == '?':
            raise ValueError("得到的结果格式不对")
        return value

parser = PydanticOutputParser(pydantic_object = Joke)

prompt = ChatPromptTemplate.from_template(
    """回答用户的问题。\n{format_instructions}\n{query}\n""",
    partial_variables ={"format_instructions":parser.get_format_instructions()})

model_init = init_chat_model(
    model = 'deepseek-r1:7b-qwen-distill-q4_K_M',
    model_provider = "ollama",
    base_url = "http://localhost:11434",
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = [ "streaming"],#配置可变参数
)

chain = prompt | model_init
chain_2 = prompt | model_init | parser

chain.invoke({"query":"给我讲一个笑话"})

chain.invoke({"query":"什么是机器学习"})

# 如果无法流式输出可以使用SimpleJsonOutputParser
# json_parser = SimpleJsonOutputParser()
# json_chain = json_prompt | model | json_parser