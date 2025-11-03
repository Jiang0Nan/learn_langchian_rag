from lear_langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

model_init = init_chat_model(
    model = 'deepseek-r1:7b-qwen-distill-q4_K_M',
    model_provider = "ollama",
    base_url = "http://localhost:11434",
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = [ "streaming"],#配置可变参数
)

class Joke(BaseModel):
    setup : str = Field(description="这个笑话的问题")
    punchline: str = Field(description="这个笑话的回答")

query="给我讲一个有关下班的笑话"

parser = JsonOutputParser(pydantic_object=Joke)
# parser.get_format_instructions()用于根据Joke生成提示词模板:
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#   \n\nAs an example, for the schema {
#                   "properties":
#                       {"foo":
#                           {"title": "Foo",
#                           "description": "a list of strings",
#                           "type": "array",
#                           "items": {"type": "string"}
#                       }
#                      },
#                   "required": ["foo"]
#                   }\n
#                   the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#                   \n\nHere is the output schema:\n```\n{
#                               "properties":
#                                   {"setup":
#                                       {"description": "这个笑话的问题",
#                                       "title": "Setup",
#                                       "type": "string"},
#                                    "punchline": {"description": "这个笑话的回答",
#                                                   "title": "Punchline",
#                                                   "type": "string"}
#                                    },
#                              "required": ["setup", "punchline"]}\n```'

prompt = ChatPromptTemplate.from_template(
    """回答用户的问题。\n{format_instructions}\n{query}\n""",
    partial_variables ={"format_instructions":parser.get_format_instructions()}
)

chain = prompt | model_init | parser
joke_query = "给我讲一个笑话"
# print(chain.invoke({"query":joke_query}))

# for s in chain.stream({"query": joke_query}):
#     print(s)

# ============================第二种不定义Pydantic格式，只会返回模型的结构

parser_1 = JsonOutputParser()
prompt_2 = ChatPromptTemplate.from_template(
    """回答用户的问题。\n{format_instructions}\n{query}\n""",
    partial_variables ={"format_instructions":parser_1.get_format_instructions()}
)
chain_2 = prompt_2 | model_init | parser_1
for s in chain_2.stream({"query": joke_query}):
    print(s)



