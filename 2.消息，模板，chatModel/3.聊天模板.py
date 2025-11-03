import os

from lear_langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

model = init_chat_model(
    model="deepseek-r1:7b-qwen-distill-q4_K_M", model_provider="ollama",
    base_url="http://localhost:11434")

prompt = \
    """
    请将下列文本翻译成{la}
    {text}
    """
# prompt 如果是jinja2的话可以写控制语句例如：
# {% if language == "英文" %}
# Please translate the following text into English:
# {% else %}
# 请将下列文本翻译成中文：
# {% endif %}
#
# {{ text }}

prompt_template = ChatPromptTemplate.from_template(
    template=prompt,#模板，f_string {'a'} ，jinja2{{ 'a' }}  mustache {{'a'}}
    partial_variables={"la": "英文"},#预先填充的变量，后面也可以更改
    # template_format="f-string", #不是必填  jinja2可以写控制语句

)

# few-shot 给一个输出案例
prompt_message = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "你是一名全科医学专家,请根据提供的患者信息做出判断。回答必须专业、清晰。"),
        ("human", "患者：45岁男性，持续咳嗽2周，有痰。"),
        ("ai", "【诊断判断】\n可能为急性支气管炎。\n\n【建议检查】\n胸部X光片，痰液检查。\n\n【治疗建议】\n多饮水，止咳药物。"),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{query}"),
    ]
    # template_format="f-string",
)

print(prompt_template)
print(prompt_message)
# input_variables=['la']
# input_types={}
# partial_variables={}
# messages=[
#   HumanMessagePromptTemplate(
#       prompt=PromptTemplate(
#           input_variables=['la'],
#           input_types={},
#           partial_variables={},
#           template='\n将下列文本从中文翻译成{la}\n'),
#           additional_kwargs={})]
# print(prompt_template.format_prompt())
# messages=[HumanMessage(content='\n将下列文本从中文翻译成英文\n', additional_kwargs={}, response_metadata={})]
print(prompt_template.format_prompt(la="法语", text="hi"))
# messages=[HumanMessage(content='\n将下列文本从中文翻译成法语\n', additional_kwargs={}, response_metadata={})]
print(prompt.format(la="法语", text="hi"))
# Human:
# 将下列文本从中文翻译成法语
print(prompt_template.format_messages(la="法语", text="hi"))
# [HumanMessage(content='\n将下列文本从中文翻译成法语\n', additional_kwargs={}, response_metadata={})]
out = prompt_template.format_prompt(la="法语", text="hi")
print(out.to_string())

output = model.invoke(prompt_template.format_messages(text="好的"))
print(f"输出结果为{output.content}")

# =============方法2 字符串提示模板
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt_template.invoke({"topic": "cats"})
# =============使用方式3
# prompt2 = ChatPromptTemplate.from_messages(
#
# )

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt_template.invoke({"topic": "cats"})


# ==============方法4  消息占位符
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

# Simple example with one message
prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})

# More complex example with conversation history
messages_to_pass = [
    HumanMessage(content="What's the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="And what about Germany?")
]

formatted_prompt = prompt_template.invoke({"msgs": messages_to_pass})
print(formatted_prompt)

# -==============方法5 老版本中 用于普通模型
from lear_langchain.prompts import PromptTemplate

template = "你是一名医生。请根据以下症状提供诊断建议：{symptoms}"

prompt = PromptTemplate(
    input_variables=["symptoms"],#老版本中显示指定变量
    template=template,
)

filled_prompt = prompt.format(symptoms="咳嗽，发热，头痛")
print(filled_prompt)
