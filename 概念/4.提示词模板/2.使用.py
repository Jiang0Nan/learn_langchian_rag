from gradio.themes.builder_app import history
from  lear_langchain.chat_models import  init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'

model = init_chat_model(model=model_name,base_url=base_url,model_provider="ollama")

# 1
prompt_one = \
    """
    请将下列文本翻译成{la}
    {text}
    """
prompt_template = ChatPromptTemplate.from_template(
    template=prompt_one,#模板，f_string {'a'} ，jinja2{{ 'a' }}  mustache {{'a'}}
    partial_variables={"la": "英文"},#预先填充的变量，后面也可以更改
    # template_format="f-string", #不是必填  jinja2可以写控制语句
)
filled_prompt_template = prompt_template.format_prompt(text="今天天气很糟糕")

# 2
prompt_message = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "你是一名全科医学专家,请根据提供的患者信息做出判断。回答必须专业、清晰。需要给出【诊断判断】【建议检查】【治疗建议】"),
        ("human", "患者：45岁男性，持续咳嗽2周，有痰。"),
        ("ai", "【诊断判断】\n可能为急性支气管炎。\n\n【建议检查】\n胸部X光片，痰液检查。\n\n【治疗建议】\n多饮水，止咳药物。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ],
    # template_format="f-string",
)
filled_prompt_message = prompt_message.format_messages(history=[HumanMessage("我胃痛"),AIMessage("【诊断判断】\n可能为急性支气管炎。\n\n【建议检查】\n胸部X光片，痰液检查。\n\n【治疗建议】\n多饮水，止咳药物。")],
                                                query="我精神萎靡")
# 3
from lear_langchain.prompts import PromptTemplate

template = "你是一名医生。请根据以下症状提供诊断建议：{symptoms}"

prompt_old = PromptTemplate(
    input_variables=["symptoms"],#老版本中显示指定变量
    template=template,
)

filled_prompt = prompt_old.format(symptoms="咳嗽，发热，头痛")

# 方4
# 1.
template_format = "你是一名医生。请根据以下症状提供诊断建议：{}".format("头痛")
# 2.位置参数
template_address = "将{0}翻译成{1}语言".format("hi","法语")
# 3.关键参数
template_key = "将{text}翻译成{la}语言".format(text="hi",la="法语")
#4.字典
text= {"text":"hi","la":"法语"}
template_dic = "将{text}翻译成{la}语言".format(**text)


prompt=prompt_old

chain =( prompt | model).with_config(configurable = { "temperature":"0.8"})

# ========================调用方法1
# 方1的调用
# for i in chain.stream({"text":"今天天气真糟糕，衣服都湿透了"}):
#     print(i.content, end="", flush=True)
# 方2的调用
inputs = {
    "history":
        [HumanMessage(content="我胃痛"),
         AIMessage(content="【诊断判断】\n可能为急性支气管炎。\n\n【建议检查】\n胸部X光片，痰液检查。\n\n【治疗建议】\n多饮水，止咳药物。")],
     "query":"我精神萎靡"
    }
# 检查参数是否传入成功
# for m in prompt.format_messages(**inputs):
#     print(f"{m.type}: {m.content}")

# for i in chain.stream(inputs):
#     print(i.content, end="", flush=True)
# 方3的调用
# for i in chain.stream({"symptoms":"咳嗽，发热，头痛"}):
#     print(i.content, end="", flush=True)


# =========调用的方法2
# 方1
# for i in model.stream(filled_prompt_template):
#     print(i.content, end="", flush=True)
# # 方2
# for i in model.stream(filled_prompt_message):
#     print(i.content, end="", flush=True)
# # 方3
# for i in model.stream(filled_prompt):
#     print(i.content, end="", flush=True)
# #方4
for i in model.stream(template_address):
    print(i.content, end="", flush=True)