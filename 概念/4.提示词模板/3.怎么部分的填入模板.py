# ============使用字符串值进行部分格式化。
# 方法1
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("将{text}翻译成{la}")
partial_prompt = prompt.partial(la="英语")
print(partial_prompt.format_prompt(text="hi"))

# 方法2
prompt = PromptTemplate(
    template = "将{text}翻译成{la}",
    input_variables = ["la"],#告诉模型后续取值
    partial_variables ={"text":"今天天气很好"},#预填入
)
print(prompt.format(la="法语"))

# ===========使用返回字符串值的函数进行部分格式化。

# 方法1
from datetime import datetime


def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt  = PromptTemplate(
    template="将{text}翻译成{la} time--{time}",
    input_variables = ["la"],#告诉模型后续取值
)

prompt = prompt.partial(time=_get_datetime)
print(prompt.format(la="英语",text="今天是个好天气"))

prompt  = PromptTemplate(
    template="将{text}翻译成{la} time--{time}",
    input_variables = ["la","text"],#告诉模型后续取值
    partial_variables = {"time":_get_datetime}
)
print(prompt.format(la="法语",text="今天是个好天气"))
