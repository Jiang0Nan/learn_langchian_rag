from langchain_core.prompts import ChatPromptTemplate, FewShotPromptWithTemplates, FewShotChatMessagePromptTemplate, \
    FewShotPromptTemplate, PromptTemplate, StringPromptTemplate
from langchain_ollama import ChatOllama

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'


model_ollama = ChatOllama(
    model = model_name,
    base_url=base_url,
    reasoning=True,#是否启用思考模式
    temperature = 0.8,
)

#====================================1.没有
print(model_ollama.invoke("What is 2 🦜 9?"))

# 使用few-shot
examples = [
    {"input":"2 🦜 2", "output": "4"},
    {"input":"3 🦜 2", "output": "5"}
]
# =================================2. 方法1
prompt_1 = PromptTemplate.from_template(
    "根据{input}',得到{output}"
)

few_shot_prompts_1 = FewShotPromptWithTemplates(
    examples =examples, # Optional[list[dict]] = None 提供的样例，这个和example_selector必选一个

    # example_selector #: Any = None 示例选择器，用于选择要格式化到提示中的示例。要么提供这个，要么提供例子。

    example_prompt = prompt_1,# PromptTemplate  需要的提示模板


    suffix= PromptTemplate.from_template("根据{input}，结果是多少？"),# StringPromptTemplate 样例后的主问题，通常是“用户真正要问的问题”
    input_variables = ["input"], #必须显示指明参数
    # example_separator=# str = "\n\n" 用于链接prefix,examples,suffix的分割符

    prefix =PromptTemplate.from_template("请根据以下示例回答问题："),#: Optional[StringPromptTemplate] = None 放在example的前面 例如：“请根据以下示例回答问题：”

    # template_format= , PromptTemplateFormat  提示模板使用的格式 'f-string', 'jinja2', 'mustache'."""

    # validate_template= , bool = False  是否提前验证模板变量匹配（可关闭避免开发期报错）

)
finale_template_1 = few_shot_prompts_1.format_prompt(input = "4 🦜 4")
model_ollama.invoke(finale_template_1)
# (few_shot_prompts_1 | model_ollama).invoke({"input":"What is 2 🦜 9?"})


# =============================3.方法3
prompt_2 = ChatPromptTemplate.from_template(
    "根据{input}',得到{output}"
)

few_shot_prompt_2 = FewShotChatMessagePromptTemplate(
    examples = examples,
    example_prompt = prompt_2,
    # input_variables=[],
    # input_types={},
    # partial_variables={},
)

print(few_shot_prompt_2.invoke({}).to_messages())

final_prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system","content":"你是一个智能助手" },
        few_shot_prompt_2,
        {"role": "human","content":"{input}" },

     ]
)
chain = final_prompt | model_ollama
for i in chain.stream({"input":"What is 2 🦜 9?"}):
    print(i.content,end=" ",flush=True)