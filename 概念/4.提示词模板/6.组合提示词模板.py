from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PipelinePromptTemplate, SystemMessagePromptTemplate, \
    PromptTemplate

full_temple = ChatPromptTemplate.from_template("""
{introduction}
{example}
{start}
"""

)

introduction_temple = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate(prompt=PromptTemplate.from_template("你扮演的角色是{role}"))]

)

example_temple = ChatPromptTemplate.from_messages(
    [("human","这里有一些交互例子:问题：{question}回答：{answer}")]
)

start_template = ChatPromptTemplate.from_messages([("human","现在认真得的开始吧！Q:{input} A:")])

final_prompt = [
    ("introduction",introduction_temple),
    ("example",example_temple),
    ("start",start_template)
]
# 这个方法不和是，没有检测到变量
# pipeline_prompt_template = PipelinePromptTemplate(
#     final_prompt = full_temple,pipeline_prompts=final_prompt
# )
#
# print(pipeline_prompt_template.input_variables)

# =============上述PipelinePromptTemplate链接不建议使用
def build_prompt(role, question, answer, input):
    return introduction_temple.format_messages(role=role) + \
           example_temple.format_messages(question=question, answer=answer) + \
           start_template.format_messages(input=input)

# 示例输入
messages = build_prompt(
    role="一名智能机器人",
    question="你喜欢什么车？",
    answer="我喜欢电动车，比如特斯拉。",
    input="你最喜欢哪个社交媒体平台？"
)

# 打印结果
for msg in messages:
    print(f"[{msg.type}] {msg.content}")


# ==========================================================================
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PipelinePromptTemplate

from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """Here's an example of an interaction:

Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

print(pipeline_prompt.input_variables)

print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)

#  由于PipelinePromptTemplate弃用，因此使用
def build_prompt_2(person, example_q, example_a, input):
    return "\n\n".join([
        introduction_prompt.format(person=person),
        example_prompt.format(example_q=example_q, example_a=example_a),
        start_prompt.format(input=input),
    ])

prompt_text = build_prompt_2(
    person="Elon Musk",
    example_q="What's your favorite car?",
    example_a="Tesla",
    input="What's your favorite social media site?",
)

print(prompt_text)




