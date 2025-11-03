from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'

message = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("what do you call a speechless parrot"),
]

model = ChatOllama(
    model = model_name,
    base_url=base_url,
    reasoning=True,#是否启用思考模式

)

res = trim_messages(
    message,
    # 保留最近的消息
    strategy = "last",
    max_tokens= 45 , #最大保留的token
    token_counter = count_tokens_approximately,#计数方式
    start_on = "human",
    end_on = (HumanMessage,"tool"),
    include_system = True,#包含系统消息
    allow_partial = True,#允许拆分
)

print(res)