from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages, BaseMessage, ToolMessage

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

from FlagEmbedding import BGEM3FlagModel

embedding = BGEM3FlagModel(model_name_or_path = r"D:\files\models\bge-m3", use_fp16=False)



def  str_encode(text:str,embedding):
    embed_res = embedding.tokenizer(
        text,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=True
    )
    return len(embed_res['input_ids'])


def token_counter(messages:list[BaseMessage]):
    # 这两个初始值根据模型来设置，具体的暂时没搞懂。后面再看
    num_len = 3
    tokens_per_message = 3
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            raise  f"有不支持的类型{message.__class__}"
        num_len += (
                tokens_per_message+str_encode(message.content,embedding) + str_encode(role,embedding)
        )
res = trim_messages(
    message,
    # 保留最近的消息
    strategy = "last",
    max_tokens= 45 ,
    token_counter = token_counter,
    start_on = "human",
    end_on = (HumanMessage,"tool"),
    include_system = True,
    allow_partial = True,
)

print(res)