import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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


os.environ["OPENAI_API_KEY"] =  "***REMOVED***proj-aFCyCOP4O-COm1nGtAfC5d-zmtopk6F1wprDn06gShx_H8vRaLuF2QIGOc9fG0pog19SidfrGoT3BlbkFJ9BPMBiAiVhrY4EQclwOXTYVspLJmKUPRr6MuCk6F_NL6G_x7tc8MpHxUqnitnUFS0dO45nwtsA"

res = trim_messages(
    message,
    max_tokens= 40,
    strategy="last",
    token_counter = ChatOpenAI(model="gpt-4o"),
    include_system = True,
)

print(res)
