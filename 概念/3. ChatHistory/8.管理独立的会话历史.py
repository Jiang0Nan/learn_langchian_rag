import os

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory


if not os.environ.get("DEEPSEEK_API_KEY"):
     os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek")

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("hhhhh"),
]
# 初始聊天历史
chat_history = InMemoryChatMessageHistory(messages=messages[:-1])


def dummy_get_session_history(session_id):
    if session_id != "1":
        return InMemoryChatMessageHistory()
    return chat_history



trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=llm,
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    # start_on="human" makes sure we produce a valid chat history
    start_on="human",
)

chain = trimmer | llm
chain_with_history = RunnableWithMessageHistory(chain, dummy_get_session_history)
chain_with_history.invoke(
    [HumanMessage("what do you call a speechless parrot")],
    config={"configurable": {"session_id": "1"}},
)