from lear_langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# 还有很多其他值得方式，形式，这里暂时不讨论后续再看https://python.langchain.com/docs/how_to/functions/
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'

model = init_chat_model(model=model_name,base_url=base_url,model_provider="ollama")

chain = prompt | model | StrOutputParser()
# chain.invoke({"topic": "bears"})

# 另外一个chain用于判断这个笑话时候好笑
analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")

composed_chain = {"joke": chain} | analysis_prompt | model | StrOutputParser()

for i in composed_chain.stream({"topic": "bears"}):
    print(i,end=" ",flush=False)
