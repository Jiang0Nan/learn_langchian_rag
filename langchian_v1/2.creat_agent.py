#1. 联网检索工具
# 2. 初始化模型
# 3.创建基于agent
import json
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, ToolMessage

if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"


search_tool = DuckDuckGoSearchRun()

model = init_chat_model("deepseek:deepseek-chat")

agent = create_agent(
    model,
    tools=[search_tool],
    system_prompt="你是一个多功能助手",
)

result = agent.invoke({"messages":[{"role":"user","content":"LangCahin1.0和Langrah的关系是什么？请提供最新消息"}]})
print(result)
print("==========")

msgs = result['messages']
tool_calls = []
tool_results = []
for msg in msgs:
    # 收集所有工具调用
    if isinstance(msg,AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append({"id":tc["id"],"name":tc["name"],"args":tc["args"]})

    # 收集所有工具结果
    if isinstance(msg,ToolMessage):
        tool_results.append({
            "id":msg.tool_call_id,
            "name":getattr(msg,"name",""),
            "content":getattr(msg,"content",""),
            "artifact":getattr(msg,"artifact",None),
        })
# 关联"调用->结果“
by_id = {r["id"]:r for r in tool_results}

paired = [ {"tool":c["name"],"args":c["args"],"result":by_id.get(c["id"] or {}).get("content")} for c in tool_calls]
print(json.dumps(paired,ensure_ascii=False,indent=2))