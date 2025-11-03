from dataclasses import dataclass
from lib2to3.fixes.fix_input import context

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
import os
import json

from pydantic import BaseModel

os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"


# @dataclass
class ContextSchema(BaseModel):
    user_name: str


@tool
def get_user_email(runtime: ToolRuntime[ContextSchema]):
    """获取用户的邮箱信息"""

    return f"{runtime.context.user_name}的邮箱为3386526336@qq.com"


agent = create_agent(
    "deepseek:deepseek-chat",
    tools=[get_user_email],
    context_schema = ContextSchema
)

re = agent.invoke(
    {"messages": [{"role": "user", "content": "hi!我的邮箱是多少"}]},
    context=ContextSchema(user_name="John Smith")
)
print(re)

