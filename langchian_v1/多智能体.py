import operator
import os
from typing import Annotated

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send

from pydantic import Field, BaseModel
from typing_extensions import TypedDict

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "learn-langchain"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

#===================== 1. 定义输入 输出状态==========================
class InputState(TypedDict):
    """question:分解的子问题"""
    question : str

class OutputState(TypedDict):
    source: str
    result:str

class ClassificationState(TypedDict):
    source: str # 哪一个agent
    question: list[str] # 处理的问题有哪些

class RouterState(TypedDict):
    question: str
    classifications: list[ClassificationState]
    results:Annotated[list[OutputState],operator.add]
    final_answer:str

# =============================2. 定义工具==============================
@tool
def search_code(question:str,repo:str = "main")-> str:
    """在代码库中搜索相关代码片段"""
    return f"在{repo}中发现与‘{repo}’匹配的代码：SRC/auth.py中的身份验证中间件"

@tool
def search_issues(question:str)-> str:
    """搜索GitHub问题并拉取请求。"""
    return f"找到3个与“{question}”匹配的问题:#142 (API授权文档)、#89 (OAuth流)、#203(令牌刷新)"

@tool
def search_prs(query: str) -> str:
    """搜索拉取请求以获取实现细节。"""
    return f"PR #156 添加了 JWT 认证，PR #178 更新了 OAuth 范围"

@tool
def search_notion(question:str)-> str:
    """在Notion工作区中搜索文档。"""
    return f"找到文档：'API 认证指南' - 涵盖 OAuth2 流程、API 密钥和 JWT 令牌"


@tool
def get_page(page_id: str) -> str:
    """通过ID获取特定的Notion页面。"""
    return f"页面内容：逐步认证设置说明"


@tool
def search_slack(query: str) -> str:
    """搜索 Slack 消息和线程。"""
    return f"在 #engineering 中找到讨论：'使用 Bearer 令牌进行 API 身份验证，请参阅文档了解刷新流程'"

@tool
def get_thread(thread_id: str) -> str:
    """获取特定的 Slack 线程。"""
    return f"获取特定的 Slack 线程。"
# =======================3. 定义智能体 （每个独立的agent）=============


model = init_chat_model(
    "deepseek-chat",
    configurable_fields = ["temperature"], # 运行中可配置的参数列表 “Any”为都可以配置
    max_retries=3,
    base_url = "https://api.deepseek.com"
)
router_llm = init_chat_model(
"deepseek-chat",
    configurable_fields = ["temperature"], # 运行中可配置的参数列表 “Any”为都可以配置
    max_retries=3,
    base_url = "https://api.deepseek.com"
)

# 分流agent
github_agent = create_agent(
    model=model,
    tools=[search_code,search_issues,search_prs],
    system_prompt= "“您是GitHub专家。通过搜索仓库、问题和拉取请求，回答有关代码、API参考和实现细节的问题。”",
)


notion_agent = create_agent(
    model=model,
    tools=[search_notion],
    system_prompt=( "您是Notion专家。"
                    "通过搜索组织的Notion工作区，回答有关内部流程、政策和团队文档的问题。" ),
)

slack_agent = create_agent(
    model=model,
    tools=[search_slack,get_thread],
    system_prompt="您是Slack专家。通过搜索相关线程和讨论，回答问题，这些线程和讨论中团队成员分享了知识和解决方案。"
)

# ==========================4. 组装agent====================
# 先分任务，在每个执行，最后合并，得到最终结果
# a. 分任务
class ClassificationResult(BaseModel):
    """将用户查询分类为特定代理子问题的结果。"""
    classifications:list[ClassificationState]=Field(description="要调用的代理及其目标子问题列表")

def classify_question(state:RouterState)-> dict:
    """对用户查询进行分类"""
    struct_llm = router_llm.with_structured_output(ClassificationResult)

    result = struct_llm.invoke([
        {
            "role": "system",
            "content": """分析此查询并确定要咨询的知识库。
对于每个相关来源，生成一个针对该来源优化的子问题。

可用来源：
- github: 代码、API 参考、实现细节、问题、拉取请求
- notion: 内部文档、流程、政策、团队维基
- slack: 团队讨论、非正式知识共享、最近对话

仅返回与查询相关的来源。每个来源应有一个针对该特定知识领域的优化子问题。

例如 "如何认证 API 请求？":
- github: "存在哪些认证代码？搜索认证中间件、JWT 处理"
- notion: "存在哪些认证文档？查找 API 认证指南"
(slack 被省略，因为此技术问题不相关)"""
        },
        {"role": "user", "content": state["question"]}
    ])

    return  {"classifications":result.classifications}

def sent_to_agents(state:RouterState)-> list[Send]:
    """根据分类向每个代理分发任务"""
    return [Send(c.get("source"),{"question":c.get('question')}) for c in state.get("classifications")]


def github_query(state:InputState)->dict:
    result = github_agent.invoke(
        {"role":"user","content":state.get("question")}
    )

    return {"results":[{"source": "github", "result": result["messages"][-1].content}]}

def notion_query(state:InputState)->dict:
    result = notion_agent.invoke(
        {"role":"user","content":state.get("question")}
    )
    return  {"results":[{"source":"notion", "result": result["messages"][-1].content}]}

def slack_query(state:InputState)->dict:
    result = slack_agent.invoke(
        {"role":"user","content":state.get("question")}
    )
    return {"results":[{"source":"slack", "result": result["messages"][-1].content}]}

def synthesize_results(state:RouterState)->dict:
    "合并多个查询结果并生成最终答案"

    if not state["results"]:
        return {"final_answer":"没有从任何已知数据集中找到相关信息"}

    prompt = [f"在{r['source'].title()}的结果是{r['result']}" for r in state['results']]

    synthesis_response  = router_llm.invoke(
        [{"role":"system","content":f"""综合这些搜索结果以回答原始问题: {state['question']}
从多个来源结合信息，避免冗余
突出显示最相关和可操作的信息
记录来源之间的任何差异
保持回答简洁且条理清晰"""},
        {"role":"user","content":"\n\n".join(prompt)}]
    )

    return {"final_answer":synthesis_response.content}

# ===========================5.组装====================
workflow = (
    StateGraph(RouterState) # StateGraph(RouterState) 用来构建一个基于状态的流程图。RouterState 定义了流程中共享的数据结构。
    .add_node("classify",classify_question)
    .add_node("github",github_query)
    .add_node("notion",notion_query)
    .add_node("slack",slack_query)
    .add_node("synthesize",synthesize_results)
    .add_edge(START,"classify")
    .add_conditional_edges("classify",sent_to_agents,["github","notion","slack"])
    .add_edge("github","synthesize")
    .add_edge("notion","synthesize")
    .add_edge("slack","synthesize")
    .add_edge("synthesize",END)
    .compile()
)


# =====================6. 使用==============
result = workflow.invoke({
    "question": "如何进行API请求的认证？"
})
print("Original query:", result["question"])
print("\nClassifications:")
for c in result["classifications"]:
    print(f"  {c['source']}: {c['question']}")
print("\n" + "=" * 60 + "\n")
print("Final Answer:")
print(result["final_answer"])

