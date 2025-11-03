import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
# 目的实用的天气预报代理
# 整体步骤为
# 2. 设计详细的系统提示词来改善agent的行为（Detailed system prompts）
# 3.创建与外部数据集成的工具（Create tools）
# 4.一致响应的模型配置（Model configuration ）
# 5.结构化输出 来可预测结果（Structured output）
# 6.类似聊天互动的会话记忆（Conversational memory）
# 7.创建并运行代理创建一个功能齐全的代理（Create and run the agent ）



#=========================1.定义系统提示符===========================
SYSTEM_PROMPT = """你是一位擅长说双关语的天气预报专家。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户所在的位置

如果用户向你询问天气，请确保你已经知道具体位置。
如果从用户的问题中可以判断他们指的是“他们当前所在的位置”，
请使用 get_user_location 工具来获取他们的位置。"""

# ===========================2.创建工具===============================
# 工具允许模型通过调用您定义的函数与外部系统交互。工具可以依赖于运行时上下文 ，也可以与代理内存交互。

from dataclasses import dataclass
from langchain.tools import tool,ToolRuntime
"""将 Python 函数转换为工具（Tool），可带参数或不带参数使用。

    参数说明:
        name_or_callable: 工具的名称或要转换的可调用对象（callable）。
            必须以位置参数的形式提供。
        runnable: 可选的可运行对象（runnable），用于转换为工具。
            必须以位置参数的形式提供。
        description: 工具的可选描述信息。
            工具描述的优先级如下：

            - `description` 参数
                （即使提供了 docstring 或 `args_schema`，也优先使用此项）
            - 工具函数的 docstring
                （即使提供了 `args_schema`，也会优先使用 docstring）
            - `args_schema` 的描述
                （仅在 `description` 和 docstring 都未提供时使用）

        *args: 额外的位置参数，必须为空。
        return_direct: 是否在调用工具后直接返回结果，
            而不是继续执行代理循环。
        args_schema: 可选的参数 schema，供用户手动指定。

        infer_schema: 是否从函数签名中自动推断参数的 schema。
            启用后，生成的工具的 `run()` 函数将支持字典类型输入。
        response_format: 工具的响应格式。
            若为 `"content"`，则工具输出会被解释为 `ToolMessage` 的内容；
            若为 `"content_and_artifact"`，则工具输出应为一个二元组，
            对应 `ToolMessage` 的 `(content, artifact)`。
        parse_docstring: 若同时启用 `infer_schema` 和 `parse_docstring`，
            将尝试从 Google 风格的函数 docstring 中解析参数描述。
        error_on_invalid_docstring: 若启用 `parse_docstring`，
            用于配置在遇到无效的 Google 风格 docstring 时是否抛出 `ValueError`。

    异常:
        ValueError: 提供了过多的位置参数。
        ValueError: 提供了 runnable 但没有提供字符串形式的名称。
        ValueError: 第一个参数既不是字符串，也不是带有 `__name__` 属性的可调用对象。
        ValueError: 若函数没有 docstring 且未提供 description，
            并且 `infer_schema=False`，则抛出此错误。
        ValueError: 若 `parse_docstring=True` 且函数的 docstring 不是有效的
            Google 风格文档，并且 `error_on_invalid_docstring=True`，则抛出此错误。
        ValueError: 若提供的 Runnable 没有对象 schema，则抛出此错误。

    返回:
        转换后的工具对象。

    要求:
        - 函数类型必须为 `(str) -> str`
        - 函数必须包含 docstring
"""

@tool
def get_weather_for_location(city:str)->str:
    """根据city获取天气"""
    return f"{city} 一直是晴天"

@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

"""运行时上下文会自动注入到工具中。

    当一个工具函数包含名为 `tool_runtime` 且类型标注为 `ToolRuntime` 的参数时，
    工具执行系统会自动注入一个实例，其中包含：

    - `state`: 当前图（graph）的状态
    - `tool_call_id`: 当前工具调用的唯一 ID
    - `config`: 当前执行的 `RunnableConfig`
    - `context`: 运行时上下文（来自 langgraph 的 `Runtime`）
    - `store`: 持久化存储的 `BaseStore` 实例（来自 langgraph 的 `Runtime`）
    - `stream_writer`: 用于流式输出的 `StreamWriter`（来自 langgraph 的 `Runtime`）

    不需要使用 `Annotated` 包装 —— 只需在参数中使用 `runtime: ToolRuntime` 即可。

    示例:
        ```python
        from langchain_core.tools import tool
        from langchain.tools import ToolRuntime

        @tool
        def my_tool(x: int, runtime: ToolRuntime) -> str:
            \"\"\"访问运行时上下文的工具。\"\"\"
            # 访问状态
            messages = tool_runtime.state["messages"]

            # 访问 tool_call_id
            print(f"工具调用 ID: {tool_runtime.tool_call_id}")

            # 访问配置
            print(f"运行 ID: {tool_runtime.config.get('run_id')}")

            # 访问运行时上下文
            user_id = tool_runtime.context.get("user_id")

            # 访问存储
            tool_runtime.store.put(("metrics",), "count", 1)

            # 流式输出
            tool_runtime.stream_writer.write("Processing...")

            return f"已处理 {x}"
        ```

    !!! 注意
        这是一个用于类型检查和检测的标记类。
        实际的运行时对象会在工具执行过程中构建。
"""

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据user id检索用户信息"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# ===============================3.配置模型========================
from langchain.chat_models import init_chat_model

"""使用模型名称和提供商，在一行中初始化一个聊天模型。

!!! note
    需要安装对应模型提供商的集成包。

    请参阅下方的 `model_provider` 参数，了解具体的包名
    （例如：`pip install langchain-openai`）。

    参考 [模型提供商的 API 文档](https://docs.langchain.com/oss/python/integrations/providers)
    以了解支持的模型参数。

参数说明:
    model: 模型名称，例如 `'o3-mini'`、`'claude-sonnet-4-5'`。

        你也可以在一个参数中同时指定模型和提供商：

        使用 `'{model_provider}:{model}'` 格式，例如 `'openai:o1'`。
    model_provider: 如果未在 `model` 参数中指定模型提供商，则在此处设置。
        支持的 `model_provider` 及其对应的集成包如下：

        - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
        - `anthropic`               -> [`langchain-anthropic`](https://docs.langchain.com/oss/python/integrations/providers/anthropic)
        - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
        - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
        - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
        - `google_genai`            -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
        - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
        - `bedrock_converse`        -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
        - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
        - `fireworks`               -> [`langchain-fireworks`](https://docs.langchain.com/oss/python/integrations/providers/fireworks)
        - `together`                -> [`langchain-together`](https://docs.langchain.com/oss/python/integrations/providers/together)
        - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
        - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
        - `groq`                    -> [`langchain-groq`](https://docs.langchain.com/oss/python/integrations/providers/groq)
        - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)
        - `google_anthropic_vertex` -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
        - `deepseek`                -> [`langchain-deepseek`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
        - `ibm`                     -> [`langchain-ibm`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
        - `nvidia`                  -> [`langchain-nvidia-ai-endpoints`](https://docs.langchain.com/oss/python/integrations/providers/nvidia)
        - `xai`                     -> [`langchain-xai`](https://docs.langchain.com/oss/python/integrations/providers/xai)
        - `perplexity`              -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)

        如果未指定 `model_provider`，系统会尝试从模型名称中自动推断：

        - `gpt-...` | `o1...` | `o3...`       -> `openai`
        - `claude...`                         -> `anthropic`
        - `amazon...`                         -> `bedrock`
        - `gemini...`                         -> `google_vertexai`
        - `command...`                        -> `cohere`
        - `accounts/fireworks...`             -> `fireworks`
        - `mistral...`                        -> `mistralai`
        - `deepseek...`                       -> `deepseek`
        - `grok...`                           -> `xai`
        - `sonar...`                          -> `perplexity`

    configurable_fields: 哪些模型参数是可配置的：

        - `None`: 不可配置。
        - `'any'`: 所有字段可配置。**详见安全提示。**
        - `list[str] | Tuple[str, ...]`: 指定可配置字段名称。

        如果模型已指定，默认为 `None`；
        如果模型未指定，默认为 `("model", "model_provider")`。

        !!! warning "安全提示"
            设置 `configurable_fields="any"` 意味着 `api_key`、`base_url` 等字段
            也能在运行时被修改，可能导致请求被重定向到他人服务。
            如果接受外部配置输入，建议明确列出允许的字段。

    config_prefix: 如果 `'config_prefix'` 是非空字符串，
        模型可通过 `config["configurable"]["{config_prefix}_{param}"]` 键进行配置；
        如果 `'config_prefix'` 是空字符串，则通过
        `config["configurable"]["{param}"]` 进行配置。

    **kwargs: 传递给底层聊天模型构造函数的额外关键字参数。
        常见包括：

        - `temperature`: 控制随机性的温度参数。
        - `max_tokens`: 最大输出 token 数。
        - `timeout`: 响应超时时间（秒）。
        - `max_retries`: 最大重试次数。
        - `base_url`: 自定义 API 端点。
        - `rate_limiter`: 控制请求速率的 `BaseRateLimiter` 实例。

        请参考对应模型提供商文档了解更多参数。

返回值:
    - 若配置不可变，返回一个对应 `model_name` 与 `model_provider` 的 `BaseChatModel`；
    - 若配置可变，则返回一个延迟初始化的“可配置聊天模型”。

异常:
    - `ValueError`: 无法推断或不支持的 `model_provider`。
    - `ImportError`: 未安装所需的模型集成包。

???+ note "初始化一个不可配置的模型"

    ```python
    # pip install langchain langchain-openai langchain-anthropic langchain-google-vertexai
    from langchain.chat_models import init_chat_model

    o3_mini = init_chat_model("openai:o3-mini", temperature=0)
    claude_sonnet = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
    gemini_2_flash = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)

    o3_mini.invoke("what's your name")
    claude_sonnet.invoke("what's your name")
    gemini_2_flash.invoke("what's your name")
    ```

??? note "部分可配置模型（无默认模型）"

    ```python
    # pip install langchain langchain-openai langchain-anthropic
    from langchain.chat_models import init_chat_model

    # 若未指定模型，自动启用可配置模式。
    configurable_model = init_chat_model(temperature=0)

    configurable_model.invoke("what's your name", config={"configurable": {"model": "gpt-4o"}})
    # GPT-4o 响应

    configurable_model.invoke(
        "what's your name",
        config={"configurable": {"model": "claude-sonnet-4-5"}},
    )
    ```

??? note "完全可配置模型（有默认模型）"

    ```python
    # pip install langchain langchain-openai langchain-anthropic
    from langchain.chat_models import init_chat_model

    configurable_model_with_default = init_chat_model(
        "openai:gpt-4o",
        configurable_fields="any",  # 允许运行时修改参数，如 temperature、max_tokens 等。
        config_prefix="foo",
        temperature=0,
    )

    configurable_model_with_default.invoke("what's your name")
    # GPT-4o 响应，temperature=0

    configurable_model_with_default.invoke(
        "what's your name",
        config={
            "configurable": {
                "foo_model": "anthropic:claude-sonnet-4-5",
                "foo_temperature": 0.6,
            }
        },
    )
    ```

??? note "为可配置模型绑定工具"

    你可以像普通模型一样，对可配置模型调用声明式方法：

    ```python
    # pip install langchain langchain-openai langchain-anthropic
    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field


    class GetWeather(BaseModel):
        '''获取指定地点的当前天气'''

        location: str = Field(..., description="城市与州，例如：San Francisco, CA")


    class GetPopulation(BaseModel):
        '''获取指定地点的当前人口'''

        location: str = Field(..., description="城市与州，例如：San Francisco, CA")


    configurable_model = init_chat_model(
        "gpt-4o", configurable_fields=("model", "model_provider"), temperature=0
    )

    configurable_model_with_tools = configurable_model.bind_tools(
        [
            GetWeather,
            GetPopulation,
        ]
    )
    configurable_model_with_tools.invoke(
        "今天哪个城市更热，哪个城市人口更多：洛杉矶还是纽约？"
    )

    configurable_model_with_tools.invoke(
        "今天哪个城市更热，哪个城市人口更多：洛杉矶还是纽约？",
        config={"configurable": {"model": "claude-sonnet-4-5"}},
    )
    ```
"""

model = init_chat_model(
    f"deepseek:deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
# =============================4.定义响应格式=======================
from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """agent返回的模式"""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# =============================5.添加memory============================
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# =======================6. 创建并运行代理============
# model只支持chatmodel不支持init_chat_model的结果
"""创建一个代理图（agent graph），该图会循环调用工具，直到满足停止条件为止。

    有关使用 `create_agent` 的更多详细信息，
    请参阅 [Agents](https://docs.langchain.com/oss/python/langchain/agents) 文档。

    参数说明:
        model: 代理所使用的语言模型。可以是字符串标识符
            （例如 `"openai:gpt-4"`），也可以是聊天模型实例（例如 `ChatOpenAI()`）。
            支持的模型字符串完整列表可参考：
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)]。
        tools: 工具列表，可为 `list[Tool]`、`dict` 或 `Callable`。
            如果为 `None` 或空列表，则代理仅包含一个模型节点，不会进入工具调用循环。
        system_prompt: 可选的系统提示词（system prompt），
            会被转换为 `SystemMessage` 并添加到消息列表的开头。
        middleware: 一组中间件实例，用于在代理执行的不同阶段拦截并修改行为。
        response_format: 可选的结构化响应配置。
            可以是 `ToolStrategy`、`ProviderStrategy` 或 Pydantic 模型类。
            如果提供此参数，代理会在对话流程中处理结构化输出。
            原始 schema 会根据模型能力被封装为合适的策略。
        state_schema: 可选的 `TypedDict` schema，用于扩展 `AgentState`。
            提供此参数时，将使用该 schema 代替默认的 `AgentState`，
            作为与中间件状态 schema 合并的基础。
            这允许用户在不创建自定义中间件的情况下添加自定义状态字段。
            通常建议通过中间件扩展 state_schema，以保持作用域对应。
            schema 必须是 `AgentState[ResponseT]` 的子类。
        context_schema: 可选的运行时上下文 schema。
        checkpointer: 可选的检查点存储对象，
            用于在单个线程（例如一次对话）中持久化图的状态（如聊天记忆）。
        store: 可选的数据存储对象，
            用于在多个线程（例如多个对话/用户）之间持久化数据。
        interrupt_before: 可选的节点名称列表，
            在这些节点执行之前触发中断。
            适用于需要在执行前添加用户确认或其它交互的场景。
        interrupt_after: 可选的节点名称列表，
            在这些节点执行之后触发中断。
            可用于直接返回或在输出上进行额外处理。
        debug: 是否启用图执行的详细日志。
            启用后，会打印每个节点的执行详情、状态更新和状态转移，
            有助于调试中间件行为及理解代理执行流程。
        name: 可选的 `CompiledStateGraph` 名称。
            在将该代理图作为子图添加到其他图中时，会自动使用此名称，
            这在构建多代理系统时尤为有用。
        cache: 可选的 `BaseCache` 实例，用于启用图执行缓存。

    返回:
        一个已编译的 `StateGraph`，可用于聊天交互。

    执行逻辑:
        代理节点首先使用消息列表（应用系统提示后）调用语言模型。
        如果返回的 `AIMessage` 包含 `tool_calls`，
        则图会调用相应的工具节点。
        工具节点执行工具后，将响应以 `ToolMessage` 的形式添加到消息列表中。
        之后代理节点再次调用语言模型。
        该过程会重复，直到模型响应中不再包含 `tool_calls`。
        最后，代理返回完整的消息列表。
"""

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])