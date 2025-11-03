from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

base_url = "http://localhost:11434"
model_name = 'deepseek-r1:7b-qwen-distill-q4_K_M'
# 1.========================两种初始化，
# 请注意
# mirostat，mirostat_eta，mirostat_tau优先级比top_k,top_p更高
model_ollama = ChatOllama(
    model = model_name,
    base_url=base_url,
    reasoning=True,#是否启用思考模式
    temperature = 0.8,
    validate_model_on_init = False,#是否验证ollama 中该模型是否存在
    mirostat = 0,#启用Mirostat采样以控制困惑，1：启用 Mirostat v1（经典版）2：启用 Mirostat v2（更稳定，0：禁用
    mirostat_eta=0.1,#影响算法响应反馈的速度从生成的文本中。较低的学习率将导致较慢的调整，而较高的学习率将使算法反应更快。(默认值为`` 0.1 `)
    mirostat_tau = 5.0,#“控制一致性和多样性之间的平衡输出的。较低的值将导致更集中和连贯的文本。(默认值:` ` 5.0 `) 类似temperature 小值（如 2~4） → 输出更集中、更可控大值（如 5~8） → 输出更发散、更有创意
    num_ctx = 2048,#设置用于生成的上下文窗口的大小下一个令牌。(默认值:` ` 2048 ')
    num_thread = 3,#设置计算期间要使用的线程数。默认情况下，Ollama会检测这一点以获得最佳性能。建议将该值设置为物理数量系统拥有的CPU核心数(相对于逻辑核心数)。
    num_predict = 128,#生成文本时要预测的最大标记数。(默认:` ` 128 `, ` `- 1 ` =无限代，` `- 2 ` =填充上下文)""
    repeat_last_n = 64,#设置模型要回溯多远以防止重复。(默认值:` ` 64 `, ` ` 0 `=禁用，``- 1 `= ` num _ CTX `) "越长的文本越大
    repeat_penalty =1.1,#设置惩罚重复的力度。更高的值(例如`` 1.5 `)将更强烈地惩罚重复，而较低的值(例如`` 0.9 ``)会比较宽大。(默认值:` ` 1.1 ')
    # seed =1 #设置用于生成的随机数种子。设置这个将使模型为生成相同的文本同样的提示
    stop = ["\\n"],
    tfs_z = 1,#无尾采样用于减少不太可能的影响输出中的令牌。较高的值(例如`` 2.0 ``)会降低影响更大，而值“1.0”禁用此设置。(默认值:` ` 1 `)
    top_p = 0.9,#从概率大于0.9中采样
    top_k = 40,#与top-k一起使用。较高的值(例如`` 0.95 ``)将产生更多样化的文本，而较低的值(例如`` 0.5 ``)将产生更集中和保守的文本。(默认值为`` 0.9 `)
    format = None,#指定输出的格式(选项:` ` json ' `，JSON模式)。
# keep_alive = #模型停留在内存的时间
)

model_init = init_chat_model(
    model = model_name,
    model_provider = "ollama",
    temperature = 0.6,
    base_url = base_url,
    config_prefix = "model_test",#配置前缀，用于在配置管理中区分不同模型的参数。
    configurable_fields = ["temperature", "base_url"],#配置可变参数

    max_retries = 3, #最大重试次数
    rate_limiter =1, #请求间隔时间
    max_tokens = -1, #生成的最大 token 数，-1 表示无限制。
    top_p = 0.9,#从概率大于0.9中采样
    top_k = 1,#从概率最高的集合中提取1个
    num_predict = -1,
    num_ctx = 4096,#上下文
    stop = ["\\n"],
    streaming = True,
)


if __name__ == '__main__':
    pass