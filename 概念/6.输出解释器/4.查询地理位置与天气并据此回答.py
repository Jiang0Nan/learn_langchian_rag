import json
import os

import requests
from IPython.core.debugger import prompt
from lear_langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from sqlalchemy.testing.suite.test_reflection import metadata

if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
model = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek")


@tool
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的地理位置，\
    注意，城市需要用对应城市的地理位置代替，例如如果需要查询北京市天气，则loc参数需要输入'116.41,39.92'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://nb6x86uvt5.re.qweatherapi.com/v7/weather/now"

    # Step 2.设置查询参数
    params = {
        "location": loc,
        "unit": "m",  # 使用摄氏度而不是华氏度
        "lang": "zh"  # 输出语言为简体中文
    }
    headers = {"X-QW-Api-Key": "4a253a9b9ffd4509b2d7827a720147e8",
               "Content-Type": "application/json"}
    # Step 3.发送GET请求
    response = requests.get(url, params=params, headers=headers)

    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)


@tool
def get_loc_by_name(name):
    """
    查询即时天气函数
    :param name: 必要参数，字符串类型，用于表示查询的城市，\
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://restapi.amap.com/v3/geocode/geo"

    # Step 2.设置查询参数
    params = {
        "key":"3c54a4a93ab69976360f63f43b6075af",
        "address": name,
    }
    # Step 3.发送GET请求
    response = requests.get(url, params=params)

    # Step 4.解析响应
    data = response.json()
    return data["geocodes"][0]['location'] if data["geocodes"] else None

get_loc_by_name("重庆")
# 调用方法1 没有@tools情况下
a=get_weather("116.41,39.92")
print(a)
print(get_weather.name)
print(get_weather.description)
print(get_weather.args)
# 调用方法2
print(get_weather.invoke({ "loc":"116.41,39.92"}))
# {"code": "200", "updateTime": "2025-09-18T09:51+08:00", "fxLink": "https://www.qweather.com/weather/dongcheng-101011600.html", "now": {"obsTime": "2025-09-18T09:48+08:00", "temp": "21", "feelsLike": "19", "icon": "100", "text": "\u6674", "wind360": "0", "windDir": "\u5317\u98ce", "windScale": "3", "windSpeed": "13", "humidity": "41", "precip": "0.0", "pressure": "1020", "vis": "30", "cloud": "0", "dew": "7"}, "refer": {"sources": ["QWeather"], "license": ["QWeather Developers License"]}}

#模型绑定工具
model_with_tools = model.bind_tools(tools=[get_weather, get_loc_by_name])
# Out[4]: RunnableBinding(bound=ChatDeepSeek(client=<openai.resources.chat.completions.completions.Completions object at 0x0000025CF84F6590>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x0000025CF90A6A70>, root_client=<openai.OpenAI object at 0x0000025CF90278E0>, root_async_client=<openai.AsyncOpenAI object at 0x0000025CF90A6A10>, model_name='deepseek-reasoner', model_kwargs={}, api_key=SecretStr('**********'), api_base='https://api.deepseek.com/v1'), kwargs={'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'description': "查询即时天气函数\n:param loc: 必要参数，字符串类型，用于表示查询天气的地理位置，    注意，城市需要用对应城市的地理位置代替，例如如果需要查询北京市天气，则loc参数需要输入'116.41,39.92'；\n:return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息", 'parameters': {'properties': {'loc': {}}, 'required': ['loc'], 'type': 'object'}}}]}, config={}, config_factories=[])
# kwargs是关于工具的相关信息
# model_with_tools.invoke("北京的天气怎么样")
#
# AIMessage(content='我需要查询北京的天气信息。不过为了获取准确的天气数据，我需要知道北京的具体地理位置坐标。您能提供一下北京的经纬度坐标吗？比如类似"116.41,39.92"这样的格式。\n\n或者如果您知道北京某个具体区域（如朝阳区、海淀区等）的坐标，我也可以为您查询该区域的天气情况。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 72, 'prompt_tokens': 249, 'total_tokens': 321, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 57}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '1c407e32-906e-44c7-9594-bd581e6154e7', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--962297a3-eea9-4fdb-8956-9de0259be3cb-0', usage_metadata={'input_tokens': 249, 'output_tokens': 72, 'total_tokens': 321, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}})

# print(model_with_tools.kwargs["tools"])

# =============================真正的使用
prompt = ChatPromptTemplate.from_template("请从以下用户问题中提取一个能代表地理位置的城市或地区名称：{q}。只返回地名本身，不要其他内容。")
extract_location_chain = prompt |  model_with_tools

tool_chain = extract_location_chain | RunnableLambda(lambda  x : {"name":x.content.strip()}) | get_loc_by_name | RunnableLambda(lambda x: {"loc":x}) | get_weather
query = "重庆的天气怎么样"
weather_data = tool_chain.invoke({ "q": query })

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "天气信息来源于OpenWeather API：https://api.openweathermap.org/data/2.5/weather"),
        ("system", "这是实时的天气数据：{weather_data}"),
        ("human", "{user_input}"),
    ]
)

final_chain =   chat_template | model

final_res = final_chain.invoke({"weather_data":weather_data,"user_input":query})
print(final_res)
