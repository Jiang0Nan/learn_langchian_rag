import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
async def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
    )
    path = "../file/"
    loader = DirectoryLoader(path, glob="**/*.pdf")
    docs = loader.load()


    # model_ollama = ChatOllama(
    #     model = model_name,
    #     base_url=base_url,
    #     reasoning=True,#是否启用思考模式
    #     temperature = 0.8,
    #     validate_model_on_init = False,#是否验证ollama 中该模型是否存在
    #     repeat_last_n = 64,#设置模型要回溯多远以防止重复。(默认值:` ` 64 `, ` ` 0 `=禁用，``- 1 `= ` num _ CTX `) "越长的文本越大
    #     repeat_penalty =1.1,#设置惩罚重复的力度。更高的值(例如`` 1.5 `)将更强烈地惩罚重复，而较低的值(例如`` 0.9 ``)会比较宽大。(默认值:` ` 1.1 ')
    #     # seed =1 #设置用于生成的随机数种子。设置这个将使模型为生成相同的文本同样的提示
    # )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper


    generator_llm = LangchainLLMWrapper(model)
    generator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=r"D:\files\models\bge-m3",query_encode_kwargs = {"batch_size":16},encode_kwargs ={"batch_size":16},show_progress = True ))


    # 生成测试集
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    # 切换语言
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("zh", llm=generator_llm)
        query.set_prompts(**prompts)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=5,query_distribution=distribution)

    # 分析测试集

    df = dataset.to_pandas()
    df.to_csv("ragas_testset_2.csv", index=False, encoding="utf-8")  # 保存到本地
    print(f"测试集已保存，共 {len(df)} 条数据，路径：ragas_testset_2.csv")

if __name__ == '__main__':
    asyncio.run(main())