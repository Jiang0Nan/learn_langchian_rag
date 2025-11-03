import asyncio
import os
import platform

from langchain_community.document_loaders import DirectoryLoader
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

base_url = "http://localhost:11434"

path = "../file/"
loader = DirectoryLoader(path, glob="**/*.pdf")
docs = loader.load()

# 使用文档创建基础知识图谱
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

# 设置 LLM 和嵌入模型
from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
    os.environ["OPENAI_API_KEY"] = "***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"
# 配置 DeepSeek 模型
llm = llm_factory(
    model="deepseek-chat",  # 模型名称（如 deepseek-chat、deepseek-reasoner）
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 官方地址
    default_headers={
        "Authorization": "Bearer ***REMOVED***a6e1ed1bc0354c6f848a672c36fa3240"  # 替换为你的API密钥
    },
)
embedding = embedding_factory(
    provider="huggingface",  # 明确指定提供商为 HuggingFace
    model=r"D:\files\models\bge-m3",  # 本地模型路径
    # 额外参数：设置批量编码大小（加速处理）
    )
# 设置转换（transforms）
# 这里我们使用了 2 个提取器和 2 个关系构建器。 - 标题提取器：从文档中提取标题 - 关键词提取器：从文档中提取关键词 - 标题分割器：根据标题将文档分割成节点

from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)


headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="key_phrases", max_num=10
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
]

apply_transforms(kg, transforms=transforms)

# 配置角色/画像
from ragas.testset.persona import Persona

persona1 = Persona(
    name="a doctor",
    role_description="You are a clinical doctor responsible for implementing clinical new technologies. You focus on specific operational details in the article, such as technical operation standards, quality control indicators, patient eligibility criteria, and steps for handling adverse reactions. You need to ask questions that help clarify how to perform your daily clinical work."
)
persona2 = Persona(
    name="a hospital manager",
    role_description="You are a hospital administrator responsible for formulating and supervising the implementation of clinical technology management systems. You focus on institutional frameworks, responsibility allocation (e.g., composition of quality control teams), assessment mechanisms, compliance requirements, and resource allocation in the article. You need to ask questions that help improve management efficiency and regulatory compliance."
)
persona3 = Persona(
    name="a intern",
    role_description="You are a medical intern learning about clinical new technology management. You focus on basic concepts (e.g., definition of graded access management), core procedures (e.g., application process for new technologies), common terms (e.g., 'restricted technologies'), and step-by-step operational guidelines in the article. You need to ask questions that help you understand foundational knowledge and entry-level operations."
)
persona_list = [persona1, persona2,persona3]

# 单跳查询
# 继承 SingleHopQuerySynthesizer 并修改生成查询创建场景的函数。
#
# 步骤：- 找到用于查询创建的合格节点集。这里我选择所有已提取关键词的节点。- 对于每个合格集 - 将关键词与一个或多个角色/画像匹配。- 创建 (节点, 角色/画像, 查询风格, 查询长度) 的所有可能组合。- 从组合中抽取所需数量的查询
from ragas.testset.synthesizers.single_hop import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from dataclasses import dataclass
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)


@dataclass
class MySingleHopScenario(SingleHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):

        property_name = "key_phrases"
        nodes = []
        for node in knowledge_graph.nodes:
            if node.type.name == "CHUNK" and node.get_property(property_name):
                nodes.append(node)

        number_of_samples_per_node = max(1, n // len(nodes))

        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.properties.get(property_name, [""])
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )
            scenarios.extend(
                self.sample_combinations(base_scenarios, number_of_samples_per_node)
            )

        return scenarios
async def main():
    query = MySingleHopScenario(llm=llm)
    await query.adapt_prompts(language="chinese",llm=llm)
    scenarios = await query.generate_scenarios(
        n=60, knowledge_graph=kg, persona_list=persona_list
    )

    # print(scenarios[0])

    # result =   [await query.generate_sample(scenario=scenarios[-1]) ]

    # 修改提示词以定制查询风格
    # 这里我用一个只生成是/否问题的指令替换了默认提示词。这是一个可选步骤

    # instruction = """Generate a Yes/No query and answer based on the specified conditions (persona, term, style, length)
    # and the provided context. Ensure the answer is entirely faithful to the context, using only the information
    # directly from the provided context.
    #
    # ### Instructions:
    # 1. **Generate a Yes/No Query**: Based on the context, persona, term, style, and length, create a question
    # that aligns with the persona's perspective, incorporates the term, and can be answered with 'Yes' or 'No'.
    # 2. **Generate an Answer**: Using only the content from the provided context, provide a 'Yes' or 'No' answer
    # to the query. Do not add any information not included in or inferable from the context."""

    # 中文简单问题提示词（事实性问题）
    simple_prompt = """基于以下中文文档和角色，生成1个简单的事实性问题及答案。
    要求：
    - 问题需用中文，围绕文档核心事实（如定义、步骤、标准）；
    - 答案必须从文档中直接提取，不添加额外信息。
    """


    # 中文推理问题提示词（需要逻辑推断）
    reasoning_prompt ="""基于以下中文文档和角色，生成1个需要推理的问题及答案。
    要求：
    - 问题需用中文，需结合文档信息进行简单逻辑推断（如因果关系、流程先后）；
    - 答案需基于文档内容推导，明确说明推理依据。
    """


    # 中文多上下文问题提示词（跨片段关联）
    multi_context_prompt = """基于以下多个中文文档片段和角色，生成1个需要关联多段内容的问题及答案。
    要求：
    - 问题需用中文，需综合至少2个文档片段的信息才能回答；
    - 答案需明确引用关联的文档片段，说明信息来源。
    """
    length_s=len(scenarios)
    prompts = [simple_prompt, reasoning_prompt, multi_context_prompt]
    test_cases = []
    for i in range(length_s):
        # 轮询使用不同提示词
        prompt_fi = query.get_prompts()["query_answer_generation_prompt"]
        prompt_fi.instruction =  prompts[i%3]
        query.set_prompts(**{"query_answer_generation_prompt": prompt_fi})
        # 生成单个测试用例
        case = await query.generate_sample(scenario=scenarios[i])  # 需结合场景生成逻辑
        print(f"{i}/{length_s}")
        test_cases.append(case)
    # prompt = query.get_prompts()["query_answer_generation_prompt"]
    # print(prompt)
    # prompt.instruction = prompt.instruction+"3. **use chinese to response"
    # print(prompt)
    # query.set_prompts(**{"query_answer_generation_prompt": prompt})
    # result = [await query.generate_sample(scenario=s) for s in scenarios[0:1]]
    print(test_cases)
    import pandas as pd
    data = []
    for sample in test_cases:
        data.append({
            "问题（user_input）": sample.user_input,
            "参考上下文（reference_contexts）": "\n".join(sample.reference_contexts),  # 列表转字符串
            "标准答案（reference）": sample.reference
            # 可选：若有系统回答，可添加 "系统回答（response）": sample.response
        })

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 保存为 Excel（需安装 openpyxl 库）
    df.to_excel("ragas_testset.xlsx", index=False, engine="openpyxl")
    print("已保存到 ragas_testset.xlsx")


if __name__ == '__main__':
    asyncio.run(main())