from pprint import pprint

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema, Function, FunctionType, AnnSearchRequest, \
    WeightedRanker
from pymilvus.milvus_client import IndexParams
from FlagEmbedding import BGEM3FlagModel
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True, "batch_size": 32}
embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\files\models\bge-m3",
    cache_folder=r"D:\files\models\bge-m3",
    model_kwargs=model_kwargs,  # 模型的参数
    encode_kwargs=encode_kwargs,  # encode的参数
    show_progress=True,
    # multi_process = True,#是否启用多进程并行编码，默认 False（注意：可能和某些环境冲突）
)
embeddings_bge = BGEM3FlagModel(
    model_name_or_path= r"D:\files\models\bge-m3",
    cache_folder=r"D:\files\models\bge-m3",
    use_fp16=True
)
# embeddings.embed_documents()

URI = "http://localhost:19530"
# vector_store = MilvusClient(
#     # embedding_function = embeddings,
#     connection_args={"uri": URI}
# )
# 参数
schema_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector_dense", dtype=DataType.FLOAT_VECTOR, description="嵌入后的密集向量", dim=1024),
    FieldSchema(name="vector_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR, description="嵌入后的稀疏向量"),
    # 稀疏向量不指定dim
    FieldSchema(name="text", dtype=DataType.VARCHAR, description="原文本", max_length=4096, enable_analyzer=True),
    FieldSchema(name="metadata", dtype=DataType.JSON, description="metadata"),
    # FieldSchema(name="drug_name" , dtype=DataType.VARCHAR,description="药品名称" ,max_length  = 50)
]

# 添加一个函数用于指定input_field_names传入参数，并使用function_type指定方法体，output_field_names进行存储
bm25_function = Function(
    name="text_to_bm25",  # 方法名
    input_field_names=["text"],  # 指定输入字段
    output_field_names=["vector_sparse"],  # 指定输出字段
    function_type=FunctionType.BM25,  # 指定量化方法
    # description = ""
    # params = Dict
)

# todo params的参数有问题
# embedding_function = Function(从
#     name = "text_to_embedding_matrix",
#     input_field_names=["text"],
#     output_field_names=["vector_dense"],
#     function_type=FunctionType.TEXTEMBEDDING,
#     params={"model_name": "bge-large-zh-v1.5"}
# )
#
# # 添加函数用于自动转换
# schema = CollectionSchema(fields=schema_fields,functions=[bm25_function,embedding_function])#法1

schema = CollectionSchema(fields=schema_fields)  # 法2
schema.add_function(bm25_function)

# =======

client = MilvusClient(URI)

if client.has_collection(collection_name="milvus_test"):
    client.drop_collection(collection_name="milvus_test")
#     注意：部分参数会被schema的覆盖，即，schema的参数优先级更高
client.create_collection(
    collection_name="milvus_test",
    # dimension=1024,  # 定义向量字段的维度（即每个向量包含的元素数量）
    # primary_field_name= "id",  # 指定主键字段的名称（用于唯一标识每条数据，类似数据库的主键）
    # id_type="int",  # 定义主键字段的数据类型
    # vector_field_name= "vector",  # 指定存储向量数据的字段名称。主要是看 schema，若有schema则该会被忽略
    # metric_type="COSINE",#定义向量相似度的计算方式  #COSINE"：余弦相似度（适合文本、图像等向量，衡量方向相似性）；"L2"：欧氏距离（适合衡量空间中两点的直线距离）；"IP"：内积（适合推荐系统等场景）。
    # auto_id = True,#控制主键是否自动生成
    # timeout = None,#设置操作超时时间
    schema=schema,  # 如果不是使用CollectionSchema创建的还需要包裹一层CollectionSchema（schema）
    # index_params = None,#建集合时是否自动为向量字段创建索引
    consistency_level="Strong",
    drop_old=True,
)

# consistency_level :
"""一致性级别	核心特点	适用场景
Strong	写入的数据立即对所有后续读写操作可见（全局一致）。	对数据一致性要求极高的场景（如金融交易记录、医疗数据等），不允许读写到旧数据。
Session	同一客户端会话内，写入的数据对后续操作可见；不同会话间可能存在延迟。	大多数普通业务场景（如知识库检索、商品推荐），兼顾一致性和性能，是默认级别。
Bounded	允许一定时间窗口内的旧数据可见（窗口可配置），超出窗口后保证数据一致。	需要平衡一致性和性能的场景（如实时日志分析），可接受短时间内的轻微数据延迟。
Eventually	写入的数据最终会被所有节点可见，但无时间保证（延迟可能较长）。	对一致性要求低，但追求极致吞吐量的场景（如海量非核心日志存储、离线数据备份）。"""

data = [
    {'text': '通用名称：非诺贝特胶囊商品名称：力平之/LIPANTHYL英文名称：Fenofibrate Capsules汉语拼音：Feinuobeite Jiaonang',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药品名称】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '化学名称为：2-[4-(4-氯苯甲酰基)苯氧基]-2-甲基-丙酸甲基乙酯',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【成份】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '化学结构式：',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【成份】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '分子式： $\\mathrm { C } _ { 2 0 } \\mathrm { H } _ { 2 1 } \\mathrm { C l } 0 _ { 4 }$ 分子量：360.84',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【成份】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '本品为硬胶囊，内容物为白色粉末。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【性状】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '供成人使用',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【适应症】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '用于治疗成人饮食控制疗法效果不理想的高胆固醇血症(IIa型)，内源性高甘油三酯血症，单纯型(IV型)和混合型(IIb和II型)。特别是饮食控制后血中胆固醇仍持续升高，或是有其他并发的危险因素时。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【适应症】', 'page_idx': 0, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '在服药过程中应继续控制饮食。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【适应症】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '目前，尚无长期临床对照研究证明非诺贝特在动脉粥样硬化并发症一级和二级预防方面的有效性。  \n尚未证明非诺贝特能降低2型糖尿病患者的冠心病发病率和死亡。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【适应症】', 'page_idx': 0, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '200mg【用法用量】',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【规格】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '配合饮食控制，该药可长期服用，并应定期监测疗效。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【规格】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '每日服用一粒，与餐同服。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【规格】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-当胆固醇的水平正常时，建议减少剂量。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【规格】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-肾功能受损患者：轻中度肾功能受损患者建议从较小的起始剂量开始使用，然后根据对肾功能和血脂的影响，进行剂量调整。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【规格】', 'page_idx': 0, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '在治疗过程中最常报告的药物不良反应为消化、胃肠道不适。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 0, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '在安慰剂对照临床试验过程中 $\\scriptstyle ( \\ n = 2 3 4 4 )$ 和上市后(未知频率)，发现了下列伴有如下指定频率的不良反',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 0, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '应：',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 1, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '<table><tr><td colspan="8"></td></tr><tr><td>MedDra系统 器官分类</td><td></td><td>&gt;1/100, &lt;1/10</td><td>不常见 &gt;1/1,000, &lt;1/100 acare</td><td>罕见 &gt;10,000, &lt;1/1,000</td><td>极罕见 &lt;1/10,0 00包括 单独报 告</td><td>上市后观察到 未知频率(无法 根据现有数据 估算）</td></tr><tr><td>血液淋巴系</td><td></td><td></td><td></td><td>血红白和白</td><td></td><td></td></tr><tr><td>免疫系统疾病</td><td></td><td></td><td>头痛</td><td>过敏</td><td></td><td></td></tr><tr><td>神经系统疾病 血管疾病</td><td></td><td></td><td>血栓栓塞(肺</td><td></td><td></td><td>疲劳</td></tr><tr><td></td><td></td><td></td><td>栓塞、深静 脉血栓)**</td><td></td><td></td><td></td></tr><tr><td>呼吸、胸和纵 膈疾病</td><td></td><td></td><td>are</td><td></td><td>a</td><td>间质性肺病</td></tr><tr><td>胃肠道疾病</td><td></td><td>胃肠道症状和指 征(腹痛、恶心 呕吐、腹泻、 胃肠胀气)</td><td></td><td>iphy</td><td></td><td></td></tr><tr><td>肝胆疾病</td><td></td><td>血清转氨酶升高 (见[注意事项])</td><td>胆结石(见 [注意事项])</td><td>肝炎</td><td></td><td>黄疸、胆结石 并发症（例如胆 囊炎、胆管炎 胆绞痛)</td></tr><tr><td>皮肤和皮下组 织疾病 ipharma 杭州</td><td></td><td>ipk</td><td>皮肤过敏(例 如：皮疹、 瘙痒症、荨 麻疹)</td><td>脱发 光敏反应</td><td></td><td>重度影响皮肤 的反应(例如 ：多形性红斑 史蒂文斯-约 逊 死溶解)</td></tr><tr><td>肌肉骨骼、结 缔组织和骨病 症</td><td></td><td></td><td>肌障碍(例如 肌痛、肌 炎、肌肉痉 挛和无力)</td><td></td><td></td><td>横纹肌溶解</td></tr><tr><td>生殖系统和乳 腺疾病 实验室检查</td><td></td><td></td><td>性功能障碍</td><td></td><td></td><td></td></tr><tr><td>pharma</td><td>血同型半胱氨酸水平升高</td><td></td><td>血清肌酐增 加 macare 州逸曜</td><td>血尿素增加</td><td>杭州</td><td></td></tr></table>\n应：[]\n[]\n图片路径：images/a6fef0e33a0c34291cf9c162a57081455b4a69966cbf91ddf2af710a79492b05.jpg',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 1, 'type': 'table',
                     'img_path': 'images/a6fef0e33a0c34291cf9c162a57081455b4a69966cbf91ddf2af710a79492b05.jpg',
                     'table_id': None, 'table_caption': '应：', 'image_id': '', 'image_caption': ''}},
    {'text': '\\*FIELD研究是在9795名2型糖尿病患者中开展的一项随机安慰剂对照的研究。在研究中观察到，服用',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 1, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '非诺贝特的患者较服用安慰剂的患者，发生胰腺炎的病例数有统计学意义的显著增加(0.8%： 0.5%；p=0.031)。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '\\*\\*肺栓塞发病率有统计学意义的显著增加(安慰剂组 $0 . 7 \\%$ ：非诺贝特组1.1%；p=0.022)，深静脉血栓发病率有统计学无显著意义的增加(安慰剂组1.0% (48/4900患者)：非诺贝特组1.4% (67/4895患者)；p=0.074)。 ca',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 2, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '\\*\\*\\*非诺贝特治疗组患者的血同型半胱氨酸水平平均增加6.5μ mol/L，且在治疗停止后可恢复至正常水平。静脉血栓事件风险增加可能与同型半胱氨酸酸水平升高相关。尚不清楚该结果的临床意义。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【不良反应】', 'page_idx': 2, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '在下列情况中，此药物禁止使用：',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-对非诺贝特或本品辅料过敏者；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '由于存在过敏反应的风险，不应给对花生或花生油或大豆卵磷脂或相关产品过敏的患者服用；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-肝功能不全者，包括原发性胆汁性肝硬化，以及不明原因持续性肝功能异常患者；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-严重肾功能受损患者，包括接受透析的患者；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-已知在治疗过程中使用非诺贝特或与之结构相似的药物，尤其是酮洛芬时，会出现光毒性或光敏反应；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-已知有胆囊疾病患者；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-慢性或急性胰腺炎，重症高甘油三酯血症引起的急性胰腺炎除外；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-孕妇与哺乳期妇女；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-与其他贝特类药物合用(详见[药物相互作用])。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【禁忌】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '警告',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '骨骼肌：曾有使用贝特类和其他降血脂药发生肌毒性，包括罕见伴随或不伴随肾功能衰竭的横纹肌溶解病例的报告。如果有低蛋白血症以及之前曾有肾功能不全，这种疾病的发生率会增加。有肌病和/或横纹肌溶解易感因素(包括年龄大于70岁，有遗传性肌病的个人或家族史、肾功能受损、糖尿病、甲状腺功能减退、以及大量摄入酒精)的患者，发生横纹肌溶解的风险可能增高。 $1 0 0 - 1 0 0 0$ 对于出现弥漫性肌肉痛、肌炎、肌痛性肌肉痉挛、肌无力、伴或不伴肌源性CPK明显增高(超过正常5倍以上)的患者，应怀疑是否出现肌毒性，对这样的病例，应停止使用非诺贝特。使用非诺贝特的患者出现上述症状应当立即报告医生。观察性研究发现，当贝特类降脂药，特别是吉非罗齐，与H MG-CoA还原酶抑制剂(他汀类)联合使用时横纹肌溶解的风险增高。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 2, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '由于处方中乳糖的存在，本品禁用于患有先天性半乳糖症，葡萄糖或半乳糖吸收障碍综合征，或乳糖酶缺乏症患者。 cal car',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 2, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '使用注意事项',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-如果在服用几个月(3-6个月)后，血脂未得到有效的改善，应考虑补充治疗或采用其他方法治疗。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 2, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-肝功能：与使用其他降脂药物一样，一些病人使用非诺贝特后可能引起转氨酶(AST或ALT)升高，通常为一过性的、轻微或无症状的。有报告非诺贝特数周至数年的治疗中发生的肝细胞性、慢性活动性、胆汁淤积性肝炎，极为罕见有慢性活动性肝炎相关的肝硬化。建议：',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 2, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '●在治疗的最初12个月，每隔3个月检查转氨酶水平；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '●应特别注意转氨酶升高的患者，当AST和ALT升高至正常值的3倍以上时，应停止治疗；',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '●如果发生提示肝炎的症状(例如：黄疸、瘙痒症)，而且实验室检查确认肝炎诊断，应停止非诺贝特治疗。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-肾功能：有非诺贝特治疗后血清肌酐升高的报告，停药后趋向于回复到基线水平。肌酐升高的临床意义尚不清楚。对于原有肾功能受损患者、老年和糖尿病患者，建议定期监测肾功能。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-胰腺炎：在接受非诺贝特治疗的病人中有报道胰腺炎的病例。这可能是由于对严重高甘油三酯血症的病人缺乏疗效，或者由于药物的直接作用，或者继发于胆结石形成或者胆汁淤积阻塞胆管。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-胆石症：非诺贝特可能增加胆固醇的分泌进入胆汁，可能导致胆石症。如果怀疑胆石症，应做胆囊检查。如果确诊胆石症，应当停止使用非诺贝特。 ha',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-非诺贝特与香豆素类口服抗凝剂合用时，可能会增强后者的抗凝效应。为了避免出血并发症，应密切监测PT和INR，并可能需要调整口服抗凝剂的剂量(见药物相互作用)。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-在考虑非诺贝特的治疗之前，应对高脂血症的继发原因进行充治疗，例如：未控制的2型糖尿病、甲状腺功能减退、肾病综合症、蛋白异常血症、阻塞性肝病或酗酒。药物治疗有关的继发性高胆固醇血症见于：利尿剂、β 受体阻滞剂、雌激素、孕激素、复方口服避孕药、免疫抑制剂和蛋白抑制剂。在这些情况下，应该查明高脂血症是原发性还是继发性(血脂水平的升高可能是由口服雌激素造成的)。 2',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-血液学改变：非诺贝特治疗的患者中观察到轻中度的血红蛋白、红细胞压积，以及白细胞的下降，然而通常在长期治疗过程中维持稳定的水平。有过非诺贝特治疗的患者发生血小板减少症和粒细胞减少症的报告。建议在非诺贝特治疗的最初12个月定期监测血液红细胞、白细胞计数。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-过敏反应：有过非诺贝特治疗的患者发生Stevens-Johnson综合症和需要住院使用皮质激素治疗的毒性表皮坏死的病例报告。报道风疹的发生率在非诺贝特治疗组与安慰剂组分别为1.1%和0%；潮红的发生率在非诺贝特治疗组与安慰剂组分别为1.4%和0.8%。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-深静脉血栓性疾病：FIELD是在9795名2型糖尿病患者中开展的一项随机安慰剂对照的研究。研究中，4900例患者入组安慰剂组，而4895例患者入组非诺贝特组。肺栓塞发病率在非诺贝特组(1.1%)高于安慰剂组(0.7%)，p=0.022。深静脉血栓发病率分别为非诺贝特组1.4%，安慰剂组1.0%，p=0.074。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【注意事项】', 'page_idx': 3, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '生育力',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【孕妇和哺乳期妇女用药】', 'page_idx': 3,
                  'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                  'image_caption': ''}},
    {'text': '在动物试验中观察到非诺贝特对生育力的影响是可逆的。尚无本品影响生育力的临床数据。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【孕妇和哺乳期妇女用药】', 'page_idx': 3,
                  'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                  'image_caption': ''}}, {'text': '孕期', 'metadata': {'drug_name': '非诺贝特胶囊[188249]',
                                                                       'chapter_name': '【孕妇和哺乳期妇女用药】',
                                                                       'page_idx': 3, 'type': 'text', 'img_path': '',
                                                                       'table_id': '', 'table_caption': '',
                                                                       'image_id': '', 'image_caption': ''}},
    {'text': '-孕妇使用非诺贝特的数据尚不充分。动物试验结果显示未见有致畸作用。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【孕妇和哺乳期妇女用药】', 'page_idx': 3,
                  'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                  'image_caption': ''}},
    {'text': '-到目前为止，临床尚未出现致畸和胚胎毒性。但对孕期使用非诺贝特的跟踪不足以排除任何危险，故一般孕妇应禁用。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【孕妇和哺乳期妇女用药】', 'page_idx': 3,
                  'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                  'image_caption': ''}}, {
        'text': '-贝特类药物不用于孕妇，但通过饮食控制不能有效降低高甘油三酯血症 $( \\mathrm { { > } 1 0 g / L ) }$ 而增加母体患急性胰腺炎危险的情况时除外。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【孕妇和哺乳期妇女用药】', 'page_idx': 3,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {'text': '哺乳期', 'metadata': {'drug_name': '非诺贝特胶囊[188249]',
                                                                            'chapter_name': '【孕妇和哺乳期妇女用药】',
                                                                            'page_idx': 4, 'type': 'text',
                                                                            'img_path': '', 'table_id': '',
                                                                            'table_caption': '', 'image_id': '',
                                                                            'image_caption': ''}}, {
        'text': '目前尚无非诺贝特和/其代谢产物是否可进入母乳的资料，不能排除对母乳喂养的婴儿产生风险。因此哺乳期不建议使用本品。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【孕妇和哺乳期妇女用药】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {
        'text': '尚未确定非诺贝特在儿童和18岁以下青少年中的安全性和疗效。目前尚无相关数据。因此不建议18岁以下的患者使用本品。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【儿童用药】', 'page_idx': 4, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '老年人的剂量选择取决于肾功能状态。肾功能正常的老年人通常不需要调整剂量。如有肾功能受损可以减少剂量。使用非诺贝特的老年患者可以进行肾功能的监测。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【老年用药】', 'page_idx': 4, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '禁止合并使用其他贝特类药物：增加不良反应如横纹肌溶解症和两种分子间的药效拮抗作用的发生率。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '观察性研究发现，当贝特类降脂药，特别是吉非罗齐，与H MG-CoA还原酶抑制剂(他汀类)联合使用时横纹肌溶解的风险增高。除非调脂治疗的获益可能超过其风险，应避免联合使用贝特类与他汀类，在原有肌肉疾病的情况下尤其如此。若经评估后确定获益大于其风险，在无任何肌肉疾病史的混合型血脂异常伴心血管高危因素的严重患者才能同时处方非诺贝特与H MG-CoA还原酶抑制剂，并应密切监测患者的潜在肌肉毒性。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {
        'text': '香豆素类口服抗凝剂：非诺贝特与香豆素类口服抗凝剂合用时，非诺贝特能够与血浆白蛋白结合紧密，从蛋白结合部位置换出抗凝剂，会增强后者的抗凝效应，使PT和INR进一步延长。为了避免出血并发症，合用非诺贝特时，应当减低口服抗凝剂的剂量，更频繁地监测PT和INR直至达到稳定。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {
        'text': '免疫抑制剂：免疫抑制剂例如环孢素、他克莫司具有肾毒性，会减低肌酐清除率并升高血清肌酐。由于贝特类包括非诺贝特主要以肾脏分泌为主要排泄途径，免疫抑制剂与非诺贝特的相互作用可能导致肾功能的恶化。应当慎重权衡联合使用非诺贝特与免疫抑制剂的风险和获益；如果必需使用则应当使用最小有效剂量，并监测肾功能。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {
        'text': '胆酸结合剂：胆酸结合剂会与同时服用的药物结合，因此，应当至少在服用胆酸结合剂前1小时或者后4-6小时在服用非诺贝特，以避免阻碍非诺贝特的吸收。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}},
    {'text': '秋水仙碱：有报告非诺贝特合用秋水仙碱发生包括横纹肌溶解在内的肌病，两者合用要谨慎。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '格列酮类：在合用非诺贝特和格列酮类的过程中，报告了可逆性ΗDL-胆固醇反常降低的一些病例。因此，如果在其中-种药物基础上加用另一种药物，建议对HDL-胆固醇进行监测，并在HDL-胆固醇太低时停止其中一种治疗。 $\\times 3 = 1 8 0 ( c m )$',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {'text': '细胞色素P450酶：',
                                             'metadata': {'drug_name': '非诺贝特胶囊[188249]',
                                                          'chapter_name': '【药物相互作用】', 'page_idx': 4,
                                                          'type': 'text', 'img_path': '', 'table_id': '',
                                                          'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '以人肝肝微粒体进行的体外研究表明：非诺贝特和非诺贝特酸不是细胞色素(CYP)P450亚型CYP3A4、Cyp2D6、CYP2E1、或CYP1A2的抑制剂。在治疗浓度下，它们是CYP2C19和CYP2A6的弱抑制剂，是CYP2C9的弱至中度抑制剂。  \n对于合用非诺贝特和经CYP2C19、CYP2A6、特别是CYP2C9代谢的治疗指数窄的药物的患者，应对其进行谨慎监测，如有必要，建议对这些药物进行剂量调整。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物相互作用】', 'page_idx': 4,
                     'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '',
                     'image_caption': ''}}, {
        'text': '非诺贝特过量没有特殊治疗，可给予对症治疗。如果发生过量，给予一般支持性的护理，包括监测生命体征，观察临床状况。必要时给予催吐或者洗胃排出未被吸收的药物，要注意维持呼吸道通畅',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物过量】', 'page_idx': 4, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '。因为非诺贝特酸与血浆白蛋白结合紧密，血液透析不能清除非诺贝特酸，过量时不应考虑使用血透。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药物过量】', 'page_idx': 5, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '力平之非诺贝特片(II)160mg与力平之非诺贝特微粒化胶囊200mg具有生物等效性。原发性高胆固醇血症(杂合子家族性和非家族性)和混合型血脂异常',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '在4项随机、双盲、安慰剂对照、平行组研究中评估了非诺贝特160mg每日剂量的作用，这些研究纳入了具有以下平均基线血脂值的患者：总胆固醇306.9 mg/dL。LDL-C 213.8 mg/dL、HDL-C 52.3mg/dL和甘油三酯191.0mg/dL。相比安慰剂，非诺贝特治疗可显著降低甘油三酯(TG)、LDLC和总胆固醇水平(P=<0.05)，且.显著升高HDLC水平 $\\left( \\mathrm { p } { = } \\langle 0 . 0 5 \\right)$ (参见表1)。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '表1治疗结束时血脂参数的平均百分比变化',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '<table><tr><td>治疗组</td><td>总胆固醇</td><td>LDL-C</td><td>H DL-C</td><td>甘油三酯</td></tr><tr><td>合并队列</td><td></td><td></td><td></td><td>e</td></tr><tr><td>平均基线血脂值306.9mg/dL (n=646)</td><td></td><td>213.8 m g/dL</td><td>52.3 m g/dL</td><td>191.0m g/dL</td></tr><tr><td>非诺贝特治疗组-18.7%* (n=361)</td><td></td><td>20.6%</td><td>+11.0%* pha</td><td>-28.9%*</td></tr><tr><td>安慰剂(n=285)</td><td>-0.4%</td><td>-2.2%</td><td>+0.7%</td><td>+7.7%</td></tr><tr><td>基线LDL- C &gt;160m g/dL, 且 T G &lt;150mg/dL</td><td></td><td></td><td></td><td></td></tr><tr><td>平均基线血脂值307.7mg/dL (n=334)</td><td></td><td>227.7 m g/dL</td><td>58.1 m g/dL</td><td>101.7 m g/dL C</td></tr><tr><td>非诺贝特治疗组 (n=193)</td><td>-22.4%*</td><td>-31.4%*</td><td>+9.8%* Var</td><td>-23.5%*</td></tr><tr><td>安慰剂(n=141) 基线LDL-</td><td>+0.2%</td><td>-2.2%</td><td>+2.6%</td><td>+11.7%</td></tr><tr><td>C &gt;160mg/dL, 且 T G≥150m g/dL</td><td></td><td></td><td></td><td></td></tr><tr><td>平均基线血脂值312.8mg/dL (n=242)</td><td></td><td>219.8 m g/dL re</td><td>46.7 m g/dL</td><td>231.9m g/dL</td></tr><tr><td>非诺贝特治疗组-16.8%* (n=126)</td><td></td><td>-20.1%* 曜</td><td>+14.6%*</td><td>-35.9%*</td></tr><tr><td>安慰剂(n=116)</td><td>-3.0%</td><td>-6.6%</td><td>+2.3%</td><td>+0.9%</td></tr><tr><td colspan="5">研究治疗的持续时间为3-6个月。 *p=&lt;vs.安慰剂</td></tr></table>\n表1治疗结束时血脂参数的平均百分比变化[]\n[]\n图片路径：images/229088a9cf007cbf63138e53a3a17c36cb7996fbc05478769e48801f835023dc.jpg',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'table',
                     'img_path': 'images/229088a9cf007cbf63138e53a3a17c36cb7996fbc05478769e48801f835023dc.jpg',
                     'table_id': '表1', 'table_caption': '表1治疗结束时血脂参数的平均百分比变化', 'image_id': '',
                     'image_caption': ''}}, {
        'text': '在受试者亚组中测定了apo B水平。与安慰剂相比，非诺贝特治疗后apoB水平相对于基线时显著下降（分别为 $- 2 5 . 1 \\%$ Vs. $2 . 4 \\%$ , $\\mathrm { p } { < } 0 . 0 0 0 1$ , $\\mathrm { n } = 2 1 3$ 和143)。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '高甘油三酯血症',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '在147例高甘油三酯血症患者中进行的两项随机、双盲、安慰剂对照临床试验研究了非诺贝特对血清甘油三酯的影响。患者依据研究方案接受8周治疗，两项研究的差异仅在于一项入组基线TG水平在500-1500mg/dL范围的患者，而另一项入组TG水平在350-500 mg/dL之间的患者。在高甘油三酯血症和正常胆固醇血症伴或不伴高乳糜微粒血症的患者中，非诺贝特160 mg每日剂量可显著降低TG、极低密度脂蛋白甘油三酯(VLDL-TG)、总胆固醇和极低密度脂蛋白胆固醇(VLDL-C)水平(p=<0.05)，且显著升高HDL-C水平 $\\left( \\mathrm { p } { = } { < } 0 . 0 5 \\right)$ 。治疗甘油三酯升高患者时通常可引起LDLC升高(参见表2)。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 5, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '表2非诺贝特对高甘油三酯血症患者的影响',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '<table><tr><td>研究1</td><td colspan="4">安慰剂</td><td colspan="4">非诺贝特</td></tr><tr><td>基线TG水平 350-499 m g/dL</td><td>N</td><td>基线 (均值)</td><td>终点 (均值)</td><td>%变化 (均值)</td><td>N</td><td>基线(均 值)</td><td>终点(均 值)</td><td>%变化 (均值)</td></tr><tr><td>甘油三酯</td><td>28</td><td>449</td><td>450</td><td>-0.5</td><td>27</td><td>432</td><td>223</td><td>-46.2*</td></tr><tr><td>VLDL-TG</td><td>19</td><td>367</td><td>350</td><td>2.7</td><td>19</td><td>350</td><td>178</td><td>-44.1*</td></tr><tr><td>总胆固醇</td><td>28</td><td>255</td><td>261</td><td>2.8</td><td>27</td><td>252</td><td>227</td><td>-9.1*</td></tr><tr><td>HDL-C</td><td>28</td><td>35</td><td>36</td><td>4$</td><td>27</td><td>34</td><td>40</td><td>19.6*</td></tr><tr><td>LDL-C</td><td>28</td><td>120</td><td>129</td><td>12</td><td>27</td><td>128</td><td>137</td><td>14.5</td></tr><tr><td>VLDL-C</td><td>27</td><td>99</td><td>99</td><td>15.8</td><td>27</td><td>92</td><td>46</td><td>-44.7*</td></tr><tr><td>研究2</td><td colspan="4">安慰剂</td><td colspan="4">非诺贝特</td></tr><tr><td>基线TG水平 500-1500m g/dL</td><td>N</td><td>基线 (均值)</td><td>终点 (均值)</td><td>%变化 (均值)</td><td>N</td><td>基线(均 值)</td><td>终点(均 值)</td><td>%变化 (均值)</td></tr><tr><td>甘油三酯</td><td>44</td><td>710</td><td>750</td><td>17.2</td><td>48</td><td>726</td><td>308</td><td>-54.5*</td></tr><tr><td>VLDL-TG</td><td>29</td><td>537</td><td>571</td><td>18.7</td><td>33</td><td>543</td><td>205</td><td>-50.6*</td></tr><tr><td>总胆固醇</td><td>44</td><td>272</td><td>271</td><td>0.4</td><td>48</td><td>261</td><td>223</td><td>-13.8*</td></tr><tr><td>HDL-C</td><td>44</td><td>27</td><td>28</td><td>5.0</td><td>48</td><td>30</td><td>36</td><td>22.9*</td></tr><tr><td>LDL-C</td><td>42</td><td>100</td><td>90</td><td>-4.2</td><td>45</td><td>103</td><td>131</td><td>45.0*</td></tr><tr><td>VLDL-C</td><td>42</td><td>137</td><td>142</td><td>11.0</td><td>45</td><td>126</td><td>54</td><td>-49.4*</td></tr><tr><td colspan="9">*p=&lt;0.05vs.安慰剂</td></tr></table>\n表2非诺贝特对高甘油三酯血症患者的影响[]\n[]\n图片路径：images/491f3bc6c192def7feb8d968aa0c8c29511782884b1f4887f1e678660fc7cd00.jpg',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'table',
                     'img_path': 'images/491f3bc6c192def7feb8d968aa0c8c29511782884b1f4887f1e678660fc7cd00.jpg',
                     'table_id': '表2', 'table_caption': '表2非诺贝特对高甘油三酯血症患者的影响', 'image_id': '',
                     'image_caption': ''}}, {'text': '长期临床对照研究',
                                             'metadata': {'drug_name': '非诺贝特胶囊[188249]',
                                                          'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                                                          'img_path': '', 'table_id': '', 'table_caption': '',
                                                          'image_id': '', 'image_caption': ''}},
    {'text': '目前，尚无长期临床对照研究证明非诺贝特在动脉粥样硬化并发症一级和二级预防方面的有效性。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '非诺贝特干预与降低糖尿病事件(FIELD)研究是一项全球多中心、随机、对照研究，共纳入9795名2型糖尿病患者，接受非诺贝特单药治疗或安慰剂治疗，主要研究终点为冠状动脉事件(包括冠心病死亡或非致命的心肌梗死)。非诺贝特组与安慰剂组相比，冠状动脉事件发生风险未显示出显著性差异(相对风险下降11%；风险比[HR]0.89,95%置信区间：0.75-1.05; p=0.16)。TG升高(≥2.3m mol/L)且伴低HDL-C(男性>1.03 m mol/L，女性>1.29 m mol/L)患者的亚组分析提示，与安慰剂组相比，非诺贝特组心血管事件风险相对下降27%(风险比[HR]0.73，95%置信区间：0.58-0.91，p=0.005)。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '控制糖尿病患者心血管风险的行动(ACCORD)血脂研究是一项随机、安慰剂对照研究，其中涉及5518名2型糖尿病患者，使用非诺贝特联合辛伐他汀进行治疗。对于非致命性心肌梗死、非致命性卒中以及心血管死亡的复合主要终点，非诺贝特加辛伐他汀的疗法与辛伐他汀单一疗法相比未显示出任何显著差异(风险比[HR]0.92,95%置信区间0.79-1.08，p=0.32；绝对风险减少：0.74%）。对于预设的血脂异常患者亚组(定义为那些基线时处于HDL-C的最低三分位(≤34mg/dl或0.88m mol/L)和TG的最高三分位(≥204mg/dl或23m mol/L)的患者)，与辛伐他汀单疗法相比，非诺贝特加辛伐他汀治疗的复合主要终点显示出31%的相对减少(风险比[HR]0.69，，95% 置信区间：0.49-0.97，p=0.03;绝对风险减少：4.95%)。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '药理作用',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '非诺贝特可降低血清胆固醇20-25%，降低甘油三酯 $4 0 - 5 0 \\%$ 。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-胆固醇的降低是通过降低低密度动脈粥样化成分(VILDL和LDL)，并且通过降低总胆固醇/HDL胆固醇比率取得的(该比率在动脉粥样化高脂血症中升高)，从而改善了血浆中胆固醇的分布。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 6, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-高胆固醇和动脉粥样硬化的关系，以及动脉硬化与冠状动脉疾病危险的关系已得到证实。低水平的HDL可增加冠状动脉疾病危险。甘油三酯升高可增加心血管疾病危险，但还不能确定这种关系是独立存在的。另外，甘油三酯可能不仅与动脉粥样化有关，而且与血栓形成有关。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-通过有效延长治疗期(显著降低胆固醇)，血管外胆固醇的沉积(腱和结节黄瘤)能够有明显的消退，甚至完全消除。 can',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-在高脂血症病人中非诺贝特有利尿酸的作用，可使血浆中尿酸平均降低25%。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '-非诺贝特治疗增加apoA1、降低apoB，从而改善apoA1/apoB比率，该比值被认为是动脉粥样化的标志。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-动物研究和人体临床研究表明，非诺贝特具有抗血小板凝集的作用，该作用是通过降低ADP、花生四烯酸和肾上腺素所致的凝集反应而实现的。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '-非诺贝特通过激活PPARα(过氧化物酶增殖体激活受体α），激活脂蛋白脂酶和减少载脂蛋白CⅢ合成，使血浆中脂蛋白颗粒降解和甘油三酯清除明显增加。PPARα 的激活也导致载脂蛋白AI和AⅡI合成的增加。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '毒理研究遗传毒性：',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '非诺贝特A mes试验、小鼠淋巴瘤试验、染色体畸变试验以及大鼠原代肝细胞的程序外DNA合成试验结果均为阴性。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '生殖毒性：',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '雌性大鼠从交配前15天开始，雄性大鼠从交配前61天开始，至交配结束，经口给予非诺贝特达300m/kg/天，按体表面积换算，相当于临床拟用最大剂量(MRH D)的10倍时，未见对生育力的影响；雌性大鼠在15mg/kg/天剂量时，按体表面积换算，相当于MRHD的0.3倍，可见母体毒性。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '大鼠妊娠第6-15天经口给予非诺贝特，14mg/kg/天剂量时，按体表面积换算，低于MRHD，未见对胚胎发育的影响；127、361mg/kg/天剂量时，可见母性毒性。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '兔妊娠第6-18天经口给予非诺贝特，15mg/kg/天剂量时，按体表面积换算，低于MRHD，未见对胚胎发育的影响；150mgkg/天剂量时，按体表面积换算，约为MRHD的10倍，可见流产。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '大鼠妊娠第15天到哺乳第21天经口给予非诺贝特15、75、300mg/kg/天，按体表面积换算，低于MRHD时，可见母体毒性。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '犬连续3个月经口给予非诺贝特酸(非诺贝特的人体主要代谢产物)25、50、75mg/kg/天，可见生殖系统可逆性变化，包括雄性睾丸空泡化和精子生成功能降低，罐性卵巢不成熟(黄体缺失)。 $\\sim$',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '致癌性：',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': 'W istar大鼠连续两年经口给予非诺贝特10、45、200mg/kg(按体表面积换算，分别约为MRHD的0.3、1、6倍)，剂量为MRHD的6倍时，可见所有动物肝脏肿瘤发生率增加，雄性动物胰腺肿瘤和良性睾丸间质细胞瘤发生率增加；为MRHD的1倍、6倍时，可见雄性胰腺肿瘤发生率增加。SD大鼠两年致癌性试验中经口给予非诺贝特10、600mg/kg/天(按体表面积换算，为MRHD的0.3倍和2倍)，可见所有动物胰腺腺泡细胞瘤发生率增加，为MRHD的2倍时，可见雄性动物睾丸间质细胞瘤发生率增加。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '小鼠连续21个月经口给予非诺贝特10、45、200mg/kg/天(按体表面积换算，分别约为MRHD的0.2倍、1倍、3倍)，其中为MRHD的3倍时，可见所有动物肝脏肿瘤发生率增加。小鼠连续18个月经口给予非诺贝特，按体表面积换算，为MRHD的3倍时，可见雄性动物肝脏肿瘤发生率增加和雌性动物肝脏腺瘤发生率增加。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 7, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '电镜结果显示，大鼠给予非诺贝特后可见过氧化物酶增殖，尚未有充分的研究对人体的影响，但其他贝特类药物给药前后，临床肝脏活检标本未见氧化物酶形态学和数量上的变化。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药理毒理】', 'page_idx': 8, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, {
        'text': '非诺贝特在血浆中未发现原型存在，吸收进入体内后，在肝脏迅速被酯酶水解成为活性代谢产物，主要活性代谢产物为非诺贝特酸。',
        'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药代动力学】', 'page_idx': 8, 'type': 'text',
                     'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '通常服药后5小时可达最大血浆浓度。每日服用一粒力平之微粒化非诺贝特胶囊后的平均血浆浓度约为15μg/ml。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药代动力学】', 'page_idx': 8, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '同一病人连续治疗，其血药浓度水平是稳定的。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药代动力学】', 'page_idx': 8, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '非诺贝特酸与血浆白蛋白结合紧密，可从蛋白结合部位取代维生素K拮抗剂，加强抗凝效果(详见[药物相互作用])。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药代动力学】', 'page_idx': 8, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}},
    {'text': '非诺贝特酸在血液中消除半衰期约为20小时。',
     'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药代动力学】', 'page_idx': 8, 'type': 'text',
                  'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}, ]
# data = [{'text': '通用名称：非诺贝特胶囊商品名称：力平之/LIPANTHYL英文名称：Fenofibrate Capsules汉语拼音：Feinuobeite Jiaonang', 'metadata': {'drug_name': '非诺贝特胶囊[188249]', 'chapter_name': '【药品名称】', 'page_idx': 0, 'type': 'text', 'img_path': '', 'table_id': '', 'table_caption': '', 'image_id': '', 'image_caption': ''}}]
process_datas = []
for item in data:
    vector_dense = embeddings.embed_query(item['text'])
    process_data = {
        "vector_dense": vector_dense,
        "text": item['text'],
        "metadata": item['metadata'],
    }
    process_datas.append(process_data)

res = client.insert(
    collection_name="milvus_test",
    data=process_datas,
    run_function=True  # 自动执行bm25_function，不设置则否则 vector_sparse 字段会是空的，稀疏索引用不起作
)

# ==========索引
# 索引是为了加快向量搜索，如果不建立索引，那么就会遍历全部向量，就会比较慢，建立索引就会快速定位相似向量，因此类型必须与你的向量类型匹配

#
# 索引类型选择：
# 稠密向量常用 IVF_FLAT（平衡速度和精度）或 HNSW（更高精度，适合实时场景）；
# 稀疏向量常用 SPARSE_INVERTED_INDEX。
# 索引参数：
# IVF_FLAT 需要指定 nlist（建议值：数据量的平方根，如 10 万条数据用 300-500）；
# HNSW 需要指定 M（邻居数量，默认 16）和 efConstruction（构建时的探索深度，默认 200）。
# 创建时机：索引必须在 “插入数据之后” 创建（索引是基于数据构建的）。
# M 控制索引的复杂度（建议 8-64，默认 16），efConstruction 控制索引质量（建议 50-200），值越大索引质量越高但构建越慢。
try:
    existing_index = client.describe_index(collection_name="milvus_test")
except:
    existing_index = None
if not existing_index:
    sparse_index = IndexParams()
    sparse_index.add_index("vector_sparse", "SPARSE_INVERTED_INDEX", metric_type="BM25")

    dense_index = IndexParams()
    dense_index.add_index("vector_dense", "HNSW", metric_type="COSINE", M=16, efConstruction="100")

    # 调用创建索引方法
    client.create_index(
        collection_name="milvus_test",
        index_params=sparse_index  # 传递 IndexParams 实例
    )
    client.create_index(
        collection_name="milvus_test",
        index_params=dense_index  # 传递 IndexParams 实例
    )

# =================检索
query_text = "并发症"
query_text_dense = embeddings.embed_query(query_text)
query_text_embedding = embeddings_bge.encode(
        sentences=query_text,# Union[List[str], str],
        batch_size=15,# Optional[int] = None,
        # max_length="",# Optional[int] = None,
        return_dense=True,# Optional[bool] = None,
        return_sparse=True,# Optional[bool] = None,
        return_colbert_vecs=True,# Optional[bool] = None,
    )
print(query_text_embedding)

def dense_search(client: MilvusClient, query_dense_embedding: list, collection_name: str, limit: int = 5,
                 output_fields: list[str] = None, timeout=None):
    search_params = {"efSearch": 64}
    res = client.search(
        collection_name=collection_name,
        data=[query_dense_embedding],
        limit=limit,
        output_fields=output_fields,
        metric_type="COSINE",
        timeout=timeout,
        anns_field="vector_dense",
        search_params=search_params
    )
    return res


def sparse_search(client: MilvusClient, query_sparse_embedding: list, collection_name: str, limit: int = 5,
                  output_fields: list[str] = None, timeout: float = None):
    res = client.search(
        collection_name=collection_name,
        data=query_sparse_embedding,
        limit=limit,
        output_fields=output_fields,
        metric_type="IP",
        timeout=timeout,
        anns_field="vector_sparse",
    )
    return res


def hybrid_search(client, query_dense_embedding, query_sparse_embedding, collection_name, sparse_weight, dense_weight,
                  output_fields, timeout: float = None, limit=5):
    dense_params = {"metric_type": "HNSW", "params": {"efSearch": 64}}
    sparse_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        data=query_dense_embedding,  # Union[List, utils.SparseMatrixInputType],
        anns_field="vector_dense",  # str,
        param=dense_params,  # Dict,
        limit=limit  # int,
        # expr=""#Optional[str] = None,
        # expr_params=#Optional[dict] = None,
    )
    sparse_req = AnnSearchRequest(
        data=query_sparse_embedding,  # Union[List, utils.SparseMatrixInputType],
        anns_field="vector_sparse",  # str,
        param=sparse_params,  # Dict,
        limit=limit  # int,
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = client.hybrid_search(
        reqs=[sparse_req, dense_req],
        collection_name=collection_name,
        ranker=rerank,
        output_fields=output_fields,
        timeout=timeout,
        limit=limit

    )
    return res


# 加载进内存，加快计算速度
client.load_collection(collection_name="milvus_test")
pprint(res)
res = client.search(
    # 目标集合的名字
    collection_name="milvus_test",
    # 需要查询的向量列表
    data=[query_text_dense],
    # 类似于sql查询语句"metadata['category'] == 'technology' and length(text) > 100"
    # filter=,
    # 返回 Top 5 条相似结果
    limit=5,
    # 指定返回的字段（按需选择，避免返回过多数据）
    output_fields=["text", "metadata"],
    # 搜索时的具体参数（与索引类型相关），例如 HNSW 索引可设置 {"efSearch": 64}（控制搜索精度和速度的平衡），默认使用创建索引时的参数
    # search_params: Optional[dict] = None,#控制检索行为，比如搜索范围，搜索精度，
    # timeout: Optional[float] = None,
    # 指定搜索的分区名称列表（如 ["partition_2023", "partition_2024"]），适用于按分区存储数据的场景，可缩小搜索范围。
    # partition_names: Optional[List[str]] = None,
    # 与查询向量匹配的字段，即我需要查询的向量应该与那个字段计算相似度
    anns_field="vector_dense",
    # ranker: Optional["Function"] = None,
    # 相似性度量方式（bge-m3 向量推荐用 COSINE，与 normalize_embeddings=True 匹配） ，
    # "COSINE"余余弦相似度，取值范围[-1,1] 值越大越相似
    # "L2" 欧氏距离，值越小越相似，
    # "IP"内积，值越大越相似
    metric_type="COSINE"  # 与索引的metric_type一致
)

res = dense_search(client,query_text_dense,"milvus_test",output_fields=["text", "metadata"])
for i in res:
    for j in i:
        print(f"{j.get('distance')}------{j.get('entity')}")
# 释放内容

client.release_collection(collection_name="milvus_test")



# ====langchain提供的milvus接口
