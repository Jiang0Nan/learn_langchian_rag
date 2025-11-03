import threading
from typing import Dict

import numpy as np
from scipy.sparse import csr_array,vstack

class Embedding:
    def __init__(self,model_name_or_path:str = r'D:\files\models\bge-m3', batch_size:int = 16,return_dense:bool=True, return_sparse:bool=True,return_colbert_vecs:bool=False, normalize_embeddings: bool = True,):
        import  torch

        try :
            from FlagEmbedding import BGEM3FlagModel
            _model_config = {'model_name_or_path' : model_name_or_path,
                             'normalize_embeddings': normalize_embeddings,
                             'use_fp16' : torch.cuda.is_available(),
                             'device':  'cuda:0' if torch.cuda.is_available() else 'cpu',
                             }
            _encode_config = {
                'batch_size': batch_size,
                'return_dense': return_dense,
                'return_sparse': return_sparse,
                'return_colbert_vecs': return_colbert_vecs,

            }
            self._encode_config = _encode_config
            self.model = BGEM3FlagModel(**_model_config)
        except Exception as e:
            print(f'加载embedding模型失败{e}')
    # def single_encode(self,text):
    #     return self.model.encode(text)

    @property
    def dim(self) -> Dict:
        return {
            "dense": self.model.model.model.config.hidden_size,
            "colbert_vecs": self.model.model.colbert_linear.out_features,
            "sparse": len(self.model.tokenizer),
        }

    def encode(self,chunks:list,title_weight:float=0.1):
        try:
            import torch
            # 提取需要嵌入的文本
            titles,contents = [],[]
            for chunk in chunks:
                titles.append(chunk.get('filename','Title'))
                contents.append(chunk.get('content','None'))
            # 对文档的文件名进行embedding
            with torch.no_grad():
                emb_title = self.model.encode(titles[0:1],return_dense=True,return_sparse=False,return_colbert_vecs=False).get('dense_vecs')
            emb_title = np.concatenate([emb_title for _ in range(len(titles))])
            # 对块的内容进行Embedding
            with torch.no_grad():
                emb_all = self.model.encode(contents,**self._encode_config)

            # 按照比例生成最后的向量
            sparse_list = []
            res_dense_embed = emb_title * title_weight + emb_all.get('dense_vecs') * (1 - title_weight)
            if self._encode_config['return_sparse'] is True:
                    sparse_dim = self.dim["sparse"]
                    for sparse_vec in emb_all["lexical_weights"]:
                        indices = [int(k) for k in sparse_vec]
                        values = np.array(list(sparse_vec.values()), dtype=np.float64)
                        row_indices = [0] * len(indices)
                        csr = csr_array((values, (row_indices, indices)), shape=(1, sparse_dim))
                        sparse_list.append(csr)
                    # sparse_vecs = vstack([sparse_emb.reshape((1,-1)) for sparse_emb in sparse_list]).tocsr()

            for i ,result in enumerate(chunks):
                result['dense_vec'] = res_dense_embed[i].tolist()
                if self._encode_config['return_sparse'] is True:
                    assert sparse_list[i] is not None
                    result['sparse_vec'] = sparse_list[i]
                if self._encode_config['return_colbert_vecs'] is True:
                    result['colbert_vecs'] = emb_all['colbert_vecs'][i]


        except Exception as e:
            print(f"encode failed {e}")
if __name__ == '__main__':
    embedding = Embedding()
    test_texts = [
        "你好，世界！",
        "我的祖国繁荣昌盛。",
        "人工智能正在改变我们的生活。",
        "今天的天气非常好，适合出门散步。",
        "Python 是一门非常适合初学者的编程语言。",
        "我喜欢阅读科技新闻和历史书籍。",
        "计算机视觉和自然语言处理是热门研究方向。",
        "学习编程需要不断练习和思考。",
        "旅行能开阔视野，增长见识。",
        "保持健康的生活习惯非常重要。",
        "最近我在学习数据科学和机器学习。",
        "阅读能够提升语言表达能力和逻辑思维。",
        "人工智能可以辅助医疗诊断，提高效率。",
        "每天运动半小时，有助于保持身体健康。",
        "软件工程师需要掌握良好的编码习惯。",
        "我喜欢用Python做小工具来提高效率。",
        "历史是了解过去、把握现在的钥匙。",
        "互联网的发展让信息获取更加便捷。",
        "开源社区为开发者提供了丰富的资源。",
        "自然语言处理可以让计算机理解人类语言。",
        "我希望将来能做一名优秀的研究人员。",
        "团队合作是完成大型项目的重要保障。",
        "学习新技术需要持续关注行业动态。",
        "阅读论文是科研工作的重要环节。",
        "合理安排时间，可以提高学习和工作的效率。",
        "音乐可以调节情绪，放松身心。",
        "养成记录笔记的习惯，有助于知识积累。",
        "在编程中遇到问题，多查文档和资料是关键。",
        "数据分析能够帮助企业做出更科学的决策。",
        "深度学习在图像识别和语音识别中应用广泛。",
        "我喜欢用笔记软件整理学习内容。",
        "科技创新推动社会进步和发展。",
        "良好的沟通能力可以提升工作效率。",
        "阅读小说不仅能放松，也能增长见识。",
        "旅游时拍照记录美好瞬间是一个好习惯。",
        "Python 中有很多强大的库可以辅助开发。",
        "掌握算法和数据结构是成为程序员的基础。",
        "写作可以训练逻辑思维和表达能力。",
        "自律的人更容易在学习和工作中取得成果。",
        "每天坚持学习新知识，可以不断提升自己。",
        "开源项目可以让你快速积累实践经验。",
        "了解行业前沿，有助于职业发展规划。",
        "积极参与讨论和交流，可以提升认知水平。",
        "学习编程要注重理论与实践相结合。",
        "人工智能的应用越来越广泛，需要关注伦理问题。",
        "团队协作和独立思考同样重要。",
        "合理安排作息，有助于长期保持高效状态。",
        "编程调试能力是程序员必备的技能。",
        "阅读文档和源码可以提升技术水平。",
        "多尝试新技术，有助于拓宽技术视野。",
        "持续练习可以让技能熟能生巧。",
        "写总结和复盘是提升效率的好方法。"
    ]

    test_texts_2 = [{
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '4周或肾性死亡的时间酮 N=2833 N=2841非奈利酮/安慰剂酮 N=3686 N=3666非奈利酮/安慰剂至事件发生时间终点：事件发生率（100患者-年）事件发生率（100患者-年）风险比（95%CI）p值 事件发生率（100患者-年）事件发生率（',
    'tokenizer_contnet': '4 周 或 肾 性 死亡 的 时间 酮 n 2833 n 2841 非 奈 利 酮 安慰剂 酮 n 3686 n 3666 非 奈 利 酮 安慰剂 至 事件 发生 时间 终点 事件 发生率 100 患者 年 事件 发生率 100 患者 年 风险 比 95 ci p 值 事件 发生率 100 患者 年 事件 发生率',
    'tokenizer_fine_grained': '4周或肾性死亡的时间酮 N=2833 N=2841非奈利酮 安慰剂酮 N=3686 N=3666非奈利酮 安慰剂至事件发生时间终点：事件发生率（100患者-年）事件发生率（100患者-年）风险比（95%CI）p值 事件发生率（100患者-年）事件发生率（'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '值 事件发生率（100患者-年）事件发生率（100患者-年）风险比（95%CI）p值复合终点：肾衰 竭、eGFR较基线值持续下降≥40%持续至少4周或肾性死亡7.6 9.1 0.82[0.73；0.93]0.0013.2 3.6 0.87[0.76；1.',
    'tokenizer_contnet': '值 事件 发生率 100 患者 年 事件 发生率 100 患者 年 风险 比 95 ci p 值 复合 终点 肾衰 竭 egfr 较 基线 值 持续 下降 40 持续 至少 4 周 或 肾 性 死亡 76 91 0 820 730 93 0 0013 23 60 87 0 76 1',
    'tokenizer_fine_grained': '值 事件发生率（100患者-年）事件发生率（100患者-年）风险比（95%CI）p值复合终点：肾衰 竭、eGFR较基线值持续下降≥40%持续至少4周或肾性死亡7.6 9.1 0.82[0.73；0.93]0.0013.2 3.6 0.87[0.76；1.'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '.0013.2 3.6 0.87[0.76；1.01]-肾衰竭 3.0 3.4 0.87[0.72；1.05]- 0.4 0.5 0.72[0.49；1.05]-eGFR较基线值下降≥40%持续至少4周7.2 8.7 0.81[0.72；0.92]- 3.0 3.5 0.87[0.75',
    'tokenizer_contnet': '0013 23 60 87 0 76 101 肾衰竭 3 0 34 0 87 0 72 1 05 0 40 50 720 49 1 05 egfr 较 基线 值 下降 40 持续 至少 4 周 72 8 7 0 81 0 720 92 3 0 35 0 87 0 75',
    'tokenizer_fine_grained': '.0013.2 3.6 0.87[0.76；1.01]-肾衰竭 3.0 3.4 0.87[0.72；1.05]- 0.4 0.5 0.72[0.49；1.05]-eGFR较基线值下降≥40%持续至少4周7.2 8.7 0.81[0.72；0.92]- 3.0 3.5 0.87[0.75'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '0.92]- 3.0 3.5 0.87[0.75；＞1.00]-肾性死亡 - - - - - - - -复合终点：心血 管死亡、非致死 性心肌梗死、非 致死性卒中或因心力衰竭住院5.1 5.9 0.86[0.75；0.99]0.0343.9 4.5 0.87',
    'tokenizer_contnet': '0 92 3 0 35 0 87 0 75 100 肾 性 死亡 复合 终点 心血 管 死亡 非 致死 性 心肌梗死 非 致死 性 卒 中 或 因 心力衰竭 住院 515 90 86 0 75 0 99 0 0343 94 50 87',
    'tokenizer_fine_grained': '0.92]- 3.0 3.5 0.87[0.75；＞1.00]-肾性死亡 - - - - - - - -复合终点：心血 管死亡、非致死 性心肌梗死、非 致死性卒中或因心力衰竭住院5.1 5.9 0.86[0.75；0.99]0.0343.9 4.5 0.87'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '75；0.99]0.0343.9 4.5 0.87     [0.76；0.98]0.02 6心血管死亡 1.7 2.0 0.86[0.68；1.08]- 1.6 1.7 0.90[0.74；1.09]-非致死性心肌梗死0.9 1.2 0.80[0.58；1.09]- 0.9 0.9 ',
    'tokenizer_contnet': '75 0 99 0 0343 94 50 87 0 76 0 98 0 02 6 心血管 死亡 172 0 0 86 0 68 108 161 7 0 90 0 74 1 09 非 致死 性 心肌梗死 0 91 20 800 58 1 09 0 90 9',
    'tokenizer_fine_grained': '75；0.99]0.0343.9 4.5 0.87 [0.76；0.98]0.02 6心血管死亡 1.7 2.0 0.86[0.68；1.08]- 1.6 1.7 0.90[0.74；1.09]-非致死性心肌梗死0.9 1.2 0.80[0.58；1.09]- 0.9 0.9'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '.80[0.58；1.09]- 0.9 0.9 0.99[0.76；1.31]-非致死性卒中 1.2 1.2 1.03[0.76；1.38]- 0.9 0.9 0.97[0.74;1.26]-因心力衰竭住院1.9 2.2 0.86[0.68；1.08]- 1.0 1.4 0.71[0.',
    'tokenizer_contnet': '800 58 1 09 0 90 90 99 0 76 131 非 致死 性 卒 中 121 21 03 0 76 138 0 90 90 97 0 74 1 26 因 心力衰竭 住院 19 2 20 86 0 68 108 101 40 71 0',
    'tokenizer_fine_grained': '.80[0.58；1.09]- 0.9 0.9 0.99[0.76；1.31]-非致死性卒中 1.2 1.2 1.03[0.76；1.38]- 0.9 0.9 0.97[0.74;1.26]-因心力衰竭住院1.9 2.2 0.86[0.68；1.08]- 1.0 1.4 0.71[0.'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '；1.08]- 1.0 1.4 0.71[0.56;0.90] -  图2：FIGARO-DKD研究中至首次发生肾衰竭、eGFR较基线值下降≥40%持续至少4周或肾性死亡的时间 图3：FIDELIO-DKD研究中至首次发生心血管死亡、非致死性心肌梗�',
    'tokenizer_contnet': '108 101 40 71 0 560 90 图 2 figaro dkd 研究 中至 首次 发生 肾衰竭 egfr 较 基线 值 下降 40 持续 至少 4 周 或 肾 性 死亡 的 时间 图 3 fidelio dkd 研究 中至 首次 发生 心血管 死亡 非 致死 性 心肌 梗',
    'tokenizer_fine_grained': '；1.08]- 1.0 1.4 0.71[0.56;0.90] - 图2：FIGARO-DKD研究中至首次发生肾衰竭、eGFR较基线值下降≥40%持续至少4周或肾性死亡的时间 图3：FIDELIO-DKD研究中至首次发生心血管死亡、非致死性心肌梗�'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '�管死亡、非致死性心肌梗死、非致死性卒中或因心力衰竭住院的时间  图4：FIGARO-DKD研究中至首次发生心血管死亡、非致死性心肌梗死、非致死性卒中或因心力衰竭住院的时间  在FIDELIO-DKD研究中，中国有372',
    'tokenizer_contnet': '管 死亡 非 致死 性 心肌梗死 非 致死 性 卒 中 或 因 心力衰竭 住院 的 时间 图 4 figaro dkd 研究 中至 首次 发生 心血管 死亡 非 致死 性 心肌梗死 非 致死 性 卒 中 或 因 心力衰竭 住院 的 时间 在 fidelio dkd 研究 中 中国 有 372',
    'tokenizer_fine_grained': '�管死亡、非致死性心肌梗死、非致死性卒中或因心力衰竭住院的时间 图4：FIGARO-DKD研究中至首次发生心血管死亡、非致死性心肌梗死、非致死性卒中或因心力衰竭住院的时间 在FIDELIO-DKD研究中，中国有372'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '的时间  在FIDELIO-DKD研究中，中国有372例患者接受了随机分组[非奈利酮组（10mg或20mg，每日一次）n=188，安慰剂组n=184]。非奈利酮组患者的平均治疗持续时间为2年。中国亚组的结果与整体研究结果基本一致。 在FIGARO-DKD�',
    'tokenizer_contnet': '的 时间 在 fidelio dkd 研究 中 中国 有 372 例 患者 接受 了 随机 分组 非 奈 利 酮 组 10mg 或 20mg 每日 一次 n 188 安慰剂 组 n 184 非 奈 利 酮 组 患者 的 平均 治疗 持续时间 为 2 年 中国 亚 组 的 结果 与 整体 研究 结果 基本一致 在 figaro dkd',
    'tokenizer_fine_grained': '的时间 在FIDELIO-DKD研究中，中国有372例患者接受了随机分组[非奈利酮组（10mg或20mg，每日一次）n=188，安慰剂组n=184]。非奈利酮组患者的平均治疗持续时间为2年。中国亚组的结果与整体研究结果基本一致。 在FIGARO-DKD�'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '研究结果基本一致。 在FIGARO-DKD研究中，中国有325例患者接受了随机分组[非奈利酮组（10mg或20mg，每日一次）n=162，安慰剂组n=163]。非奈利酮组患者的平均治疗持续时间为2.8年。中国亚组的结果与整体研究结果基本一�',
    'tokenizer_contnet': '研究 结果 基本一致 在 figaro dkd 研究 中 中国 有 325 例 患者 接受 了 随机 分组 非 奈 利 酮 组 10mg 或 20mg 每日 一次 n 162 安慰剂 组 n 163 非 奈 利 酮 组 患者 的 平均 治疗 持续时间 为 28 年 中国 亚 组 的 结果 与 整体 研究 结果 基本 一',
    'tokenizer_fine_grained': '研究 结果 基本一致 。 在FIGARO-DKD研究中，中国有325例患者接受了随机分组[非奈利酮组（10mg或20mg，每日一次）n=162，安慰剂组n=163]。非奈利酮组患者的平均治疗持续时间为2.8年。中国亚组的结果与整体研究结果基本一�'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '。中国亚组的结果与整体研究结果基本一致。 【药理毒理】 药理作用 Finerenone是一种非甾体选择性盐皮质激素受体（MR）拮抗剂，可被醛固酮和皮质醇激活并调节基因转录。MR过度激活会导致纤维化和炎症，Fineren',
    'tokenizer_contnet': '中国 亚 组 的 结果 与 整体 研究 结果 基本一致 药理 毒理 药理作用 finerenon 是 一种 非甾体 选择性 盐 皮质激素 受体 mr 拮抗剂 可 被 醛固酮 和 皮质醇 激活 并 调节 基因 转录 mr 过度 激活 会 导致 纤维化 和 炎症 fineren',
    'tokenizer_fine_grained': '。中国亚组的结果与整体研究结果基本一致。 【 药理 毒理 】 药理 作用 Finerenone是一种非甾体选择性盐皮质激素受体（MR）拮抗剂，可被醛固酮和皮质醇激活并调节基因转录。MR过度激活会导致纤维化和炎症，Fineren'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '活会导致纤维化和炎症，Finerenone在上皮（如肾脏）和非上皮（如心脏和血管）组织中阻断MR介导的钠重吸收和MR过度激活。Finerenone对MR具有高效价和选择性，对雄激素、孕激素、雌激素和糖皮质激素受体无亲',
    'tokenizer_contnet': '活会 导致 纤维化 和 炎症 finerenon 在 上皮 如 肾脏 和 非 上皮 如 心脏 和 血管 组织 中 阻断 mr 介导 的 钠 重吸收 和 mr 过度 激活 finerenon 对 mr 具有 高效 价 和 选择性 对 雄激素 孕激素 雌激素 和 糖皮质 激素 受体 无 亲',
    'tokenizer_fine_grained': '活会导致纤维化和炎症，Finerenone在上皮（如肾脏）和非上皮（如心脏和血管）组织中阻断MR介导的钠重吸收和MR过度激活。Finerenone对MR具有高效价和选择性，对雄激素、孕激素、雌激素和糖皮质激素受体无亲'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '�素和糖皮质激素受体无亲和力。 毒理研究 遗传毒性 Finerenone Ames试验、中国仓鼠V79细胞染色体畸变试验和小鼠体内骨髓微核试验结果均为阴性。 生殖毒性 Finerenone未见对雄性大鼠生育力的影响，在AUC相当于人体最',
    'tokenizer_contnet': '素 和 糖皮质 激素 受体 无 亲和力 毒理 研究 遗传 毒性 finerenon ame 试验 中国 仓鼠 v79 细胞 染色体 畸变 试验 和 小鼠 体内 骨髓 微 核试验 结果 均 为 阴性 生殖 毒性 finerenon 未 见 对 雄性 大鼠 生育力 的 影响 在 auc 相当于 人体 最',
    'tokenizer_fine_grained': '�素和糖皮质激素受体无亲和力。 毒 理 研究 遗传 毒 性 finerenon Ames试验、中国仓鼠V79细胞染色体畸变试验和小鼠体内骨髓微核试验结果均为阴性。 生殖 毒 性 Finerenone未见对雄性大鼠生育力的影响，在AUC相当于人体最'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '鼠生育力的影响，在AUC相当于人体最大暴露量的20倍时可见对雌性大鼠生育力的毒性。 在大鼠胚胎-胎仔毒性试验中，剂量为10mg/kg/天（相当于人体游离药物AUC的19倍）时，可见胎盘重量减轻和胎仔毒性，包括胎仔体',
    'tokenizer_contnet': '鼠 生育力 的 影响 在 auc 相当于 人体 最大 暴露 量 的 20 倍时 可见 对 雌性 大鼠 生育力 的 毒性 在 大鼠 胚胎 胎 仔 毒性 试验 中 剂量 为 10mg kg 天 相当于 人体 游离 药物 auc 的 19 倍 时 可见 胎盘 重量 减轻 和 胎 仔 毒性 包括 胎 仔 体',
    'tokenizer_fine_grained': '鼠生育力的影响，在AUC相当于人体最大暴露量的20倍时可见对雌性大鼠生育力的毒性。 在大鼠胚胎-胎仔毒性试验中，剂量为10mg/kg/天（相当于人体游离药物AUC的19倍）时，可见胎盘重量减轻和胎仔毒性，包括胎仔体'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '�轻和胎仔毒性，包括胎仔体重降低和骨化延迟；剂量为30mg/kg/天（相当于人体游离药物AUC的25倍）时，可见内脏和骨骼畸形的发生率增加（轻度水肿、脐带缩短、囟门轻度增大），一例胎仔可见包括罕见',
    'tokenizer_contnet': '轻 和 胎 仔 毒性 包括 胎 仔 体重 降低 和 骨化 延迟 剂量 为 30mg kg 天 相当于 人体 游离 药物 auc 的 25 倍 时 可见 内脏 和 骨骼 畸形 的 发生率 增加 轻度 水肿 脐带 缩短 囟门 轻度 增大 一例 胎 仔 可见 包括 罕见',
    'tokenizer_fine_grained': '�轻和胎仔毒性，包括胎仔体重降低和骨化延迟；剂量为30mg kg 天（相当于人体游离药物AUC的25倍）时，可见内脏和骨骼畸形的发生率增加（轻度水肿、脐带缩短、囟门轻度增大），一例胎仔可见包括罕见'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '轻度增大），一例胎仔可见包括罕见畸形（双主动脉弓）在内的复杂畸形。最大无毒性反应剂量（大鼠3mg/kg，兔2.5mg/kg）相当于人体游离药物AUC的10-13倍。 在大鼠围产期发育毒性试验中，大鼠妊娠和哺乳期间给',
    'tokenizer_contnet': '轻度 增大 一例 胎 仔 可见 包括 罕见 畸形 双 主动脉弓 在内 的 复杂 畸形 最大 无毒性 反应 剂量 大鼠 3mg kg 兔 2 5mg kg 相当于 人体 游离 药物 auc 的 10 13 倍 在 大鼠 围产期 发育 毒性 试验 中 大鼠 妊娠 和 哺乳 期间 给',
    'tokenizer_fine_grained': '轻度增大），一例胎仔可见包括罕见畸形（双主动脉弓）在内的复杂畸形。最大无毒性反应剂量（大鼠3mg/kg，兔2.5mg/kg）相当于人体游离药物AUC的10-13倍。 在大鼠围产期发育毒性试验中，大鼠妊娠和哺乳期间给'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '中，大鼠妊娠和哺乳期间给药，暴露量相当于人体游离药物AUC的4倍时，可见幼仔死亡率增加和其他毒性反应（幼仔体重降低、耳廓张开延迟），并可见子代活动略有增加，但未见其他神经行为改变。最大无毒性反应剂量（1mg',
    'tokenizer_contnet': '中 大鼠 妊娠 和 哺乳 期间 给 药 暴露 量 相当于 人体 游离 药物 auc 的 4 倍时 可见 幼 仔 死亡率 增加 和 其他 毒性 反应 幼 仔 体重 降低 耳廓 张开 延迟 并 可见 子代 活动 略有 增加 但 未 见 其他 神经 行为 改变 最大 无毒性 反应 剂量 1mg',
    'tokenizer_fine_grained': '中，大鼠妊娠和哺乳期间给药，暴露量相当于人体游离药物AUC的4倍时，可见幼仔死亡率增加和其他毒性反应（幼仔体重降低、耳廓张开延迟），并可见子代活动略有增加，但未见其他神经行为改变。最大无毒性反应剂量（1mg'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '经行为改变。最大无毒性反应剂量（1mg/kg）相当于人体游离药物AUC的2倍。 致癌性 在小鼠和大鼠2年致癌性研究中，未见肿瘤发生率明显增加。在雄性小鼠中，在相当于人体游离药物AUC的26倍剂量下可见睾丸间质细胞�',
    'tokenizer_contnet': '经 行为 改变 最大 无毒性 反应 剂量 1mg kg 相当于 人体 游离 药物 auc 的 2 倍 致癌性 在 小鼠 和 大鼠 2 年 致癌性 研究 中 未 见 肿瘤 发生率 明显增加 在 雄性 小鼠 中 在 相当于 人体 游离 药物 auc 的 26 倍 剂量 下 可见 睾丸 间质 细胞',
    'tokenizer_fine_grained': '经行为改变。最大无毒性反应剂量（1mg/kg）相当于人体游离药物AUC的2倍。 致癌 性 在小鼠和大鼠2年致癌性研究中，未见肿瘤发生率明显增加。在雄性小鼠中，在相当于人体游离药物AUC的26倍剂量下可见睾丸间质细胞�'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '26倍剂量下可见睾丸间质细胞瘤发生率增加，但不认为具有临床相关性。 【贮藏】 密闭，不超过30℃保存。请将药品置于儿童触及不到的地方。 【包装】 药用铝箔，聚氯乙烯/聚偏二氯乙烯固体药用复合硬片包装。',
    'tokenizer_contnet': '26 倍 剂量 下 可见 睾丸 间质 细胞 瘤 发生率 增加 但 不 认为 具有 临床 相关性 贮藏 密闭 不 超过 30 保存 请 将 药品 置于 儿童 触及 不到 的 地方 包装 药用 铝箔 聚氯乙烯 聚 偏 二 氯乙烯 固体 药用 复合 硬 片 包装',
    'tokenizer_fine_grained': '26倍剂量下可见睾丸间质细胞瘤发生率增加，但不认为具有临床相关性。 【贮藏】 密闭，不超过30℃保存。请将药品置于儿童触及不到的地方。 【包装】 药用铝箔，聚氯乙烯/聚偏二氯乙烯固体药用复合硬片包装。'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '�乙烯固体药用复合硬片包装。 10mg：7片/盒，14片/盒，28片/盒。 20mg：7片/盒，14片/盒。 【有效期】 36个月 【执行标准】  JX20220072 【批准文号】 10mg：国药准字HJ20220057 20mg：国药准字HJ20220058 【生产企业】 企业名称：B',
    'tokenizer_contnet': '乙烯 固体 药用 复合 硬 片 包装 10mg 7 片 盒 14 片 盒 28 片 盒 20mg 7 片 盒 14 片 盒 有效期 36 个 月 执行 标准 jx20220072 批准文号 10mg 国药准字 hj20220057 20mg 国药准字 hj20220058 生产 企业 企业名称 b',
    'tokenizer_fine_grained': '�乙烯固体药用复合硬片包装。 10mg：7片 盒，14片 盒，28片 盒。 20mg：7片 盒，14片 盒。 【有效期】 36个月 【执行标准】 JX20220072 【批准文号】 10mg：国药准字HJ20220057 20mg：国药准字HJ20220058 【生产企业】 企业名称：B'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '准字HJ20220058 【生产企业】 企业名称：Bayer AG 生产地址：Kaiser-Wilhelm-Allee，51368 Leverkusen，Germany [境内联系人] 名称：拜耳医药保健有限公司 注册地址：北京市北京经济技术开发区荣京东街7号 邮政编码：100176 电话和传真号码：010-59218282；010-59218181（传真）；400-810-0360',
    'tokenizer_contnet': '准 字 hj20220058 生产 企业 企业名称 bayer ag 生产 地址 kaiser wilhelm alle 51368 leverkusen germani 境内 联系人 名称 拜 耳 医药保健 有限公司 注册 地址 北京市 北京经济技术开发区 荣 京 东街 7 号 邮政编码 100176 电话 和 传真号码 010 59218282 010 59218181 传真 400 810 0360',
    'tokenizer_fine_grained': '准字HJ20220058 【 生产 企 业 】 企业 名称 ： B a y e r ag 生产地址：Kaiser-Wilhelm-Allee，51368 Leverkusen，Germany [ 境内 联系人 ] 名称：拜耳医药保健有限公司 注册地址：北京市北京经济技术开发区荣京东街7号 邮政编码：100176 电话和传真号码：010-59218282；010-59218181（传真）；400-810-0360'},
    {
        'filename': 'D:\\projects\\learn_langchain_rag\\langchain\\2.消息，模板，chatModel\\file\\drug description\\非奈利酮片[190125,190124].pdf',
    'content': '59218282；010-59218181（传真）；400-810-0360（热线）   ',
    'tokenizer_contnet': '59218282 010 59218181 传真 400 810 0360 热线',
    'tokenizer_fine_grained': '59218282；010-59218181（传真）；400-810-0360（热线）'}]
    import  time
    start = time.time()
    embedding.encode(test_texts_2,batch_size=4)
    end = time.time()
    print(end - start)

