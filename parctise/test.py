# import re
# import bisect
# from typing import List, Tuple, Dict
# from langchain_text_splitters import TokenTextSplitter
#
# def merge_pages_with_markers(pages: List[Tuple[str, int]], page_marker_fmt="[[PAGE_{:06d}]]\n") -> Tuple[str, List[int]]:
#     """
#     pages: list of tuples (page_text, page_index)
#     返回:
#       joined_text: 带页标记的合并文本
#       page_starts: 每页（带marker）的起始字符索引列表，长度等于 len(pages)
#     """
#     parts = []
#     page_starts = []
#     cur = 0
#     for text, page_idx in pages:
#         marker = page_marker_fmt.format(page_idx)
#         entry = marker + (text if text is not None else "")
#         page_starts.append(cur)
#         parts.append(entry)
#         cur += len(entry)
#     joined_text = "\n".join(parts)
#     return joined_text, page_starts
#
# def charpos_to_page(char_pos: int, page_starts: List[int]) -> int:
#     """
#     给定字符位置 char_pos，和 page_starts 列表（每页起始字符索引），
#     返回该位置属于哪个页（返回 page index 的列表下标），
#     若 char_pos 恰好等于某页起始，则属于该页。
#     """
#     # bisect_right-1 可以得到 index
#     idx = bisect.bisect_right(page_starts, char_pos) - 1
#     if idx < 0:
#         return 0
#     return idx
#
# def split_joined_text_with_pages(
#     pages: List[Tuple[str, int]],
#     chunk_size: int = 512,
#     chunk_overlap_size: int = 100,
#     encoding_name: str = "cl100k_base",
# ) -> List[Dict]:
#     """
#     pages: [(page_text, page_index), ...]  —— page_index 建议是页码（从0或1开始）
#     返回: list of dict, 每项包含:
#         {
#           "text": chunk_text (cleaned, 去掉 marker),
#           "page_start": idx_of_first_page,    # 对应 pages 中的 page_index（不是序号）
#           "page_end": idx_of_last_page,
#           "page_indexes": [list of page_index ...], # 可选
#           "char_start": pos_in_joined_text,
#           "char_end": pos_in_joined_text
#         }
#     """
#     # 1. 合并并记录每页起始位置（包含 marker）
#     joined_text, page_starts = merge_pages_with_markers(pages, page_marker_fmt="[[PAGE_{:06d}]]\n")
#
#     # 2. 用 TokenTextSplitter 分块（基于 tokenizer）
#     splitter = TokenTextSplitter.from_tiktoken_encoder(
#         encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap_size
#     )
#     chunks = splitter.split_text(joined_text)  # 这里返回的是字符串 list
#
#
#     results = []
#     search_start = 0  # 在 joined_text 中的查找起点，递增以避免匹配到前面的相同子串
#     for chunk in chunks:
#         # 为避免 chunk 中包含 marker，查找时保留 marker；后面再清理
#         found = joined_text.find(chunk, search_start)
#         if found == -1:
#             # 极少数情况：分割器可能移除了一些空白导致找不到完全相同子串
#             # 备选策略：更宽容的查找（去掉重复空白再找），或者从 0 开始找到第一个后移
#             # 这里采用从 0 开始找第一个未被占用的匹配位置
#             found = joined_text.find(chunk)
#             if found == -1:
#                 # 仍找不到，作为最后手段，跳过该 chunk 或记录为无法定位
#                 char_start = None
#                 char_end = None
#                 page_start_idx = pages[0][1] if pages else None
#                 page_end_idx = pages[-1][1] if pages else None
#                 clean_text = re.sub(r"\[\[PAGE_\d+\]\]\n", "", chunk)
#                 results.append({
#                     "text": clean_text,
#                     "page_start": page_start_idx,
#                     "page_end": page_end_idx,
#                     "page_indexes": list(range(page_start_idx, page_end_idx + 1)) if page_start_idx is not None else [],
#                     "char_start": char_start,
#                     "char_end": char_end,
#                     "note": "not found exact location"
#                 })
#                 continue
#
#         char_start = found
#         char_end = found + len(chunk)
#
#         # 更新下次查找起点：放到当前匹配结束处，防止重复匹配
#         search_start = char_end
#
#         # 3. 将 char_start,char_end 映射到页序号
#         start_page_seq = charpos_to_page(char_start, page_starts)
#         end_page_seq = charpos_to_page(max(char_end - 1, char_start), page_starts)
#
#         # 转换为页面真实 page_index（pages 参数的第二项）
#         page_start_index = pages[start_page_seq][1]
#         page_end_index = pages[end_page_seq][1]
#
#         # 清理 chunk 中可能出现的 marker
#         clean_text = re.sub(r"\[\[PAGE_\d+\]\]\n", "", chunk)
#
#         # 还可以把页码区间扩展为具体页号列表（如需）
#         if page_start_index == page_end_index:
#             page_indexes = [page_start_index]
#         else:
#             # pages 中不一定连续的 page_index，不过常见情况是连续
#             # 我们直接返回从 start 到 end 的范围（包含）
#             page_indexes = list(range(page_start_index, page_end_index + 1))
#
#         results.append({
#             "text": clean_text,
#             "page_start": page_start_index,
#             "page_end": page_end_index,
#             "page_indexes": page_indexes,
#             "char_start": char_start,
#             "char_end": char_end,
#         })
#
#     return results
#
# # -------------------------
# # 使用示例
# # -------------------------
# if __name__ == "__main__":
#     # pages: list of (text, page_number). page_number 可自定从0或1开始
#     pages = [
#         ("第一页的文本内容。这里演示。", 1),
#         ("第二页的文本内容。很短。", 2),
#         ("第三页，内容有点长，需要跨多个 chunk 才能切完。这里继续填充一些文字以演示跨页情形。" * 5, 3),
#         ("第四页的尾巴。", 4),
#     ]
#
#     chunks_with_pages = split_joined_text_with_pages(pages, chunk_size=80, chunk_overlap_size=20)
#
#     for c in chunks_with_pages:
#         print("----- CHUNK -----")
#         print("pages:", c["page_start"], "-", c["page_end"], "->", c["page_indexes"])
#         print("chars:", c["char_start"], c["char_end"])
#         print(c["text"][:120].replace("\n", " "))
#         print()

# 重拍模型
from FlagEmbedding import FlagReranker
import torch

reranker = FlagReranker(r"D:\files\models\bge-reranker-large", use_fp16=torch.cuda.is_available())

pairs = [
    ("如何缓解头痛？", "服用止痛药如布洛芬。"),
    ("如何缓解头痛？", "今天天气很好。")
]
scores = reranker.compute_score(pairs)

print(scores)  # 输出: [0.92, 0.05]


# 创建文档
from langchain.schema import Document

# 创建一个文档对象
doc = Document(
    page_content="这是一个示例文档内容，用于展示如何在LangChain中创建Document对象。",
    metadata={
        "source": "example_source",  # 文档的来源
        "doc_id": "123456",          # 文档的ID
        "author": "John Doe",        # 文档的作者
        "category": "tutorial"       # 文档的分类
    }
)

# 查看文档的内容和元数据
print(doc.page_content)  # 打印文档的内容
print(doc.metadata)      # 打印文档的元数据

# 嵌入模型
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
model_name = r"D:\files\models\bge-m3"
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity


model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    # query_instruction="为这个句子生成表示以用于检索相关文章：",
    show_progress=True,
)

model.embed_documents(doc.p)




#  最后回答的装饰
def decorate_answer(answer):
    """
    装饰最终答案 - 添加引用信息和格式化

    Args:
        answer: LLM生成的原始答案

    Returns:
        dict: 包含格式化答案和引用信息的字典
    """
    nonlocal knowledges, kbinfos, prompt

    # 插入引用标记：在答案中标记引用来源
    answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]],
                                             [ck["vector"] for ck in kbinfos["chunks"]], embd_mdl, tkweight=0.7,
                                             vtweight=0.3)

    # 获取被引用的文档ID
    idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
    recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
    if not recall_docs:
        recall_docs = kbinfos["doc_aggs"]
    kbinfos["doc_aggs"] = recall_docs

    # 准备引用信息
    refs = deepcopy(kbinfos)
    for c in refs["chunks"]:
        if c.get("vector"):
            del c["vector"]  # 移除向量数据减少响应大小

    # 处理API密钥错误提示
    if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
        answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"

    # 格式化引用块
    refs["chunks"] = chunks_format(refs)
    return {"answer": answer, "reference": refs}
