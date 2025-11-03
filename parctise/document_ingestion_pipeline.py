import copy
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import fitz
import xxhash
from future.backports.datetime import datetime

from parctise.db.milvus_server import MilvusServer
from parctise.parser.pdf_only_text_parser import PDFOnlyTextLoader
from parctise.utils import rag_tokenizer
from parctise.utils.embedding import Embedding
from parctise.utils.rag_tokenizer import tokenizer
from parctise.utils.tools import charpos_to_page

def process_chunks(chunk, base_doc):
    doc = copy.deepcopy(base_doc)
    doc['content'] = chunk
    chunk = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", chunk)
    doc['tokenizer_content'] = rag_tokenizer.tokenize(chunk)
    doc['tokenizer_fine_grained'] = rag_tokenizer.fine_grained_tokenize(chunk)
    doc['id'] = xxhash.xxh64(
        (chunk + base_doc.get('doc_id', 'filename')).encode('utf-8', 'surrogatepass')).hexdigest()
    doc['create_time'] = str(datetime.now())[0:19]
    doc['create_timestamp'] = int(datetime.now().timestamp() * 1000)
    doc['doc_id'] = hashlib.md5(base_doc.get('filename', '').encode('utf-8')).hexdigest()
    doc['page_ids'] = base_doc.get('page_ids', [])
    return doc


def get_chunk(filename: Optional[Union[str, bytes]], start_pages: int = 0, end_pages=1000, type: str = 'pdf_only_text',
              lang: str = "Chinese", **kwargs):
    doc = {
        'filename': filename,
    }

    if type == 'pdf_only_text':
        parser = PDFOnlyTextLoader()
        load_texts, metadata = parser(filename, start_pages, end_pages, **kwargs)  # 读取纯文本中的文字

        # 分块并与页码合并
        page_marker_fmt = "[[PAGE_{:06d}]]\n"
        page_starts = []  # 标记起始字符位置
        cur_char = 0  # 记录当前所在的字符位置
        texts = []  # 存纯文本

        # 拼接纯文本
        for text, page_idx in load_texts:
            marker = page_marker_fmt.format(page_idx)
            entry = marker + "".join(text if text is not None else "")
            page_starts.append(cur_char)
            cur_char += len(entry)
            texts.append(entry)

        joined_text = ''.join(texts)

        joined_text = tokenizer._strQ2B(joined_text)  # 转为text
        chunks = parser.split_text_by_token(joined_text)  # 基于token进行切片

        start_search = 0  # 从哪里开始查找字串，避免重复
        res = []
        for chunk in chunks:
            page_ids = []

            # 去掉 “�” 及其他常见替代字符
            chunk = chunk.replace("�", "")
            # 去掉不可见控制字符（如零宽空格、BOM等）
            chunk = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\uFEFF]", "", chunk)

            # 找出chunk再原来文本中的位置并合并页码
            find_chunk_pos = joined_text.find(chunk, start_search)
            if find_chunk_pos == -1:
                find_chunk_pos = joined_text.find(chunk)
                print(f"第一次没找到，正在进行第二次查找")
                if find_chunk_pos == -1:
                    print(f"第二次没有找到字串{chunk}")

            char_start = find_chunk_pos  # 第一次出现的位置

            char_end = find_chunk_pos + len(chunk)  # 结束的位置

            start_search = char_end - 200  # 下一次的起点

            start_page_id = charpos_to_page(char_start, page_starts)
            end_page_id = charpos_to_page(max(char_end - 1, char_end), page_starts)  # 取该块最后一个字符位置

            page_start_id = load_texts[start_page_id][1]
            page_end_id = load_texts[end_page_id][1]

            # 清理 chunk 中可能出现的 marker
            clean_text = re.sub(r"\[\[PAGE_\d+\]\]\n", "", chunk)

            if page_start_id == page_end_id:
                page_ids.append(start_page_id)
            else:
                page_ids.append([i for i in range(page_start_id, page_end_id + 1)])
            doc['page_ids'] = json.dumps(page_ids[0])
            processed_chunk = process_chunks(clean_text, doc)

            res.append(processed_chunk)  # 分词
        return res


def process_pipeline(file_path,embedding,db):
    if ".pdf" in file_path:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        bath_process_page = 64
        # db = MilvusServer(collection_name='new_milvus_collection_test')

        # 使用线程将 Milvus 插入与下一轮编码重叠
        executor = ThreadPoolExecutor(max_workers=2)
        prev_future = None

        for i in range(0, page_count, bath_process_page):

            if i + bath_process_page > page_count:
                end_pages = page_count
            else:
                end_pages = i + bath_process_page
            process_start_time = time.time()
            chunks = get_chunk(file_path, start_pages=i, end_pages=end_pages)
            embedding.encode(chunks)

            # 保持仅一个在途插入任务，避免内存堆积
            if prev_future is not None:
                prev_future.result()
            prev_future = executor.submit(db.insert, chunks, None, 1024)

            print(f"处理{i}-{end_pages}花费的时间{time.time() - process_start_time}")

        # 等待最后一次插入完成
        if prev_future is not None:
            prev_future.result()

        # 批量写入完成后再刷新并创建索引
        try:
            db.flush()
            db.create_indexes(dense_M=32, dense_ef_construction=200)
        except Exception as e:
            print(f"索引创建或刷新失败: {e}")

def _init_env():
    try:
        embedding = Embedding()
        db = MilvusServer(collection_name='new_milvus_collection_test')
        dense_dim = embedding.dim['dense']
        db.init_collection(dense_dim=dense_dim)
        return embedding, db
    except Exception as e:
        print(f"初始化失败{e}")

def main():
    embedding, db = _init_env()
    file_path = r"./file"
    file_path_list = []
    for root,dirs,files in os.walk(file_path):
        for name in files:
            file_path_list.append(os.path.abspath(os.path.join(root,name)))
    for file_path in file_path_list:
        process_pipeline(file_path,embedding,db)
if __name__ == '__main__':
    main()
