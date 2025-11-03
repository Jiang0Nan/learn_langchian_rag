import re
from collections import OrderedDict
from typing import Optional, List
from FlagEmbedding import BGEM3FlagModel

import fitz
import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter,RecursiveJsonSplitter
import copy
import json
import os
from langchain_milvus import Milvus
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

from loguru import logger
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from ollama import embeddings
from paddle.base.libpaddle import INT64
from pymilvus import FieldSchema, DataType, MilvusClient, AnnSearchRequest, WeightedRanker, CollectionSchema
from pymilvus.milvus_client import IndexParams


# 由于本pdf内容全是图片，因此需要先把pdf拆成图片，再用 ocr等工具进行识别，重新整合

# pdf拆成图片
# todo 这里后面可以放在缓存里面，直接边解析，边存，记得要显示进度  由于miner u分割了，因此做备选
def pdf_to_image(input_path:str,out_path="./2.消息，模板，chatModel/file/image/")->list[str]:
    """检查输出目录是否存在，不存在则创建
    :param input_path : pdf 所在目录
    :param out_path : 输出目录
    """
    try:
        images = []
        if not os.path.exists(out_path):
            os.makedirs(out_path,exist_ok=True)
            logger.info(f"没有输出目录，创建完成{out_path}")


        file = fitz.open(input_path)
        file_name = os.path.split(input_path)[-1].replace(".pdf","")
        file_length_tqdm = tqdm.tqdm(range(len(file)),desc="pdf转为图片 进度条")
        for page_num in file_length_tqdm:
            page = file[page_num]
            pix = page.get_pixmap(dpi=300)
            image_path = os.path.join(out_path,f"{file_name}_{page_num+1}.png")
            pix.save(image_path)
            images.append(image_path)
            file_length_tqdm.set_description(f"{file_name}已完成{page_num+1}/{len(file)}")
        file.close()
        return images
    except Exception as e:
        logger.error(f"{e}")


def find_all_files(input_path:str)->list:
    """查找所有文件
    :param input_path 文件目录
    """
    all_files = []
    try:
        for root, dirs, files in os.walk(input_path):
            for name in files:
                all_files.append(os.path.join(input_path,name))
        return all_files
    except Exception as e:
        logger.error(f"{e}")

# todo 由于本项目中miner u处理速度较慢且部分不准确，但是官网很快且准确，猜测gp的原因，后续在其他机器上试试
def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=False,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable)

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url)

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)


def pre_process_single_V1(json_file_path:str)->List[dict]:
    """数据预处理
    :param json_file_path :需要处理的json文件路径
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    drug_name = os.path.split(json_file_path)[-1].replace("_content_list.json","")
    current_chapter = ''
    before_json_line = {}
    data_list = []
    for json_line in json_file:
        table_id = ''
        table_caption = ''
        is_chapter = False
        image_id = ''
        image_caption = ''
        embedding_text = ''

        if json_line.get("type") == "text" and json_line.get("text").strip() == "":
            continue
        if json_line.get("type") == "text":
            before_json_line = json_line
            match_title = re.match(r"【([^】]+)】", json_line.get("text").strip())
            embedding_text = json_line.get("text").strip()
            if match_title:
                current_chapter = match_title.group(0)
                text = json_line.get("text").replace(current_chapter, "")
                embedding_text = text
                is_chapter = True

        if not is_chapter:
            if json_line.get("type") == "table":
                embedding_text = json_line.get("table_body", "")
                if before_json_line.get("type") == "text":
                    table_id = re.match("^表\d*\s*", before_json_line.get("text").strip())
                    if table_id:
                        table_id = table_id.group(0)

                    table_caption = before_json_line.get("text").strip()
                elif before_json_line.get("type") == "image":
                    if len(before_json_line.get("image_caption", "")) > 1:
                        for image_caption in before_json_line.get("image_caption", ""):
                            table_id_tmp = re.match("^表\d*\s*", before_json_line.get("image_caption", "").strip())
                            if table_id_tmp:
                                table_id = table_id_tmp.group(0)
                                table_caption = image_caption

                    elif len(before_json_line.get("image_caption", "")) == 1:
                        table_id_tmp = re.match("^表\d*\s*", before_json_line.get("image_caption", "").strip())
                        if table_id_tmp:
                            table_id = table_id_tmp.group(0)
                            table_caption = before_json_line.get("image_caption").strip()
                    if not  table_id :
                        table_id = ''
                    #todo需要查看是否存在描述信息在上一个text中如果没匹配上 但中间间隔image结构如 text image image
                embedding_text += '\n' + table_caption + f"{json_line.get('table_caption', '')}\n{json_line.get('table_footnote', '')}\n图片路径：{json_line.get('img_path', '')}"
            # todo 使用模型描述文本在进行拼接
            if json_line.get("type") == "image":
                if len(json_line.get("image_caption")) > 0:
                    image_id_macht = re.match("^图\d*\s*", json_line.get("image_caption")[0])
                    if image_id_macht:
                        image_id = image_id_macht.group(0)
                    elif before_json_line.get("type") == "text":
                        image_id = re.match("^图\d*\s*", before_json_line.get("text").strip())
                        if image_id_macht:
                            image_id = image_id_macht.group(0)
                        image_caption = before_json_line.get("text").strip()
                elif before_json_line.get("type") == "text":
                    image_id = re.match("^图\d*\s*", before_json_line.get("text").strip())
                    if image_id:
                        image_id = image_id.group(0)
                    image_caption = before_json_line.get("text").strip()
                if not image_id :
                    image_id = ''
                embedding_text = f"{image_caption}+\n{image_caption}\n{json_line.get('table_footnote', '')}\n图片路径：{json_line.get('img_path', '')}"
                before_json_line = json_line
        # elif  is_chapter:
        #     current_chapter_data = []
        #     preprocessed_data[current_chapter] = current_chapter_data
        # if json_line.get("type") =="text"  and json_line.get("text").strip() != "" and is_chapter:
        # if current_chapter not in  preprocessed_data.keys() :
        #     preprocessed_data[current_chapter] = [json_line]
        # else :
        #     preprocessed_data.get(current_chapter).append(json_line)
        # embedding_text = js
        # 提取（后续用于过滤和引用）
        if embedding_text == '':
            continue
        metadata = {
            "drug_name": drug_name,  # 药品名
            "chapter_name": current_chapter,  # 章节名
            "page_idx": json_line.get("page_idx", -1),  # 页码
            "type": json_line.get("type", "未知"),  # 类型（text/table/image）
            "img_path": json_line.get("img_path", ""),  # 图片/表格图片路径（若有）

            "table_id": table_id,
            "table_caption": table_caption,
            "image_id": image_id,
            "image_caption": image_caption  # 图片描述
        }
        data_list.append({
            "embedding_text": embedding_text,
            "metadata": metadata
        })
    return  data_list

def generate_embeddings_v1(data_list:Optional[List[dict]], embeddings, batch_size=16):
    """
    # 批量生成向量（按批次处理，避免内存占用过高）
    :param data_list: 需要处理的list
    :param embeddings: 选择的embeddings
    :param batch_size: 批次大小
    :return:
    """
    vectors = []
    texts = [item["embedding_text"] for item in data_list]
    # 批量生成（适合数据量较大时）
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_vectors = embeddings.embed_documents(batch_texts)
        vectors.extend(batch_vectors)
    # 关联 向量+文本+元数据
    vector_data = []
    for i in range(len(data_list)):
        vector_data.append({
            "vector": vectors[i],
            "text": data_list[i]["embedding_text"],
            "metadata": data_list[i]["metadata"],
            "drug_name": data_list[i]["metadata"].get("drug_name",''),  # 药品名
        })
    return vector_data

def generate_embeddings_v2(data_list:Optional[List[dict]], embeddings, batch_size=16):
    """
    # 批量生成向量（按批次处理，避免内存占用过高）
    :param data_list: 需要处理的list
    :param embeddings: 选择的embeddings
    :param batch_size: 批次大小
    :return:
    """
    vectors = []
    texts = [item["embedding_text"] for item in data_list]
    # 批量生成（适合数据量较大时）
    # todo，我觉得这里有点不合适，需要调整
    for text in texts:
        # batch_texts = texts[i:i+batch_size]
        query_text_embedding = embeddings.encode(
            sentences=text,  # Union[List[str], str],
            return_dense=True,  # Optional[bool] = None,
            return_sparse=True,  # Optional[bool] = None,
            return_colbert_vecs=True,  # Optional[bool] = None,
        )

        query_text_embedding_lexical_weights= query_text_embedding.get("lexical_weights",[])
        sorted_weight_dict = sorted(query_text_embedding_lexical_weights.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = OrderedDict(sorted_weight_dict[:10])
        query_text_embedding["lexical_weights"]=sorted_dict
        vectors.append({
            "dense_vecs": query_text_embedding['dense_vecs'],
            "lexical_weights": query_text_embedding['lexical_weights'],
            # "colbert_vecs": query_text_embedding['colbert_vecs'],
        })
    # 关联
    vector_data = []
    try:
        for i in range(len(data_list)):
            vector_data.append({
                "dense_vector": vectors[i].get("dense_vecs",""),
                "dense_sparse": vectors[i].get("lexical_weights",""),
                # "dense_colbert_vecs": vectors[i].get("colbert_vecs",""),
                "text": data_list[i]["embedding_text"],
                "metadata": data_list[i]["metadata"],
                "drug_name": data_list[i]["metadata"].get("drug_name",''),  # 药品名
            })
    except Exception as e:
        print(e)
    return vector_data



def init_milvus()->MilvusClient:
    URI = "http://localhost:19530"
    vector_store = MilvusClient(
        # embedding_function = embeddings,
        connection_args={"uri": URI}
    )
    # 参数
    schema_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, description="嵌入后的密集向量", dim=1024),
        # FieldSchema(name="dense_colbert_vecs", dtype=DataType.FLOAT_VECTOR, description="嵌入后的密集向量", dim=1024),
        FieldSchema(name="dense_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR, description="嵌入后的"),
        # 稀疏向量不指定dim
        FieldSchema(name="text", dtype=DataType.VARCHAR, description="原文本", max_length=4096, enable_analyzer=True),
        FieldSchema(name="metadata", dtype=DataType.JSON, description="metadata"),
        FieldSchema(name="drug_name" , dtype=DataType.VARCHAR,description="药品名称" ,max_length  = 1024)
    ]


    if vector_store.has_collection(collection_name="milvus_test"):
        vector_store.drop_collection(collection_name="milvus_test")
    #     注意：部分参数会被schema的覆盖，即，schema的参数优先级更高
    vector_store.create_collection(
        collection_name="milvus_test",
        # dimension=1024,  # 定义向量字段的维度（即每个向量包含的元素数量）
        # primary_field_name= "id",  # 指定主键字段的名称（用于唯一标识每条数据，类似数据库的主键）
        # id_type="int",  # 定义主键字段的数据类型
        # vector_field_name= "vector",  # 指定存储向量数据的字段名称。主要是看 schema，若有schema则该会被忽略
        # metric_type="COSINE",#定义向量相似度的计算方式  #COSINE"：余弦相似度（适合文本、图像等向量，衡量方向相似性）；"L2"：欧氏距离（适合衡量空间中两点的直线距离）；"IP"：内积（适合推荐系统等场景）。
        # auto_id = True,#控制主键是否自动生成
        # timeout = None,#设置操作超时时间
        schema=CollectionSchema(schema_fields),  # 如果不是使用CollectionSchema创建的还需要包裹一层CollectionSchema（schema）
        # index_params = None,#建集合时是否自动为向量字段创建索引
        consistency_level="Strong",
        drop_old=True,
    )

    # 索引
    try:
        existing_index = vector_store.describe_index(collection_name="milvus_test")
    except:
        existing_index = None
    if not existing_index:
        sparse_index = IndexParams()
        sparse_index.add_index("dense_sparse", "SPARSE_INVERTED_INDEX", metric_type="IP")

        dense_index = IndexParams()
        dense_index.add_index("dense_vector", "HNSW", metric_type="COSINE", M=16, efConstruction="100")

        # colbert_index = IndexParams()
        # colbert_index.add_index("dense_colbert_vecs", "HNSW", metric_type="COSINE", M=16, efConstruction="100")

        # 调用创建索引方法
        vector_store.create_index(
            collection_name="milvus_test",
            index_params=sparse_index  # 传递 IndexParams 实例
        )
        vector_store.create_index(
            collection_name="milvus_test",
            index_params=dense_index  # 传递 IndexParams 实例
        )
        # vector_store.create_index(
        #     collection_name="milvus_test",
        #     index_params=colbert_index  # 传递 IndexParams 实例
        # )
    return vector_store


def milvus_insert(data,client: MilvusClient)->dict:
    #  return Dict: Number of rows that were inserted and the inserted primary key list.
    milvus_data = []

    for item in data:
        # dense_vector 一维向量直接插入
        milvus_data.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "drug_name": item["drug_name"],
            "dense_vector": item["dense_vector"].tolist(),
            "dense_sparse": dict(item["dense_sparse"])
        })


    res = client.insert(
        collection_name="milvus_test",
        data=milvus_data
    )
    return res

if __name__ == '__main__':
    # =========超参数=========
    base_url = r"D:\projects\learn_langchain_rag\parctise\file\01_医疗核心制度"
    image_out_path = r"D:\projects\learn_langchain_rag\parctise\file\image"

    pdf_files_dir = os.path.join(base_url)
    embeddings_bge = BGEM3FlagModel(
        model_name_or_path=r"D:\files\models\bge-m3",
        cache_folder=r"D:\files\models\bge-m3",
        use_fp16=True
    )
    vector_store = init_milvus()
    # ======== 1. 提取 pdf
    output_dir = os.path.join(base_url, "output")
    # pdf_suffixes = [".pdf"]
    # image_suffixes = [".png", ".jpeg", ".jpg"]
    #
    # doc_path_list = []
    # for doc_path in Path(pdf_files_dir).glob('*'):
    #     if doc_path.suffix in pdf_suffixes + image_suffixes:
    #         doc_path_list.append(doc_path)
    #
    # """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    # # 使用miner u解析pdf文件，观察结构发现，一个是带有元素据（页码）的json,一个是最终的md文件
    # parse_doc(doc_path_list, output_dir, backend="pipeline")

    # =======预处理文件 , 没有页码，等元素据 todo，可以用json来进行分块，保留元数据
    # 处理方法1，
    # text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    # for root, dirs,files in os.walk(base_url):
    #     for file in files:
    #        if  os.path.splitext(file)[-1] == ".md":
    #            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
    #               file_md = []
    #               for line in f:
    #                   f_line = line.replace("#","") #识别的结果级别不准确
    #                   file_md.append(f_line)
    #               single_md_split = text_spliter.split_text(''.join(file_md))

    # 处理方法2使用json
    json_paths = []
    for root, dirs,files in os.walk(output_dir):
        for file in files:
            if os.path.splitext(file)[-1] == ".md":
                json_path = os.path.join(root, os.path.splitext(file)[0]+"_content_list.json")
                json_paths.append(json_path)

    # 处理生成的json文件
    for json_path in json_paths:
        single_data_list = pre_process_single_V1(json_path)
        print(single_data_list)
        # json_spliter = RecursiveJsonSplitter(max_chunk_size=1000)
        # # json_spliter_result = json_spliter.split_json(single_data_list,
        # #                                               # convert_lists=True #会拆成子串
        #                                               )
        # 向量化
        vector_data = generate_embeddings_v2(single_data_list, embeddings_bge)
        # 存入数据库
        milvus_insert(vector_data,vector_store )


