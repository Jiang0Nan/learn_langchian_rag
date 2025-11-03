import numpy as np
from PIL import Image
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pymupdf4llm
import os
from paddleocr import PaddleOCR

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="learn_langchain_rag"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
base_url="../2.消息，模板，chatModel/file/drug description/"


# def find_all_files(path):
#     """查找当前目录下符合要求的文件"""
#     all_files = []
#     for dirpath,dirnames,filenames in os.walk(path):
#         for filename in filenames:
#             all_files.append(os.path.join(dirpath,filename))
#     return all_files
#
# #================PyPDFDirectoryLoader  只能加载文本
# loader = PyPDFDirectoryLoader(
#     path=base_url
#
# )

# import datetime
# start = datetime.datetime.now()
# out = loader.lazy_load()
# end = datetime.datetime.now()
# print(end - start)
# print(out)
#
# start = datetime.datetime.now()
# out = loader.alazy_load()
# end = datetime.datetime.now()
# print(end - start)
# print(out)
#
# # start = datetime.datetime.now()
# # out = await  loader.aload()
# # end =datetime.datetime.now()
# # print(end - start)
# # print(out)
#
# start = datetime.datetime.now()
# out = loader.load_and_split()#弃用
# end = datetime.datetime.now()
# print(end - start)
# print(out)
# for i in out:
#     print(i)
#
#
# start = datetime.datetime.now()
# out = loader.load()
# end = datetime.datetime.now()
# print(end - start)
# print(out)

# 不符合需求，本案例中pdf内部是由图片粘贴而成
# all_files = find_all_files(base_url)
# for file in all_files:
#     with open(file, 'rb') as f:
#
#         paf_content = pymupdf4llm.to_markdown(doc=f,
#                                 write_images=True,  #是否将 PDF 中的图片 / 矢量图形保存为独立文件
#                                 embed_images=True,  #是否在输出的 markdown 文本中嵌入图片（通过 base64 编码，将图片数据直接写入文本，无需外部文件）。
#                                 image_path = "../2.消息，模板，chatModel/file/image/",  #图片保存的文件夹路径
#                                 image_format="png",  #提取图片的格式,影响图片质量和文件大小（如 PNG 支持透明，JPG 压缩率更高）。
#                                 dpi=96,  # 生成图片的分辨率
#                                 graphics_limit=4,  # 矢量图形的数量限制：若页面中矢量图形（如线条、形状）数量超过此值，则忽略所有矢量图形。
#                                 force_text=True,  #True：尝试提取所有文本，即使背景是图片；False：可能跳过图片上的文本。
#                                 page_chunks=True,
#                                 page_separators=True,  #是否在输出中添加 “分页符”（如--- Page 1 ---）区分不同页面。
#                                 # margins="",#页边距设置：忽略与页边距区域重叠的内容（如页眉、页脚、页边空白处的文本）。
#                                 show_progress=True,#是否在处理过程中打印进度（如 “Processing page 1/10”）。
#                                 page_width=360.0,#当 PDF 页面布局不固定（如不同页面宽度不同）时，强制使用此宽度作为统一假设。
#                                 table_strategy="",#表格检测策略：指定工具如何识别 PDF 中的表格（如基于线条、单元格对齐、内容布局等）
#                                 )
#
#         print(paf_content[:500])
#
#
# #============== panddleocr 识别准确率有问题且慢
# # 调用ocr进行识别
# ocr =  PaddleOCR(
#     # 文档方向分类相关参数：
#     # doc_orientation_classify_model_name = ,#文档方向分类模型的名称。
#     # doc_orientation_classify_model_dir=,#文档方向分类模型的目录路径。
#     # use_doc_orientation_classify=,#是否启用文档方向分类。
#
#     # 文档去畸变相关参数
#     # doc_unwarping_model_name= ,#文档去畸变模型的名称。
#     # doc_unwarping_model_dir=,#文档去畸变模型的目录路径。
#     # use_doc_unwarping=,#是否启用文档去畸变。
#
#     # 文本检测相关参数：
#     text_detection_model_name="ch_PP-OCRv4_det",#文本检测模型名称。
#     text_detection_model_dir=r"D:\files\models\ch_PP-OCRv4_det_infer\ch_PP-OCRv4_det_infer",#文本检测模型的目录路径。
#     # text_det_limit_side_len=1,#文本检测限制长边的最大值，防止文本过长，影响检测效果。
#     # text_det_limit_type=,#文本检测限制的类型（如长边、宽边限制）。
#     text_det_thresh=0.6,#文本检测的阈值，用于确定文本是否被检测为有效区域。
#     text_det_box_thresh=0.6,#文本框检测的阈值。
#     # text_det_unclip_ratio=,#文本框未剪裁区域的比率，用于对文本框进行调整。
#
#     # #文本行方向分类相关参数：
#     # textline_orientation_model_name=,#文本行方向分类模型的名称。
#     # textline_orientation_model_dir=,#文本行方向分类模型的目录路径。
#     use_textline_orientation=True,#是否启用文本行方向分类。
#     # 文本识别相关参数：
#     text_recognition_model_name="ch_PP-OCRv3_rec_slim",#文本识别模型的名称。
#     text_recognition_model_dir=r"D:\files\models\ch_PP-OCRv3_rec_slim_infer\ch_PP-OCRv3_rec_slim_infer",#文本识别模型的目录路径。
#     text_recognition_batch_size=4,#文本识别时的批量大小。
#     # text_rec_score_thresh=,#文本识别的分数阈值，用于过滤低置信度的文本。
#     # text_rec_input_shape=,#文本识别模型的输入形状。
#     lang='ch',#使用的语言（例如，'en'，'ch'，'fr'等）。
#     # ocr_version=,#OCR版本，可能的值包括 PP-OCRv3，PP-OCRv4，PP-OCRv5 等，具体版本可以根据需求进行选择。
#     return_word_box=True,#是否返回每个文本框的坐标信息，通常在需要标注或进一步分析时使用。
#
# )
# image_path = r"D:\projects\learn_langchain_rag\langchain\2.消息，模板，chatModel\file\image\枸橼酸氯米芬胶囊_1.png"
# with Image.open(image_path) as image:
#     # image.show()
#     image = np.array(image.convert('RGB'))
#     ocr_result = ocr.predict(image)
#     for i in ocr_result:
#         print(i)
#         # i.save_to_img(os.path.join(image_out_path, "ocr_result"))


# ===========PyPDFLoade没有携带页码
from pprint import pprint

from langchain_community.document_loaders import PyPDFLoader

file_path = r"D:\projects\learn_langchain_rag\langchain\2.消息，模板，chatModel\file\drug description\非奈利酮片[190125,190124].pdf"
def loader_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages

a = loader_pdf(file_path)
pprint(a)