import os

from langchain_unstructured import UnstructuredLoader
os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'
loader = UnstructuredLoader(
    file_path=r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf", #文件的路径,可以是路径的表
    strategy ='hi_res'  ,#'hi_res', 'fast', 'ocr_only'（控制图像+PDF处理方式）
    # file=,# 打开的文件对象或者文件流
    # coordinates=True,#会在提取每个 Element（例如段落、标题、表格等）时，附加该文本块在原始文档中的位置信息
    # partition_via_api=True,# 是否使用 Unstructured.io API（而非本地库）进行文档解析 如果问True则需要本地配置等,参考他的说明
    # post_processors=,#对每个提取文本块应用的后处理函数列表，如去除多余空格等例如[lambda x: x.strip()]
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)
print(docs)
first_page_docs = [doc for doc in docs if doc.metadata.get("page_number") == 1]

for doc in first_page_docs:
    print(doc.page_content)




# from unstructured.partition.pdf import partition_pdf
#
# elements = partition_pdf(
#     filename=r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf", #文件的路径,可以是路径的表
#     extract_images_in_pdf=True,
#     infer_table_structure=True
# )
#
# for element in elements:
#     if element.category == "Image":
#         print(element.metadata.image_path)  # 有可能是 base64 或临时路径
#

