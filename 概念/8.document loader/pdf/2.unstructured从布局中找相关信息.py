import os

import fitz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from langchain_unstructured import UnstructuredLoader
from multipart import file_path


def plot_pdf_with_boxes(pdf_page, segments):
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height)
            for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(
            scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
        )
        ax.add_patch(rect)

    # Make legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(
                patches.Patch(color=category_to_color[category], label=category)
            )
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.savefig("../output/unstructured布局信息.png")


def render_page(doc_list: list, page_number: int, print_text=True) -> None:
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc.metadata.get("page_number") == page_number
    ]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")

if __name__ == '__main__':
    file_path = r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\去水印_智能筛选.pdf" # 文件的路径,可以是路径的表
    os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'
    loader = UnstructuredLoader(
        file_path=file_path, #文件的路径,可以是路径的表
        strategy ='hi_res'  ,#'hi_res', 'fast', 'ocr_only'（控制图像+PDF处理方式）
        # file=,# 打开的文件对象或者文件流
        # coordinates=True,#会在提取每个 Element（例如段落、标题、表格等）时，附加该文本块在原始文档中的位置信息
        # partition_via_api=True,# 是否使用 Unstructured.io API（而非本地库）进行文档解析 如果问True则需要本地配置等,参考他的说明
        # post_processors=,#对每个提取文本块应用的后处理函数列表，如去除多余空格等例如[lambda x: x.strip()]
    )
    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    render_page(docs, 12, False)
    #
    from IPython.display import HTML, display

    # 表格文本在文档内容中折叠为单个字符串，但元数据包含其行和列的表示形式
    segments = [
        doc.metadata
        for doc in docs
        if doc.metadata.get("page_number") == 9 and doc.metadata.get("category") == "Table"
    ]

    # display(HTML(segments[0]["text_as_html"]))#langchian的包中没有这个属性
    # 结构可能具有父子关系 -- 例如，段落可能属于带有标题的部分。如果某个部分特别感兴趣（例如，用于索引），我们可以隔离相应的 Document 对象。
    # 本文本中没有这个案例
    conclusion_docs = []
    parent_id = -1
    for doc in docs:
        if doc.metadata["category"] == "Title" and "Conclusion" in doc.page_content:
            parent_id = doc.metadata["element_id"]
        if doc.metadata.get("parent_id") == parent_id:
            conclusion_docs.append(doc)

    for doc in conclusion_docs:
        print(doc.page_content)
