import io

import fitz
from PIL import Image, ImageDraw

file = fitz.open(r"D:\files\学习笔记\学习资料\Advanced RAG Techniques Whitepaper WillowTree.pdf")

print(f"文档总页数: {len(file)}\n")
for page_index in range(len(file)):
    first_page = file[page_index]
    # “rawdict” 和 “blocks”
    dpi = 150
    zoom = dpi/72

    pix = first_page.get_pixmap(dpi=dpi)
    pix_image = Image.open(io.BytesIO(pix.tobytes()))
    # 展示图片
    draw = ImageDraw.Draw(pix_image)
    for block in first_page.get_text("rawdict")["blocks"]:
        if block["type"] == 0:
            region = fitz.Rect(block['bbox'])
            x1, y1, x2, y2 = region
            region=(x1*zoom,y1*zoom,x2*zoom,y2*zoom)
            draw.rectangle(region, outline="red", width=2)
        if block["type"] == 1:
            region = fitz.Rect(block['bbox'])
            x1, y1, x2, y2 = region
            region = (x1 * zoom, y1 * zoom, x2 * zoom, y2 * zoom)
            draw.rectangle(region, outline="blue", width=6)
    pix_image.show()

file.close()

