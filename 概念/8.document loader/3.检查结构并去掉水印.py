import pikepdf

input_path =r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf"
pdf = pikepdf.open(input_path)
for i, page in enumerate(pdf.pages):
    print(f"\n=== 第 {i+1} 页对象结构 ===")
    print(page.keys())  # 页面的顶层键
    if "/Resources" in page:
        resources = page["/Resources"]
        if "/XObject" in resources:
            print("XObject 列表：", list(resources["/XObject"].keys()))

# 方法3
import fitz  # pip install pymupdf

# =====================3
doc = fitz.open(input_path)

# 用于记录图片使用频率（判断是否是重复水印）
image_freq = {}

# 第一次遍历：统计图片引用频率。没用，因为每页是不同的 xref 对象
# for page in doc:
#     for img in page.get_images(full=True):
#         xref = img[0]
#         image_freq[xref] = image_freq.get(xref, 0) + 1

# 第二次遍历：按规则删除可疑水印
for page_index, page in enumerate(doc):
    img_list = page.get_images(full=True)
    for img in img_list:
        xref = img[0]
        width, height = img[2], img[3]
        print(f"第 {page_index + 1} 页 图片 xref={xref}, size=({width}x{height}), freq={image_freq.get(xref)}")

        # 👇 删除条件：尺寸较小 & 出现多次（即水印）
        if width == 200 and height == 200 :#and image_freq[xref] > 2:
            print(f"第 {page_index+1} 页 删除可能的水印图像 xref={xref}, size=({width}, {height})")
            doc._deleteObject(xref)

doc.save("去水印_智能筛选.pdf")
