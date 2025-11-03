import pikepdf
# 适合文本/图形水印）
input_pdf = r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf"
output_pdf = "去水印.pdf"

with pikepdf.open(input_pdf) as pdf:
    for page in pdf.pages:
        # 遍历页面对象，删除标记为水印的内容
        if "/Annots" in page:
            annots = page["/Annots"]
            new_annots = [a for a in annots if "/Watermark" not in str(a)]
            page["/Annots"] = new_annots
    pdf.save(output_pdf)
