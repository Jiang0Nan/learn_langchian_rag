from pdf2image import convert_from_path
from PIL import Image

# images = convert_from_path(r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf", dpi=200)
#
# for i, img in enumerate(images):
#     # 假设水印在右下角，可以裁剪或覆盖
#     width, height = img.size
#     img_crop = img.crop((0, 0, width-100, height-100))  # 简单裁剪
#     img_crop.save(f"./output/page_{i+1}.png")

import os
from pathlib import Path
import pikepdf
from pdf2image import convert_from_path
from PIL import Image
import img2pdf

def remove_watermark_pikepdf(input_path, output_path):
    """
    尝试使用 pikepdf 删除 PDF 内嵌的文本/图层水印
    """
    with pikepdf.open(input_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if "/Annots" in page:
                page["/Annots"] = pikepdf.Array([])  # 清空注释（很多水印以注释方式加的）
            if "/XObject" in page.Resources:
                xobjs = page.Resources["/XObject"]
                to_delete = []
                for name in xobjs:
                    obj = xobjs[name]
                    if "/Subtype" in obj and obj["/Subtype"] == "/Form":
                        if "/Watermark" in str(obj) or "/Artifact" in str(obj):
                            to_delete.append(name)
                for name in to_delete:
                    del xobjs[name]
        pdf.save(output_path)
    print(f"[✓] PikePDF 模式：去水印成功 → {output_path}")

def remove_watermark_image_mode(input_path, output_path, crop_bottom=80):
    """
    将 PDF 转为图片，裁剪底部（假设水印在底部），然后重新合成 PDF
    """
    images = convert_from_path(input_path, dpi=200)
    temp_imgs = []
    for i, img in enumerate(images):
        width, height = img.size
        cropped_img = img.crop((0, 0, width, height - crop_bottom))  # 裁剪底部区域
        temp_img_path = f"_temp_page_{i}.jpg"
        cropped_img.save(temp_img_path, "JPEG")
        temp_imgs.append(temp_img_path)

    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(temp_imgs))

    # 清理临时图片
    for f_path in temp_imgs:
        os.remove(f_path)

    print(f"[✓] 图片裁剪模式：去水印成功 → {output_path}")

def main():
    input_pdf = r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file\非奈利酮片[190125,190124].pdf"
    out_pdf_1 = "./output/去水印_pikepdf.pdf"
    out_pdf_2 = "./output/去水印_image.pdf"

    # 模式 1：尝试通过 PDF 对象删除水印
    try:
        remove_watermark_pikepdf(input_pdf, out_pdf_1)
    except Exception as e:
        print(f"[×] PikePDF 模式失败：{e}")

    # 模式 2：转换成图片裁剪底部水印再合成 PDF
    try:
        remove_watermark_image_mode(input_pdf, out_pdf_2, crop_bottom=80)
    except Exception as e:
        print(f"[×] 图片模式失败：{e}")

if __name__ == "__main__":
    main()
