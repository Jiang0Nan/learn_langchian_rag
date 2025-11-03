import copy
import io
import os
import re
from io import BytesIO
from typing import Optional, Union

from PIL import Image
from pypdf import PdfReader  as pdf_reader
from pypdf.errors import PdfReadError

import fitz
def process_outlines(outlines, depth) -> list:
    """
    用于处理目录
    """
    processed_outlines = []
    for line in outlines:
        if isinstance(line, dict):
            processed_outlines.append((line['/Title'], depth))
        process_outlines(line, depth + 1)
    return processed_outlines

def extract_images(pdf_path, save_folder):
    """提取文档中中的图片，固定用于PyPdf中"""
    reader = pdf_reader(pdf_path)
    os.makedirs(save_folder, exist_ok=True)

    for page_index, page in enumerate(reader.pages):
        if '/Resources' not in page or '/XObject' not in page['/Resources']:
            continue
        xobjects = page['/Resources']['/XObject'].get_object()
        for name, obj in xobjects.items():
            xobj = obj.get_object()
            if xobj['/Subtype'] == '/Image':
                filter_type = xobj.get('/Filter')
                img_data = xobj.get_data()
                ext = None

                if filter_type == '/DCTDecode':
                    ext = 'jpg'
                elif filter_type == '/JPXDecode':
                    ext = 'jp2'
                elif filter_type == '/FlateDecode':
                    # 原始像素流，需要构建图片
                    width = xobj['/Width']
                    height = xobj['/Height']
                    bpc = xobj['/BitsPerComponent']
                    color_space = xobj['/ColorSpace']
                    if color_space == '/DeviceRGB':
                        mode = "RGB"
                    elif color_space == '/DeviceGray':
                        mode = "L"
                    else:
                        mode = "RGB"  # 简化处理
                    image = Image.frombytes(mode, (width, height), img_data)
                    ext = 'png'
                else:
                    continue  # 无法识别的流

                if filter_type in ['/DCTDecode', '/JPXDecode']:
                    image = Image.open(io.BytesIO(img_data))

                image.save(os.path.join(save_folder, f"page{page_index+1}_{name[1:]}.{ext}"))


class PDFOnlyTextLoader:
    def __call__(self, filename:Optional[Union[str,bytes]],start_page:int=0,end_page=1000, **kwargs):

            try:

                self.loader = pdf_reader(filename if isinstance(filename,str) else BytesIO(filename),strict=False)
            #     with fitz.open(filename) as doc:
            #         for page_num in range(start_page, end_page):
            #             page = doc.load_page(page_num)
            #             page.get_images()
            #             page.get_image_bbox()
            #             text = page.get_text("text")
            #             print(f"第 {page_num + 1} 页内容：")
            #             print(text)
                # 尝试访问首页，强制触发XRef表、对象树解析
                _ =  self.loader.pages[0]

                # 访问页数（会检查页树完整性）
                _ = len( self.loader.pages)
                pages = []
                for i in range(start_page, end_page):
                    lines = []
                    page = self.loader.pages[i]
                    lines.extend([i.replace('\u3000','') for i in page.extract_text().split('\n') ])
                    pages.append((lines,i))
                outline = self.loader.outline

                if outline:
                    self.outline = process_outlines(outline, 0)
                return pages, self.loader.metadata,
            except PdfReadError as e:
                print(f"PDF结构错误: {e}")
            except Exception as e:
                print(f"PDF异常: {e}")


    @staticmethod
    def split_text_by_token(text:str, chunk_size:int=512, delimiter="\n。；！？\n\n", chunk_overlap_size=100):
        from langchain_text_splitters import TokenTextSplitter
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap_size
        )
        texts = text_splitter.split_text(text)
        return texts

