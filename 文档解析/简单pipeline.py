#!/usr/bin/env python3
"""
pdf_page_to_md.py

功能：
  - 对指定 PDF 的单页（page_num，1-indexed）执行：
      1) 布局检测（PubLayNet via layoutparser） -> 检测 Text/Table/Figure/Title/List
      2) OCR（pytesseract）提取文本（对每个 Text/Title/List 块）
      3) 表格提取（pdfplumber 尝试精确解析表格为 DataFrame -> Markdown）
      4) 对 Figure/Formula 区域裁切保存图片，并将图片插入 Markdown
      5) 输出该页的 Markdown 文件与布局可视化图片
注意：
  - 公式不会被自动转为 LaTeX，只会以图片形式保存并在 Markdown 中插入占位。
  - 对扫描件、中文文档，建议把 pytesseract 换为 PaddleOCR；此处使用 pytesseract 保持依赖最小。
Usage:
  python pdf_page_to_md.py input.pdf 3 out_dir
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import pdfplumber
import layoutparser as lp
import numpy as np
import pandas as pd

# -------- CONFIG ----------
MODEL_LP = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"  # layoutparser detectron2 PubLayNet
TESSERACT_LANG = "eng"  # 如果是中文改为 "chi_sim"（需安装对应语言包）
FONT_PATH = None  # 若要在可视化上显示标签，指定系统字体路径或留 None
DPI = 200
MIN_OCR_CONF = 30  # pytesseract confidence threshold (0-100)
# --------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_layout_model():
    # Detectron2LayoutModel 会自动下载模型 config / weights（需要 detectron2 支持）
    print("Loading layout model:", MODEL_LP)
    model = lp.Detectron2LayoutModel(MODEL_LP, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4])
    return model

def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = DPI) -> Image.Image:
    # convert_from_path uses 1-indexed pages
    pil_pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)
    if not pil_pages:
        raise RuntimeError(f"Cannot render page {page_num}")
    return pil_pages[0].convert("RGB")

def detect_layout(image: Image.Image, model) -> lp.Layout:
    arr = np.array(image)
    layout = model.detect(arr)
    # layout is a Layout (list of LayoutElement)
    return layout

def sort_layout(layout: lp.Layout) -> lp.Layout:
    # sort by top (y), then left (x)
    sorted_layout = lp.Layout([b for b in layout])
    sorted_layout.sort(key=lambda b: (b.block.y_1, b.block.x_1))
    return sorted_layout

def ocr_crop_text(image: Image.Image, bbox: Tuple[int,int,int,int], lang: str = TESSERACT_LANG) -> Tuple[str, int]:
    """
    使用 pytesseract 对裁剪区域 OCR，返回 (text, conf_avg)
    bbox: (x0, y0, x1, y1) in pixels
    """
    crop = image.crop(bbox)
    # pytesseract image_to_data 获得每个 word 的置信度，返回 dict
    data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, lang=lang)
    texts = []
    confs = []
    for i, txt in enumerate(data.get("text", [])):
        t = (txt or "").strip()
        if t:
            texts.append(t)
            try:
                confs.append(int(data.get("conf", [])[i]))
            except:
                pass
    joined = " ".join(texts).strip()
    conf_avg = int(np.mean(confs)) if confs else 0
    return joined, conf_avg

def visualize_layout(image: Image.Image, layout: lp.Layout, out_path: Path, font_path: str = FONT_PATH):
    draw = ImageDraw.Draw(image)
    font = None
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 14)
    except Exception:
        font = None
    color_map = {
        "Text": (0,0,0),
        "Title": (200,0,0),
        "List": (0,150,0),
        "Table": (0,0,200),
        "Figure": (150,0,150),
    }
    for i, b in enumerate(layout):
        x0, y0, x1, y1 = list(map(int, b.block.x_1 - (b.block.width), [0,0,0,0])) if False else (int(b.block.x_1 - b.block.width), int(b.block.y_1 - b.block.height), int(b.block.x_1), int(b.block.y_1))  # placeholder not used
        # simpler: layoutparser block has .coordinates
        x0, y0, x1, y1 = int(b.block.x_1 - b.block.width), int(b.block.y_1 - b.block.height), int(b.block.x_1), int(b.block.y_1)
        # but above is awkward; better use b.block.x_1, b.block.y_1 etc — use b.block.x_1 etc are coordinates of bottom-right, and width/height are b.width,b.height
        x0 = int(b.block.x_1 - b.width) if hasattr(b, "width") else int(b.block.x_1 - b.block.width)
        y0 = int(b.block.y_1 - b.height) if hasattr(b, "height") else int(b.block.y_1 - b.block.height)
        x1 = int(b.block.x_1)
        y1 = int(b.block.y_1)
        label = b.type
        color = color_map.get(label, (255,0,0))
        draw.rectangle([x0,y0,x1,y1], outline=color, width=3)
        text_pos = (x0+3, max(y0-18, 0))
        draw.text(text_pos, label, fill=color, font=font)
    image.save(out_path)
    print("Saved layout visualization:", out_path)

def crop_and_save(image: Image.Image, bbox: Tuple[int,int,int,int], out_path: Path):
    crop = image.crop(bbox)
    crop.save(out_path)
    return out_path

def bbox_intersection(a, b):
    # a, b are (x0,y0,x1,y1)
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 < x0 or y1 < y0:
        return 0
    return (x1-x0)*(y1-y0)

def pdfplumber_extract_tables_near_bbox(pdf_path: str, page_num: int, target_bbox: Tuple[int,int,int,int]) -> List[pd.DataFrame]:
    # pdfplumber page numbers are 0-indexed
    dfs = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num-1]
        # find tables (table extraction objects)
        tables = page.find_tables()
        for t in tables:
            # table.bbox is (x0, top, x1, bottom) in PDF coordinate (origin bottom-left?) -> use t.bbox
            try:
                tbbox = t.bbox  # (x0, top, x1, bottom)
                # Convert pdfplumber bbox to pixel coordinates relative to page image size:
                # We will get page.width and page.height in PDF points. When rendering to image we used DPI; approximate mapping
                # Simpler approach: try to extract table as DataFrame
                df = t.extract()
                dfs.append(df)
            except Exception:
                continue
    return dfs

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"]*len(cols)) + " |"
        rows = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join([str(x) for x in r.values]) + " |")
        return "\n".join([header, sep] + rows)

# -------- Main conversion for one page ----------
def convert_pdf_page_to_md(pdf_path: str, page_num: int, out_dir: str):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    image = pdf_page_to_image(pdf_path, page_num, dpi=DPI)
    layout_model = load_layout_model()
    layout = detect_layout(image, layout_model)
    # layout elements: type in {'Text','Title','List','Table','Figure'}
    layout = layout.sort(key=lambda b: (b.block.y_1 - b.block.height, b.block.x_1 - b.block.width))  # top-left order

    # Prepare outputs
    md_lines = []
    resources_dir = out_dir / "resources"
    ensure_dir(resources_dir)

    # attempt table extraction for entire page (pdfplumber)
    table_dfs = []
    try:
        table_dfs = pdfplumber_extract_tables_near_bbox(pdf_path, page_num, None)
    except Exception:
        table_dfs = []

    table_used = 0
    img_count = 0
    formula_count = 0

    for i, block in enumerate(layout):
        # layoutparser block has coordinates as block.x_1 etc. We'll get bounding box in pixel coords:
        # layout.block format: x_1, y_1, width, height where (x_1 - width, y_1 - height) = top-left
        x0 = int(block.block.x_1 - block.block.width)
        y0 = int(block.block.y_1 - block.block.height)
        x1 = int(block.block.x_1)
        y1 = int(block.block.y_1)
        bbox = (max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1))

        if block.type in ("Title", "Text", "List"):
            text, conf = ocr_crop_text(image, bbox, lang=TESSERACT_LANG)
            # heuristic: if OCR result seems like math (lots of symbols), treat as formula image
            is_formula = False
            if len(text) < 3:
                # too short -> maybe image or formula; save image as fallback and put placeholder
                is_formula = True
            else:
                # detect math-like patterns
                sym_count = sum(1 for ch in text if ch in "=+−×÷^_{}[]()\\/%$<>" )
                if sym_count / max(1, len(text)) > 0.12:
                    is_formula = True
            if is_formula:
                formula_count += 1
                fname = f"page_{page_num:03d}_formula_{formula_count}.png"
                crop_and_save(image, bbox, resources_dir / fname)
                md_lines.append(f"\n\n$$\n![formula]({resources_dir.name}/{fname})\n$$\n\n")
            else:
                # map Titles -> heading, Lists -> bullet
                if block.type == "Title":
                    md_lines.append("# " + text)
                elif block.type == "List":
                    # crude split by lines
                    for ln in text.splitlines():
                        if ln.strip():
                            md_lines.append("- " + ln.strip())
                else:
                    md_lines.append(text)
        elif block.type == "Table":
            # try to pick a table df from pdfplumber results
            if table_used < len(table_dfs):
                df = table_dfs[table_used]
                md = dataframe_to_markdown(df)
                md_lines.append("\n\n" + md + "\n\n")
                table_used += 1
            else:
                # fallback: crop and save table image and placeholder
                table_img_name = f"page_{page_num:03d}_table_{table_used+1}.png"
                crop_and_save(image, bbox, resources_dir / table_img_name)
                md_lines.append(f"\n\n<!-- TABLE_IMAGE: {resources_dir.name}/{table_img_name} -->\n\n")
                table_used += 1
        elif block.type == "Figure":
            img_count += 1
            fname = f"page_{page_num:03d}_fig_{img_count}.png"
            crop_and_save(image, bbox, resources_dir / fname)
            md_lines.append(f"![figure]({resources_dir.name}/{fname})")
        else:
            # unknown block, do OCR -> append raw text
            text, conf = ocr_crop_text(image, bbox, lang=TESSERACT_LANG)
            if text.strip():
                md_lines.append(text)

    # Save layout visualization
    vis_img = image.copy()
    vis_path = out_dir / f"page_{page_num:03d}_layout.png"
    # layoutparser has convenient draw_box method, but we implement simple drawing:
    try:
        # draw using block bboxes and labels
        draw = ImageDraw.Draw(vis_img)
        font = ImageFont.load_default()
        for b in layout:
            x0 = int(b.block.x_1 - b.block.width)
            y0 = int(b.block.y_1 - b.block.height)
            x1 = int(b.block.x_1)
            y1 = int(b.block.y_1)
            draw.rectangle([x0,y0,x1,y1], outline="red", width=2)
            draw.text((x0+3, max(y0-12,0)), b.type, font=font, fill="red")
        vis_img.save(vis_path)
        print("Saved layout visualization:", vis_path)
    except Exception as e:
        print("Visualization failed:", e)

    # Combine Markdown
    md_content = "\n\n".join([ln for ln in md_lines if ln.strip() != ""])
    md_path = out_dir / f"page_{page_num:03d}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Saved markdown:", md_path)
    return md_path

# ------------------ CLI ------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pdf_page_to_md.py <input.pdf> <page_num(1-indexed)> <out_dir>")
        sys.exit(1)
    pdf = sys.argv[1]
    page = int(sys.argv[2])
    out = sys.argv[3]
    convert_pdf_page_to_md(pdf, page, out)
