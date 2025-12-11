
import os.path

import fitz
import layoutparser as lp
import cv2

# from transformers import AutoModel
# model = AutoModel.from_pretrained("microsoft/layoutlmv3-base", dtype="auto" ,cache_dir=r"D:\files\models")
# ===============定义超参数===========
dpi = 150
zoom = dpi/72
save_path = "./images"
page_index = 3
if not os.path.exists(save_path):
    os.makedirs(save_path)
# =============获取文件===========
file = fitz.open(r"D:\files\学习笔记\学习资料\1706.03762v7.pdf")
# ========获取需要处理的内容========
first_page = file[page_index]
# 缩放
mat = fitz.Matrix(zoom, zoom)
# 转为像素
pix = first_page.get_pixmap(matrix=mat, alpha=False)
# 设置保存路径和文件名
output_filename = os.path.join(save_path, f"page_{page_index}.png")
pix.save(output_filename)

file.close()

# ==========版面识别==============
image = cv2.imread(output_filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Layout Parser版本: {lp.__version__}")
print(f"可用模型: {lp.models.get_all_available_models()}")
model = lp.AutoLayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
layout = model.detect(image)
lp.draw_box(image, layout, box_width=3).show()
# 合并