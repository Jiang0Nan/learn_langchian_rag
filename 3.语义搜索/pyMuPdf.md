fitz.Document（文档对象）
    ├── fitz.Page（页面对象）
    │     ├── get_text()     → 提取文本
    │     ├── get_images()   → 获取页面中的图像信息
    │     ├── get_image_bbox() → 获取图像在页面中的位置（可选）
    │     ├── get_drawings() → 获取矢量绘图对象
    │     └── get_links()    → 获取超链接
    └── metadata、outline、page_count 等属性
    
    
| 方法                   | 说明                    |
| -------------------- | --------------------- |
| `get_text("text")`   | 提取纯文字                 |
| `get_text("blocks")` | 返回每个文本块（坐标 + 内容）      |
| `get_text("dict")`   | 结构化输出，带字体、位置、颜色等      |
| `get_images()`       | 返回页面中所有嵌入图片的信息        |
| `get_drawings()`     | 获取矢量路径信息（线、矩形、表格框等）   |
| `get_links()`        | 获取超链接目标               |
| `get_pixmap()`       | 将整页渲染成图片              |
| `rect`               | 返回页面的边界（fitz.Rect 对象） |
