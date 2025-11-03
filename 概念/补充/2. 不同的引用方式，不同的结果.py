from FlagEmbedding import BGEM3FlagModel

# 初始化模型和 tokenizer
embedding = BGEM3FlagModel(
    model_name_or_path=r"D:\files\models\bge-m3",
    use_fp16=False
)

# 待处理的文本
text = "这是一段测试文本，用于演示 tokenizer 的编码和解码"

# 1. 编码（tokenize）：将文本转换为 token ids
# return_tensors="pt" 表示返回 PyTorch 张量，也可设为 "np" 返回 numpy 数组
tokenized = embedding.tokenizer(
    text,
    return_attention_mask=False,
    return_token_type_ids=False,
    add_special_tokens=True,  # 自动添加特殊符号（如 [CLS]、[SEP]）
    return_tensors="pt"  # 可选，根据需要选择返回类型
)

# 获取 token ids（编码后的数字序列）
token_ids = tokenized["input_ids"]
print("编码后的 token ids：", token_ids)

# 2. 解码（decode）：将 token ids 转换回文本
# 需要传入上面得到的 token_ids，注意如果是张量需先转为列表（或用 squeeze 处理）
decoded_text = embedding.tokenizer.decode(
    token_ids.squeeze().tolist()  # 移除多余维度并转为列表
)
print("解码后的文本：", decoded_text)