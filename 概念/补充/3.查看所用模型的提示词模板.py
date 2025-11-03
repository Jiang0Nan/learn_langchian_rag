from gguf import GGUFReader

gguf_file = r"D:\app\ollama_models\blobs\sha256-96c415656d377afbff962f6cdb2394ab092ccbcbaab4b82525bc4ca800fe8a49"
reader = GGUFReader(gguf_file)

print("🔍 GGUF Metadata (包含提示模板相关信息):\n")
for key, value in reader.fields.items():
    if "tokenizer" in key.lower() or "template" in key.lower() or "prompt" in key.lower():
        print(f"{key}: {value}")
# 获取聊天模板字段
chat_template_field = reader.fields["tokenizer.chat_template"]

# 解码 memmap 的 uint8 为字符串
chat_template_bytes = chat_template_field.parts[-1]  # 最后一部分是内容
chat_template = bytes(chat_template_bytes).decode("utf-8")

print("💬 Chat Template:\n")
print(chat_template)