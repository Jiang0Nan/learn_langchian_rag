import lear_langchain.ragflow.api.db.services.user_service as us

print("模块真实路径:", us.__file__)
print("模块识别名:", us.__name__)
from lear_langchain.ragflow.rag.app.naive import chunk
def progress_cb(progress=None, msg=""):
    try:
        p = 0.0 if progress is None else float(progress)
    except Exception:
        p = 0.0
    print(f"[chunk] {p:.2f} | {msg}")


if __name__ == '__main__':

    test_file = r"D:\projects\learn_langchain_rag\langchain\2.消息，模板，chatModel\file\drug description\非奈利酮片[190125,190124].pdf"


    # 可根据需要调整 parser_config，例如纯文本快速解析：{"layout_recognize": "Plain Text"}
    result = chunk(
        test_file,
        lang="Chinese",
        callback=progress_cb,
        parser_config={"layout_recognize":"Plain Text","chunk_token_num": 512},
    )
    print(f"Total chunks: {len(result)}")
    if result:
        print("First item:")
        print(result[0])

