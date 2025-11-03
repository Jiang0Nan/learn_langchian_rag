from langchain_community.document_loaders import DirectoryLoader,UnstructuredPDFLoader

loader=DirectoryLoader(
    path=r"D:\projects\learn_langchain_rag\langchain\概念\8.document loader\file",
    glob="**/*.pdf", # Union[List[str], Tuple[str], str] = "**/[!.]*",
    show_progress=True,#: bool = False,
    loader_cls=UnstructuredPDFLoader,
    use_multithreading=True,#使用多线程
    loader_kwargs={
        "mode": "elements",  # 一定要用 elements 才能看到每一块
        "strategy": "hi_res",  # 使用高分辨率 OCR 模式
        "infer_table_structure": True,
    }
)
docs = loader.load()
