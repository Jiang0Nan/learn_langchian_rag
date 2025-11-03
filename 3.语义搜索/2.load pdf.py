from pprint import pprint

from langchain_community.document_loaders import PyPDFLoader

file_path = r"D:\projects\learn_langchain_rag\langchain\2.消息，模板，chatModel\file\drug description\非奈利酮片[190125,190124].pdf"
def loader_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages
if __name__ == '__main__':

    a = loader_pdf(file_path)
    pprint(a)
