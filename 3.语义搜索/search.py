import numpy as np
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, FieldSchema, DataType, CollectionSchema
from pymilvus.milvus_client import IndexParams


def dense_search(client: MilvusClient, query_dense_embedding: list, collection_name: str, limit: int = 5,
                 output_fields: list[str] = None, timeout=None):
    search_params = {"efSearch": 64}
    res = client.search(
        collection_name=collection_name,
        data=[query_dense_embedding],
        limit=limit,
        output_fields=output_fields,
        metric_type="COSINE",
        timeout=timeout,
        anns_field="dense_vector",
        search_params=search_params
    )
    return res


def sparse_search(client: MilvusClient, query_sparse_embedding: list, collection_name: str, limit: int = 5,
                  output_fields: list[str] = None, timeout: float = None):
    res = client.search(
        collection_name=collection_name,
        data=query_sparse_embedding,
        limit=limit,
        output_fields=output_fields,
        metric_type="IP",
        timeout=timeout,
        anns_field="dense_sparse",
    )
    return res


def hybrid_search(client, query_dense_embedding, query_sparse_embedding, collection_name:str="test", sparse_weight:float=0.7, dense_weight:float=0.3,limit:int=5,
                  output_fields:list=None, timeout: float = None):
    if output_fields is None:
        output_fields = ['text']
    if output_fields is None:
        output_fields = ['text']
    dense_params = {"metric_type": "COSINE", "params": {"efSearch": 64}}
    sparse_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        data=query_dense_embedding,  # Union[List, utils.SparseMatrixInputType],
        anns_field="dense_vector",  # str,
        param=dense_params,  # Dict,
        limit=limit  # int,
        # expr=""#Optional[str] = None,
        # expr_params=#Optional[dict] = None,
    )
    sparse_req = AnnSearchRequest(
        data=query_sparse_embedding,  # Union[List, utils.SparseMatrixInputType],
        anns_field="dense_sparse",  # str,
        param=sparse_params,  # Dict,
        limit=limit  # int,
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = client.hybrid_search(
        reqs=[sparse_req, dense_req],
        collection_name=collection_name,
        ranker=rerank,
        output_fields=output_fields,
        timeout=timeout,
        limit=limit

    )
    return res


def init_milvus() -> MilvusClient:
    URI = "http://localhost:19530"
    vector_store = MilvusClient(
        # embedding_function = embeddings,
        connection_args={"uri": URI}
    )

    return vector_store
def normalize_vectors(dense_vec, sparse_vec):
    dense = [float(x) for x in dense_vec]  # numpy.float32 -> float
    sparse = {int(k): float(v) for k, v in sparse_vec.items()}
    return dense, sparse

if __name__ == '__main__':
    from FlagEmbedding import BGEM3FlagModel
    embeddings_bge = BGEM3FlagModel(
        model_name_or_path=r"D:\files\models\bge-m3",
        cache_folder=r"D:\files\models\bge-m3",
        use_fp16=True
    )
    # query_text="枸橼酸氯米芬胶囊成份的成分是什么"
    import pandas as pd
    df = pd.read_excel(r"D:\projects\learn_langchain_rag\parctise\ragas\new_ragas_testset_2.xlsx",engine="openpyxl")
    column_name = "问题（user_input）"  # 修改为你需要的列名
    question_list = df[column_name].tolist()
    search_result_list = []
    client = init_milvus()
    client.load_collection(collection_name="milvus_test")
    for i , query_text in enumerate(question_list):

        query_text_embedding = embeddings_bge.encode(
                sentences=query_text,  # Union[List[str], str],
                return_dense=True,  # Optional[bool] = None,
                return_sparse=True,  # Optional[bool] = None,
                return_colbert_vecs=True,  # Optional[bool] = None,
            )
        query_text_dense_vecs = query_text_embedding.get('dense_vecs')
        query_text_sparse_vecs = query_text_embedding.get('lexical_weights')

        dense,sparse = normalize_vectors(query_text_dense_vecs, query_text_sparse_vecs)
        res = hybrid_search(client, [dense], [sparse],"milvus_test", output_fields=["text", "metadata"])
        # res = sparse_search(client, query_text_sparse_vecs,"milvus_test", 5,["text", "metadata"])
        # res = hybrid_search(client, query_text_dense_vecs,query_text_sparse_vecs, "milvus_test", output_fields=["text", "metadata"])
        for i in res:
            for j in i:
                print(f"{j.get('distance')}------{j.get('entity')}")
        item = ''
        for i in res[0]:
            item += "\n\n"+i.get('entity').get('text')

        search_result_list.append(item)
    # 释放内容

    client.release_collection(collection_name="milvus_test")

    new_col = search_result_list + [None] * (len(df) - len(question_list))

    df["旧版检索结果"] = new_col
    df.to_excel(r"D:\projects\learn_langchain_rag\parctise\ragas\new_ragas_testset_3.xlsx", index=False,
                engine="openpyxl")

    # search_result_list.append()
