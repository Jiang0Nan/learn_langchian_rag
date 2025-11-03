import time
from typing import Optional

from pymilvus import FieldSchema, DataType, MilvusClient, CollectionSchema, AnnSearchRequest, WeightedRanker
from pymilvus.milvus_client import IndexParams


class MilvusServer:
    def __init__(self, url: str = "http://localhost:19530",
                 collection_name: str = 'milvus_test'):
        self.collection_name = collection_name
        self.vector_store = MilvusClient(
            # embedding_function = embeddings,
            connection_args={"uri": url},

        )

    def init_collection(self, dense_dim: int = 1024, colbert_dim: int = 1024, ):
        # 参数
        schema_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=32),
            # todo 后面多个文档时启用
            FieldSchema(name="doc_id", description='文档的id', dtype=DataType.VARCHAR,max_length=32),
            FieldSchema(name='filename', description='文档名', dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name='content', description='原文本', dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name='tokenizer_content', description='原文本分词后', dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name='tokenizer_fine_grained', description='', dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name='create_time', description='创建时间', dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name='create_timestamp', description='时间戳', dtype=DataType.INT64),
            # FieldSchema(name='vec', description='文档名或文档id的bge-m3向量',dtype=DataType.FLOAT_VECTOR,dim = vec_dim),
            FieldSchema(name="page_ids", description='页码', dtype=DataType.VARCHAR, max_length=512),

            FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, description="嵌入后的密集向量", dim=dense_dim),
            # FieldSchema(name="colbert_vecs", dtype=DataType.FLOAT_VECTOR, description="嵌入后的密集向量", dim=colbert_dim,nullable=True),
            FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR, description="嵌入后的稀疏向量"),
            # # 稀疏向量不指定dim
        ]

        if self.vector_store.has_collection(collection_name=self.collection_name):
            self.vector_store.drop_collection(collection_name=self.collection_name)
        #     注意：部分参数会被schema的覆盖，即，schema的参数优先级更高
        self.vector_store.create_collection(
            collection_name=self.collection_name,
            timeout=20,  # 设置操作超时时间
            schema=CollectionSchema(schema_fields, enable_dynamic_field=True),
            consistency_level="Session",
            drop_old=True,
        )

        # 索引

    def flush(self):
        try:
            self.vector_store.flush(collection_name=self.collection_name)
        except Exception as e:
            print(e)

    def create_indexes(self, dense_M: int = 16, dense_ef_construction: int = 100):
        try:
            existing_index = self.vector_store.describe_index(collection_name=self.collection_name)
        except Exception:
            existing_index = None
        if not existing_index:
            sparse_index = IndexParams()
            sparse_index.add_index("sparse_vec", "SPARSE_INVERTED_INDEX", metric_type="IP")

            dense_index = IndexParams()
            dense_index.add_index("dense_vec", "HNSW", metric_type="COSINE", M=dense_M,
                                  efConstruction=dense_ef_construction)

            self.vector_store.create_index(
                collection_name=self.collection_name,
                index_params=dense_index
            )
            self.vector_store.create_index(
                collection_name=self.collection_name,
                index_params=sparse_index
            )

    def insert(self, data: list, embedding_results: dict, batch_size: int = 32 ):

        batch_size = batch_size

        for i in range(0, len(data), batch_size):
            max_try = 3
            now_try_times = 0

            while now_try_times < max_try:
                try:

                    res = self.vector_store.insert(
                        collection_name=self.collection_name,
                        data=data[i:i + batch_size],
                    )
                    print(f'成功插入，{res}')
                    break
                except Exception as e:
                    now_try_times += 1
                    print(f'插入失败{e}，重试中{now_try_times}/{max_try}...')
                    time.sleep(2 ** now_try_times)

    def search(self):
        pass

    def dense_search(self, query_dense_embedding: list, collection_name: str = None, limit: int = 5,
                     output_fields: list[str] = None, timeout=None,filter_str:str = "",):
        """
        query_dense_embedding : 查询的稠密向量
        collection_name : 搜索的表名
        limit: 返回前几个
         output_fields : 返回的字段名
         timeout : 设置查询时间限制单位s
        """
        search_params = {"efSearch": 100}
        res = self.vector_store.search(
            collection_name=collection_name if collection_name else self.collection_name,
            data=[query_dense_embedding],
            limit=limit,
            output_fields=output_fields,
            timeout=timeout,
            anns_field="dense_vec",
            search_params=search_params,
            filter=filter_str,
        )
        return res

    def sparse_search(self, query_sparse_embedding: list, collection_name: str = None, limit: int = 5,
                      output_fields: list[str] = None, timeout: float = None,filter_str:str = "",):
        res = self.vector_store.search(
            collection_name=collection_name if collection_name else self.collection_name,
            data=[query_sparse_embedding],
            limit=limit,
            output_fields=output_fields,
            metric_type="IP",
            timeout=timeout,
            anns_field="sparse_vec",
            filter = filter_str
        )
        return res

    def hybrid_search(self, query_dense_embedding, query_sparse_embedding, collection_name: str = None,
                      sparse_weight: float = 0.7, dense_weight: float = 0.3, limit: int = 5,
                      output_fields: list = None, timeout: float = None,expr :str = None,expr_params : Optional[dict] = None,):
        if output_fields is None:
            output_fields = ['text']
        if output_fields is None:
            output_fields = ['text']
        dense_params = {"metric_type": "COSINE", "params": {"efSearch": 100}}
        sparse_params = {"metric_type": "IP", "params": {'drop_ratio_search':0}}
        dense_req = AnnSearchRequest(
            data=[query_dense_embedding],  # Union[List, utils.SparseMatrixInputType],
            anns_field="dense_vec",
            param=dense_params,
            limit=limit,
            expr= expr ,  #过滤表达式（类似 SQL 的 where 语句）
            expr_params=expr_params,
        )
        sparse_req = AnnSearchRequest(
            data=[query_sparse_embedding],  # Union[List, utils.SparseMatrixInputType],
            anns_field="sparse_vec",
            param=sparse_params,
            limit=limit,
            expr=expr,
            expr_params = expr_params
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.vector_store.hybrid_search(
            reqs=[sparse_req, dense_req],
            collection_name=collection_name if collection_name else self.collection_name,
            ranker=rerank,
            output_fields=output_fields,
            timeout=timeout,
            limit=limit
        )
        return res

    def load_collection(self, collection_name: str = None,timeout : Optional[float] = None , replica_number:int = 3):
        if collection_name is None:
            collection_name = self.collection_name

        self.vector_store.load_collection(collection_name, timeout=timeout, replica_number=replica_number)

    def release_collection(self, collection_name: str = None,timeout:Optional[float]=None, ):
        if collection_name is None:
            collection_name = self.collection_name
        self.vector_store.release_collection(collection_name,timeout)
