import json
from typing import List, Optional

import numpy as np
from pymilvus import Collection, DataType, connections, CollectionSchema, FieldSchema
from langchain.schema import Document

from vector_store.vector_store_base import VectorStoreBase
from utils.embedding_util import normalize_embeddings

import logging
logger = logging.getLogger(__name__)


class MilvusStore(VectorStoreBase):

    def __init__(self, ctx):
        self.ctx = ctx
        self.embeddings = ctx["embedding"]

        # default collection field
        self.id_field = "id"
        self.text_field = "text"
        self.metadata_field = "metadata"
        self.vector_field = "embedding"
        self.fields = [self.embeddings, self.id_field, self.text_field, self.vector_field]

        self.host = ctx["vector_store_host"]
        self.port = ctx["vector_store_port"]
        self.username = ctx["vector_store_username"]
        self.password = ctx["vector_store_password"]

        self.connect_name = ctx["vector_store_name"]
        self.collection_name = ctx["vector_store_name"] + "_collection"

        if (self.username is None) != (self.password is None):
            raise ValueError(
                "Both Username and Password must be set to use authentication for Milvus"
            )

        self.index_type = "HNSW"
        self.index_params = {
            "metric_type": "IP",
            "index_type": self.index_type,
            "param": {"M": 8, "efConstruction": 64},
        }

        # use HNSW by default
        self.index_params_map = {
            "HNSW": {"params": {"ef": 10}}
        }


    def _connect_server(self):
        try:
            connections.connect(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                alias=self.connect_name
            )
            logger.info("connect milvus server successfully")
        except Exception as e:
            logging.exception("fail to connect milvus server")

    def _init_collection(self):
        if not connections.has_connection(self.connect_name):
            self._connect_server()

        id = FieldSchema(
            name=self.id_field,
            dtype=DataType.VARCHAR,
            max_length=50,
            is_sparse=True
        )
        text = FieldSchema(
            name=self.text_field,
            dtype=DataType.VARCHAR,
            max_length=1000
        )
        metadata = FieldSchema(
            name=self.metadata_field,
            dtype=DataType.VARCHAR,
            max_length=65535
        )
        embedding = FieldSchema(
            name=self.vector_field,
            dtype=DataType.FLOAT_VECTOR,
            dim=self.embeddings.client.get_sentence_embedding_dimension()
        )
        schema = CollectionSchema(
            fields=[id, text, metadata, embedding],
            description=self.collection_name,
            enable_dynamic_shape=True
        )
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.connect_name
        )

        self.collection.create_index(
            field_name="id",
            index_name="id_index"
        )
        self.collection.create_index(
            field_name="embedding",
            index_name=self.index_params
        )

        logging.info("""create index for collection: {}""".format(self.collection_name))

    def load_document(self, documents):
        """add document embeddings to collection"""
        batch_size = 50
        batch_list = [
            documents[i: i+batch_size] for i in range(0, len(documents), batch_size)
        ]

        for batch in batch_list:
            texts = [doc.page_content for doc in batch]
            metadatas = [json.dumps(doc.metadata, ensure_ascii=False) for doc in batch]
            ids = [doc.metadata["id"] for doc in batch]

            batch_insert_dict = {
                "id": list(ids),
                "text": list(texts),
                "metadata": list(metadatas)
            }
            try:
                batch_insert_dict["embedding"] = self.embeddings.embed_documents(list(texts))
            except NotImplementedError as e:
                batch_insert_dict["embedding"] = [
                    self.embeddings.embed_query(x) for x in texts
                ]

            batch_insert_dict["embedding"] = normalize_embeddings(np.array(batch_insert_dict["embedding"])).tolist()

            insert_list = [batch_insert_dict[x] for x in self.fields]
            res = self.collection.insert(insert_list)
            self.collection.flush()

    def delete_entity(self, ids):
        """delete entity from collection"""
        batch_size = 50
        batch_list = [
            ids[i: i + batch_size] for i in range(0, len(ids), batch_size)
        ]

        for batch_ids in batch_list:
            expr = "id in " + str(batch_ids)
            self.collection.delete(expr)
        self.collection.flush()

    def update_document(self, documents):
        ids = [doc.metadata["id"] for doc in documents]
        self.delete_entity(ids)
        self.load_document(documents)

    def similar_search(self, text, top_k):
        docs_and_scores = self._search(text, top_k)
        return [doc for doc, _ in docs_and_scores]

    def similar_search_with_threshold(self, text, top_k):
        docs_and_scores = self._search(text, top_k)
        return docs_and_scores

    def _search(self,
                query: str,
                k: int = 4,
                param: Optional[dict] = None,
                expr: Optional[str] = None,
                partition_names: Optional[List[str]] = None,
                round_decimals: int = -1,
                timeout: Optional[int] = None,
                **kwargs):

        self.collection.load()
        if param is None:
            param = self.index_params_map[self.index_type]
        data = normalize_embeddings(np.array(self.embeddings.embed_query(query)))

        output_fields = self.fields[:]
        output_fields.remove(self.vector_field)

        res = self.collection.search(
            data,
            self.vector_field,
            param,
            k,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            round_decimal=round_decimals,
            timeout=timeout,
            **kwargs
        )
        ret = []
        for result in res[0]:
            ret.append(
                (
                    Document(
                        page_content=result.entity.get(self.text_field),
                        metadata=json.loads(result.entity.get(self.metadata_field))
                    ),
                    result.distince
                )
            )
        return ret

    def close(self):
        connections.disconnect()
