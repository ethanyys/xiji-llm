from vector_store.milvus_store import MilvusStore

connector = {
    "milvus": MilvusStore
}


class VectorStoreConnector:
    """
    Vector Store Connector, can connect different types of vector, provide api to load document、update document、similar search ...
    """
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.connector_class = connector[ctx["vector_store_type"]]
        self.client = self.connector_class(ctx)

    def load_document(self, docs):
        self.client.load_document(docs)

    def update_document(self, docs):
        self.client.update_document(docs)

    def query_document_metadatas(self):
        return self.client.query_document_metadatas()

    def similar_search(self, docs, top_k):
        return self.client.similar_search(docs, top_k)

    def similar_search_with_score(self, docs, top_k):
        return self.client.similar_search_with_score(docs, top_k)



