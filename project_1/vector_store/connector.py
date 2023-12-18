from project_1.vector_store.milvus_store import MilvusStore

connector = {
    "milvus": MilvusStore
}


class VectorStoreConnector:
    """
    Vector Store Connector, can connect different types of vector, provide api to load document、update document、similar search ...
    """

    def __init__(self, ctx) -> None:
        self.ctx = ctx



