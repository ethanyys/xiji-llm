
from project_1.vector_store.vector_store_base import VectorStoreBase


class MilvusStore(VectorStoreBase):

    def __init__(self, ctx):
        self.ctx = ctx

