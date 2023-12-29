from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """base class for vector store implementations"""

    @abstractmethod
    def load_document(self, document) -> None:
        pass

    @abstractmethod
    def save_document(self, document) -> None:
        pass

    @abstractmethod
    def similar_search(self, text, topk) -> None:
        pass

    @abstractmethod
    def similar_search_with_threshold(self, text, topk) -> None:
        pass

