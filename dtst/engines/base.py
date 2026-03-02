from abc import ABC, abstractmethod


class SearchEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def search(self, query: str, page: int) -> list[str]:
        pass
