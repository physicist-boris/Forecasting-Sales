from abc import ABC, abstractmethod

from typing import Any


class AbstractNode(ABC):
    """
    congig or blueprint class for all execution nodes
    """
    @staticmethod
    @abstractmethod
    def process(*args: Any, **kwargs: Any) -> Any:
        """
        Node to process data
        """
        raise NotImplementedError