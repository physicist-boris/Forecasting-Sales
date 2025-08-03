from abc import ABC, abstractmethod

from typing import Any


class AbstractNode(ABC):
    """
    congig or blueprint class for all execution nodes
    """

    @abstractmethod
    def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
        """
        Save data locally
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(*args: Any, **kwargs: Any) -> Any:
        """
        Load locally saved data
        """
        raise NotImplementedError

    @abstractmethod
    def load_source(*args: Any, **kwargs: Any) -> Any:
        """
        Load the sources for execution node
        """
        raise NotImplementedError
