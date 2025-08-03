"""
Utilities to load and save pickle objects
"""

import pickle

from typing import Any, Union
from pathlib import Path

def save(obj: Any, path: Union[str, Path]) -> None:
    """
    Saves the object as a pickle
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=-1)


def load(path: Union[str, Path]) -> Any:
    """
    Load the object from pickle
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
