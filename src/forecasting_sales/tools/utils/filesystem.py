"""
Utilities for the filesystem
"""
from pathlib import Path


def root_directory() -> Path:
    """
    Function to get the root directory
    """
    return Path(__file__).parent.parent.parent.parent
