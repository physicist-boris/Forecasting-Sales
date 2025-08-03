"""
Module for logging utilities
"""
import logging  # pylint: disable=import-self
import time
from functools import wraps
from typing import Any, Callable
from inspect import getmodule

def log_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for logging node execution time.

    Args:
        func: Function to be executed.
    Returns:
        Decorator for logging the running time.
    """

    @wraps(func)
    def with_log(*args: Any, **kwargs: Any) -> Any:
        """
        Decorator to run execution of function or method.
        """
        log = logging.getLogger(__name__)
        log.info("Running %s - %s",
                 getmodule(func).__name__, # type: ignore
                 func.__qualname__)
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        elapsed = time_end - time_start
        log.info("The execution of %s -  %s took %0.2f seconds",
                 getmodule(func).__name__, # type: ignore
                 func.__qualname__,
                 elapsed)
        return result

    return with_log
