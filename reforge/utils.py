"""Utility wrappers and functions

Description:
    This module provides utility functions and decorators for the reForge workflow.
    It includes decorators for timing and memory profiling functions, a context manager
    for changing the working directory, and helper functions for cleaning directories and
    detecting CUDA availability.

Usage Example:
    >>> from utils import timeit, memprofit, cd, clean_dir, cuda_info
    >>>
    >>> @timeit
    ... def my_function():
    ...     # Function implementation here
    ...     pass
    >>>
    >>> with cd("/tmp"):
    ...     # Perform operations in /tmp
    ...     pass
    >>>
    >>> cuda_info()

Requirements:
    - Python 3.x
    - cupy
    - Standard libraries: logging, os, time, tracemalloc, contextlib, functools, pathlib

Author: DY
Date: YYYY-MM-DD
"""

import logging
import os
import time
import tracemalloc
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import cupy as cp

# Use an environment variable (DEBUG=1) to toggle debug logging
DEBUG = os.environ.get("DEBUG", "0") == "1"
LOG_LEVEL = logging.DEBUG if DEBUG else logging.WARNING
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("reforge")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.debug("Debug mode is enabled.")


def timeit(func):
    """Decorator to measure and log the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start timer
        result = func(*args, **kwargs)      # Execute the function
        end_time = time.perf_counter()      # End timer
        execution_time = end_time - start_time
        logger.debug(
            "Function '%s.%s' executed in %.6f seconds.",
            func.__module__,
            func.__name__,
            execution_time,
        )
        return result
    return wrapper


def memprofit(func):
    """Decorator to profile and log the memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()  # Start memory tracking
        result = func(*args, **kwargs)  # Execute the function
        current, peak = tracemalloc.get_traced_memory()  # Get memory usage
        logger.debug(
            "Memory usage after executing '%s.%s': %.2f MB, Peak: %.2f MB",
            func.__module__,
            func.__name__,
            current / 1024**2,
            peak / 1024**2,
        )
        tracemalloc.stop()  # Stop memory tracking
        return result
    return wrapper


@contextmanager
def cd(newdir):
    """
    Context manager to temporarily change the current working directory.

    Parameters:
        newdir (str or Path): The target directory to change into.

    Yields:
        None. After the context, reverts to the original directory.
    """
    prevdir = Path.cwd()
    os.chdir(newdir)
    logger.info("Changed working directory to: %s", newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)


def clean_dir(directory=".", pattern="#*"):
    """
    Remove files matching a specific pattern from a directory.

    Parameters:
        directory (str or Path, optional): Directory to search (default: current directory).
        pattern (str, optional): Glob pattern for files to remove (default: "#*").
    """
    directory = Path(directory)
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()


def cuda_info():
    """
    Check CUDA availability and log CUDA device information if available.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    if cp.cuda.is_available():
        logger.info("CUDA is available")
        device_count = cp.cuda.runtime.getDeviceCount()  # pylint: disable=c-extension-no-member
        logger.info("Number of CUDA devices: %s", device_count)
        return True
    logger.info("CUDA is not available")
    return False


def cuda_detected():
    """
    Check if CUDA is detected without logging detailed device information.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    if cp.cuda.is_available():
        return True
    logger.info("CUDA is not available")
    return False
