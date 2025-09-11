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
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import cupy as cp

# Use an environment variable (DEBUG=1) to toggle debug logging
DEBUG = os.environ.get("DEBUG", "0") == "1"
LOG_LEVEL = logging.DEBUG if DEBUG else logging.WARNING
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("reforge")
logger.setLevel(LOG_LEVEL)
logger.debug("Debug mode is enabled.")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="Bio")
warnings.filterwarnings("ignore") 


def timeit(*args, **kwargs):
    """Backwards-compatible timeit decorator"""
    # If called with no args, it's being used as @timeit
    if len(args) == 0:
        return _timeit(**kwargs)
    
    # If first arg is a function, it's being used as @timeit
    if len(args) == 1 and callable(args[0]):
        return _timeit()(args[0])
        
    # If first arg is a level, it's being used as @timeit(level=...)
    if len(args) == 1 and isinstance(args[0], int):
        return _timeit(level=args[0])
    
    # New style with explicit parameters
    return _timeit(*args, **kwargs)


def _timeit(level=logging.DEBUG, unit='s'):
    """Decorator to measure and log execution time of a function, with adjustable log level and time unit.
    
    Args:
        level (int): Logging level (default: logging.DEBUG)
        unit (str): Time unit to display. Options:
            - 'ms': milliseconds
            - 's': seconds (default)
            - 'm': minutes
            - 'auto': automatically choose best unit
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Convert to requested unit
            if unit == 'ms' or (unit == 'auto' and execution_time < 1):
                display_time = execution_time * 1000
                unit_str = 'milliseconds'
            elif unit == 'm' or (unit == 'auto' and execution_time > 60):
                display_time = execution_time / 60
                unit_str = 'minutes'
            else:  # seconds is default
                display_time = execution_time
                unit_str = 'seconds'
            
            logger.log(
                level,
                "Function '%s.%s' executed in %.6f %s",
                func.__module__,
                func.__name__,
                display_time,
                unit_str,
            )
            return result
        return wrapper
    return decorator


def memprofit(*args, **kwargs):
    """Backwards-compatible memory profiling decorator"""
    # If called with no args, it's being used as @memprofit
    if len(args) == 0:
        return _memprofit(**kwargs)
    
    # If first arg is a function, it's being used as @memprofit
    if len(args) == 1 and callable(args[0]):
        return _memprofit()(args[0])
        
    # If first arg is a level, it's being used as @memprofit(level=...)
    if len(args) == 1 and isinstance(args[0], int):
        return _memprofit(level=args[0])
    
    # New style with explicit parameters
    return _memprofit(*args, **kwargs)


def _memprofit(level=logging.DEBUG):
    """Decorator to profile and log the memory usage of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()  # Start memory tracking
            result = func(*args, **kwargs)  # Execute the function
            current, peak = tracemalloc.get_traced_memory()  # Get memory usage
            logger.log(
                level,
                "Memory usage after executing '%s.%s': %.2f MB, Peak: %.2f MB",
                func.__module__,
                func.__name__,
                current / 1024**2,
                peak / 1024**2,
            )
            tracemalloc.stop()  # Stop memory tracking
            return result
        return wrapper
    return decorator


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
