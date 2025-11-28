"""Utility wrappers and functions

Description:
    This module provides utility functions and decorators for the reForge workflow.
    It includes decorators for timing and memory profiling functions, a context manager
    for changing the working directory, and helper functions for cleaning directories and
    detecting CUDA availability.

Requirements:
    - Python 3.x

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

def get_logger(name="reforge"):
    """Get the configured logger instance.
    
    Since logging is now configured in reforge.__init__.py, this function
    simply returns the already-configured logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name (default: "reforge")
        
    Returns
    -------
    logging.Logger
        The configured logger instance
    """
    return logging.getLogger(name)


# Backward compatibility - provide logger at module level
# For new code, prefer: from reforge.utils import get_logger; logger = get_logger()
logger = get_logger()


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
