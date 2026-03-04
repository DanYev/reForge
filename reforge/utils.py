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
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


def get_logger(name="reforge", level=logging.INFO):
    """FOR BACKWARDS COMPATIBILITY. Get a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", 
            datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


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


#####################################################################
# Failure and error handling utilities
#####################################################################

T = TypeVar("T")

def cleanup_failure(
	_func: Optional[Callable[..., T]] = None,
	*,
	mdruns_dirname: str = "mdruns",
	log_glob: str = "*.log",
	remove_system_on_failure: bool = False,
) -> Union[Callable[[Callable[..., T]], Callable[..., T]], Callable[..., T]]:
	"""Decorator to clean up generated directories on failure.

	If the wrapped function raises an exception, this decorator will:
	1) Determine a target directory from the wrapped function's args:
	   - If args look like (sysdir, sysname, runname, ...): delete
	     Path(sysdir)/sysname/mdruns/<runname>
	   - If args look like (sysdir, sysname, ...): delete Path(sysdir)/sysname
	   - If remove_system_on_failure=True: always delete Path(sysdir)/sysname
	2) Before deleting, find all matching log files (default: '*.log') under the
	   target directory and copy them into the *execution directory* (the current
	   working directory at the moment the wrapped function is entered).

	Usage:
		@cleanup_failure
		def md_npt(sysdir, sysname, runname): ...

		@cleanup_failure(remove_system_on_failure=True)
		def setup_martini(sysdir, sysname): ...
	"""

	def decorator(func: Callable[..., T]) -> Callable[..., T]:
		@wraps(func)
		def wrapper(*args, **kwargs):
			_dest = Path(sys.argv[0]).parent
			try:
				return func(*args, **kwargs)
			except Exception:
				try:
					target_dir = _infer_target_dir(
						args,
						mdruns_dirname=mdruns_dirname,
						remove_system_on_failure=remove_system_on_failure,
					)
					if target_dir is not None and target_dir.exists():
						# If we are currently inside the directory we're about to delete,
						# rmtree may fail (cwd becomes an unlinked dir). Move to a safe dir.
						try:
							cwd_r = Path.cwd().resolve()
							target_r = target_dir.resolve()
							if cwd_r.is_relative_to(target_r):
								os.chdir(_dest)
						except Exception:
							pass

						_copy_logs(target_dir, _dest, log_glob=log_glob)
						try:
							shutil.rmtree(target_dir)
						except Exception as rm_exc:
							print(f"cleanup_failure: failed to remove {target_dir}: {rm_exc}", file=sys.stderr)
				except Exception as cleanup_exc:
					print(f"cleanup_failure: cleanup error: {cleanup_exc}", file=sys.stderr)
				raise

		return wrapper

	# Support both @cleanup_failure and @cleanup_failure(...)
	if _func is not None:
		return decorator(_func)
	return decorator


def _infer_target_dir(
	args: tuple,
	*,
	mdruns_dirname: str,
	remove_system_on_failure: bool,
) -> Optional[Path]:
	if len(args) < 2:
		return None

	sysdir = args[0]
	sysname = args[1]
	if not isinstance(sysdir, (str, os.PathLike)) or not isinstance(sysname, (str, os.PathLike)):
		return None

	sys_root = Path(sysdir) / sysname
	if remove_system_on_failure:
		return sys_root

	# If runname is provided, default to cleaning only that run directory.
	if len(args) >= 3:
		runname = args[2]
		if isinstance(runname, (str, os.PathLike)):
			return sys_root / mdruns_dirname / Path(runname)

	return sys_root


def _copy_logs(target_dir: Path, dest_dir: Path, *, log_glob: str) -> None:
	if not dest_dir.exists():
		dest_dir.mkdir(parents=True, exist_ok=True)

	try:
		target_dir_r = target_dir.resolve()
	except Exception:
		target_dir_r = target_dir

	for log_path in target_dir_r.rglob(log_glob):
		if not log_path.is_file():
			continue
		dest_path = dest_dir / log_path.name
		try:
			if dest_path.exists():
				dest_path.unlink()
			shutil.copy2(str(log_path), str(dest_path))
		except Exception:
			pass


