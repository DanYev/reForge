# my_package/__init__.py

import os
import importlib
import warnings
import logging

# Configure logging first
debug = os.environ.get("DEBUG", "0") == "1"
_LOG_FORMAT = "[%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
_DATE_FORMAT = "%H:%M:%S"

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )
else:
    # If something configured logging before `reforge` was imported, `basicConfig`
    # is a no-op. Update existing handlers to keep log lines consistent.
    _formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    for _handler in root_logger.handlers:
        _handler.setFormatter(_formatter)

# Set up main package logger
logger = logging.getLogger()
log_level = logging.DEBUG if debug else logging.INFO
logger.setLevel(log_level)