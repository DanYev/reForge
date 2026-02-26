# my_package/__init__.py

import os
import importlib
import warnings
import logging

# Configure logging first
debug = os.environ.get("DEBUG", "0") == "1"
log_level = logging.DEBUG if debug else logging.INFO
_LOG_FORMAT = "[%(filename)s:%(lineno)d] - %(levelname)s - %(name)s - %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Set up main package logger
logging.basicConfig(
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        # level=log_level,
        # force=True,
    )
logger = logging.getLogger("reforge")
logger.setLevel(log_level)
logger.info("reforge package initialized with log level: %s", logging.getLevelName(log_level))

if debug:
    logger.debug("reforge package initialized in debug mode")


# Global warning suppression for MDAnalysis
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
# Suppress specific MDAnalysis guesser messages
warnings.filterwarnings("ignore", message=".*types have already been read from the topology file.*")
warnings.filterwarnings("ignore", message=".*guesser will only guess empty values.*")
# Also suppress the logger-based INFO messages from MDAnalysis
logging.getLogger('MDAnalysis.topology.guessers').setLevel(logging.WARNING)
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)

# Make logger available at package level
__all__ = [logger]
do_not_import = ["__init__.py", "insane.py", "martinize_nucleotides_old.py", "plotting.py"]

# Lazy loading: modules are imported on first access
def __getattr__(name):
    """Lazy import modules on first access."""
    package_dir = os.path.dirname(__file__)
    module_file = f"{name}.py"
    
    if module_file in do_not_import:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path = os.path.join(package_dir, module_file)
    if os.path.exists(module_path):
        module = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = module
        return module
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# List available modules for introspection
package_dir = os.path.dirname(__file__)
for module in os.listdir(package_dir):
    if module.endswith(".py") and module not in do_not_import:
        module_name = module[:-3]
        __all__.append(module_name)
