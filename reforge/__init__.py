# my_package/__init__.py

import os
import importlib
import warnings
import logging

# Global warning suppression for MDAnalysis
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
# Suppress specific MDAnalysis guesser messages
warnings.filterwarnings("ignore", message=".*types have already been read from the topology file.*")
warnings.filterwarnings("ignore", message=".*guesser will only guess empty values.*")
# Also suppress the logger-based INFO messages from MDAnalysis
logging.getLogger('MDAnalysis.topology.guessers').setLevel(logging.WARNING)
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)

__all__ = []
do_not_import = ["__init__.py", "insane.py", "martinize_nucleotides_old.py"]

package_dir = os.path.dirname(__file__)
for module in os.listdir(package_dir):
    if module.endswith(".py") and module not in do_not_import:
        module_name = module[:-3]
        imported_module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = imported_module
        __all__.append(module_name)
