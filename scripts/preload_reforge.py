#!/usr/bin/env python3
"""
Pre-import reforge and common dependencies for faster interactive sessions.

Usage in interactive Python:
    python -i scripts/preload_reforge.py
    
Or add to PYTHONSTARTUP:
    export PYTHONSTARTUP=/path/to/reforge/scripts/preload_reforge.py
    python
"""
print("Pre-loading reforge modules...")

import numpy as np
import reforge
from reforge import rfgmath, io, pdbtools, utils, mdm

# Import commonly used submodules
try:
    from reforge.rfgmath import rcmath, rpymath
    print("✓ rfgmath modules loaded")
except ImportError as e:
    print(f"⚠ Could not load rfgmath: {e}")

try:
    from reforge.forge import enm
    print("✓ forge modules loaded")
except ImportError as e:
    print(f"⚠ Could not load forge: {e}")

print("Ready! All common reforge modules are pre-imported.")
print("Available: reforge, np, rcmath, rpymath, enm, io, pdbtools, utils, mdm")
