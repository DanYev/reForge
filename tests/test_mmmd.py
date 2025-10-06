"""
Test Suite for reforge.mdsystem.mmmd Package and mm_md Workflow
=====================================
"""

from pathlib import Path
import pytest
import shutil
import sys
from reforge.mdsystem.mmmd import MmSystem, MmRun

# Add the project root to Python path to find workflows
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from workflows import mm_md

# Create a mm_md instance for testing.
sysdir = 'tests'
sysname = 'mm_sys'
runname = 'mm_run'
mdsys = MmSystem(sysdir, sysname)
mdrun = MmRun(sysdir, sysname, runname)
in_pdb = '../1btl.pdb'

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    if mdsys.root.exists():
        shutil.rmtree(mdsys.root)
    yield mdsys  # This runs the tests
    # shutil.rmtree(mdsys.root) # Cleanup - only runs if tests complete successfully

def test_setup_aa():
    mm_md.setup_aa(sysdir, sysname)
    assert mdsys.syspdb.exists()

def test_md_nve():
    mm_md.md_nve(sysdir, sysname, runname)
    md_file = mdrun.root / 'md.trr'
    assert md_file.exists()

def test_md_npt():
    mm_md.md_npt(sysdir, sysname, runname)
    md_file = mdrun.root / 'md.trr'
    assert md_file.exists()

def test_extend():
    mm_md.extend(sysdir, sysname, runname)
    md_file = mdrun.root / 'md_1.trr'
    assert md_file.exists()

def test_trjconv():
    mm_md.trjconv(sysdir, sysname, runname)
    samples_file = mdrun.root / 'samples.trr'
    top_file = mdrun.root / 'topology.pdb'
    assert samples_file.exists()
    assert top_file.exists()