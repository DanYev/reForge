"""
Test Suite for reforge.mdsystem.mmmd Package and mm_md Workflow
=====================================
"""

from pathlib import Path
import pytest
import shutil
import sys
import os
import errno
import time
from reforge.mdsystem.mmmd import MmSystem, MmRun

# Add the project root to Python path to find workflows
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from workflows import mm_md
from workflows.mm_md import INPDB  # Import the constant

# Create a mm_md instance for testing.
sysdir = 'tests'
sysname = 'mm_sys'
runname = 'mm_run'
mdsys = MmSystem(sysdir, sysname)
mdrun = MmRun(sysdir, sysname, runname)
in_pdb = Path('workflows').resolve() / 'structures' / '1PZP.pdb'


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    if mdsys.root.exists():
        shutil.rmtree(mdsys.root)
    yield mdsys  # This runs the tests
    # os.chdir(project_root)  # Ensure we're back in the project root before cleanup
    # shutil.rmtree(mdsys.root) # Need them for the next tests

def test_setup_aa():
    mdsys.prepare_files()
    shutil.copy(in_pdb, mdsys.root / INPDB)
    mm_md.setup_aa(sysdir, sysname)
    assert mdsys.syspdb.exists()

def test_md_nve():
    mm_md.md_nve(sysdir, sysname, runname, nsteps=10000)  
    md_file = mdrun.rundir / 'md.trr'
    assert md_file.exists()
    
def test_md_npt():
    mm_md.md_npt(sysdir, sysname, runname, nsteps=10000)
    md_file = mdrun.rundir / 'md.trr'
    assert md_file.exists()

def test_extend():
    mm_md.extend(sysdir, sysname, runname, until_time=40) # in ps
    md_file = mdrun.rundir / 'md_1.trr'
    assert md_file.exists()

def test_trjconv():
    mm_md.trjconv(sysdir, sysname, runname)
    samples_file = mdrun.rundir / 'samples.trr'
    top_file = mdrun.rundir / 'topology.pdb'
    assert samples_file.exists()
    assert top_file.exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])