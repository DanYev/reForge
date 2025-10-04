"""
Test Suite for reforge.mdsystem.gmxmd Package and gmx_md Workflow
=====================================
"""

from pathlib import Path
import shutil
import pytest
import sys
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

# Add the project root to Python path to find workflows
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from workflows import gmx_md

# Create a gmxSystem instance for testing.
sysdir = 'tests'
sysname = 'test_sys'
runname = 'test_run'
mdsys = GmxSystem(sysdir, sysname)
mdrun = GmxRun(sysdir, sysname, runname)
in_pdb = '../dsRNA.pdb'

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    if mdsys.root.exists():
        shutil.rmtree(mdsys.root)
    yield mdsys  # This runs the tests
    shutil.rmtree(mdsys.root) # Cleanup - only runs if tests complete successfully

def test_prepare_files():
    mdsys.prepare_files()

def test_sort_input_pdb():
    mdsys.sort_input_pdb(in_pdb)
    assert (Path(mdsys.root) / "inpdb.pdb").exists()

def test_gmx():
    mdsys.gmx('')

def test_clean_pdb_gmx():
    mdsys.clean_pdb_gmx(clinput='6\n7\n', ignh='yes')

def test_split_chains():
    mdsys.split_chains()
    assert (Path(mdsys.nucdir) / "chain_A.pdb").exists()
    assert (Path(mdsys.nucdir) / "chain_B.pdb").exists()

def test_clean_chains_gmx():
    mdsys.clean_chains_gmx(clinput='6\n7\n', ignh='yes')
    assert (Path(mdsys.nucdir) / "chain_A.pdb").exists()
    assert (Path(mdsys.nucdir) / "chain_B.pdb").exists()

def test_setup_martini():
    gmx_md.setup_martini(sysdir, sysname)

def test_md_npt():
    gmx_md.md_npt(sysdir, sysname, runname)

def test_extend():
    gmx_md.extend(sysdir, sysname, runname)

def test_trjconv():
    gmx_md.trjconv(sysdir, sysname, runname)