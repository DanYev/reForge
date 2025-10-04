"""
Test Suite for reforge.mdsystem.gmxmd Package
=====================================

This module contains unit tests for the functions provided by the 
`reforge.gmxmd` package (and related CLI commands). These tests verify
the correct behavior of file preparation, PDB sorting, Gromacs command 
execution, PDB cleaning, chain splitting, and other functionality related 
to setting up molecular dynamics (MD) simulations with Gromacs.

Usage:
    Run the tests with pytest from the project root:

        pytest -v tests/test_gmxmd.py

Author: DY
"""

from pathlib import Path
import shutil
import pytest
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.cli import run

# Create a gmxSystem instance for testing.
mdsys = GmxSystem('tests', 'test_sys')
mdrun = GmxRun('tests', 'test_sys', 'test_run')
in_pdb = '../dsRNA.pdb'

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    if mdsys.root.exists():
        shutil.rmtree(mdsys.root)
    yield mdsys  # This runs the tests
    shutil.rmtree(mdsys.root) # Cleanup - only runs if tests complete successfully

def test_prepare_files():
    """
    Test that mdsys.prepare_files() correctly prepares the file structure.
    """
    test_dir = Path("tests") / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    mdsys.prepare_files()

def test_sort_input_pdb():
    """
    Test that sort_input_pdb() properly sorts and renames the input PDB file.
    """
    mdsys.sort_input_pdb(in_pdb)
    assert (Path(mdsys.root) / "inpdb.pdb").exists()

def test_gmx():
    """
    Test that mdsys.gmx() executes without error.
    """
    mdsys.gmx('')

def test_clean_pdb_gmx():
    """
    Test that clean_pdb_gmx() processes the PDB file as expected.
    """
    mdsys.clean_pdb_gmx(clinput='6\n7\n', ignh='yes')

def test_split_chains():
    """
    Test that split_chains() outputs chain files as expected.
    """
    mdsys.split_chains()
    assert (Path(mdsys.nucdir) / "chain_A.pdb").exists()
    assert (Path(mdsys.nucdir) / "chain_B.pdb").exists()

def test_clean_chains_gmx():
    """
    Test that clean_chains_gmx() processes chain PDB files as expected.
    """
    mdsys.clean_chains_gmx(clinput='6\n7\n', ignh='yes')
    assert (Path(mdsys.nucdir) / "chain_A.pdb").exists()
    assert (Path(mdsys.nucdir) / "chain_B.pdb").exists()


