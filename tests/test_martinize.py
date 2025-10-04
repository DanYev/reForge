#!/usr/bin/env python
"""
Simple CG Protein and RNA
=================

Test martinizing functions in MDSystem class.

Requirements:
    - GROMACS
    - Python 3.x
    - Vermouth

Author: DY
"""
from pathlib import Path
import pytest
import shutil
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

# Global variables for tests
mdsys = GmxSystem("tests", "test_sys")
mdrun = GmxRun("tests", "test_sys", "test_run")
protein_pdb = mdsys.root / ".." / "1btl.pdb"
rna_pdb = mdsys.root / ".." / "dsRNA.pdb"

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    if mdsys.root.exists():
        shutil.rmtree(mdsys.root)
    yield mdsys  # This runs the tests
    shutil.rmtree(mdsys.root) # Cleanup - only runs if tests complete successfully

def test_prepare():
    mdsys.prepare_files(pour_martini=True)
    mdsys.clean_pdb_mm(protein_pdb, add_missing_atoms=True, add_hydrogens=True, pH=7.0)
    mdsys.split_chains()
    assert (Path(mdsys.prodir) / "chain_A.pdb").exists()

def test_martini_en():
    mdsys.martinize_proteins_en(ef=700, el=0.0, eu=0.9, p='backbone', pf=500, append=False)
    assert (Path(mdsys.topdir) / "chain_A.itp").exists()

def test_martini_go():
    mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=1.1, p="backbone", pf=500, append=False)
    assert (Path(mdsys.topdir) / "chain_A.itp").exists()

def test_martinize_rna():
    shutil.copy(rna_pdb, mdsys.nucdir / "chain_AB.pdb")
    mdsys.martinize_rna(merge=True, elastic="yes", ef=100, el=0.5, eu=1.2, p="backbone", pf=500, append=False)
    assert (Path(mdsys.topdir) / "chain_AB.itp").exists()

def test_make_cg_structure():
    mdsys.make_cg_structure()
    assert mdsys.solupdb.exists()

def test_make_cg_topology():
    mdsys.make_cg_topology()
    assert mdsys.systop.exists()

