#!/usr/bin/env python
"""
Simple CG Protein
=================

Test Go-Martini setup

Requirements:
    - GROMACS
    - Python 3.x

Author: DY
"""
from pathlib import Path
import shutil
import pytest
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

# Create a gmxSystem instance for testing.
mdsys = GmxSystem("tests", "test_sys")
mdrun = GmxRun("tests", "test_sys", "test_run")
test_dir = Path("tests") / "test"
test_pdb = Path(test_dir / ".." / "1btl.pdb").resolve()
if test_dir.exists():
    shutil.rmtree(test_dir)

def test_prepare():
    mdsys.prepare_files()
    mdsys.clean_pdb_mm(test_pdb, add_missing_atoms=True, add_hydrogens=True, pH=7.0)
    mdsys.split_chains()


def test_martini_en():
    mdsys.martinize_proteins_en(ef=700, el=0.0, eu=0.9, p='backbone', pf=500, append=False)


def test_martini_go():
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)


def test_martinize_rna():
    """
    Test that martinize_rna() executes without error.
    """
    mdsys.martinize_rna()
    assert (Path(mdsys.topdir) / "chain_A.itp").exists()
    assert (Path(mdsys.topdir) / "chain_B.itp").exists()


def test_make_cg_structure():
    """
    Test that make_cg_structure() creates the solute PDB file.
    """
    mdsys.make_cg_structure()
    assert mdsys.solupdb.exists()


def test_make_cg_topology():
    """
    Test that make_cg_topology() creates the system topology file.
    """
    mdsys.make_cg_topology()
    assert mdsys.systop.exists()


if __name__ == '__main__':
    pytest.main([str(Path(__file__).resolve())])
