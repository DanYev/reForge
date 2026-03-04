"""
===============================================================================
File: test_pdbtools.py
Description:
    This module contains unit tests for the pdbtools module. It covers tests for
    the Atom, AtomList, and AtomListCollection classes, as well as tests for higher-
    level functions such as pdb2system, pdb2atomlist, sort_pdb, clean_pdb,
    rename_chain_in_pdb, and rename_chain_and_histidines_in_pdb.

Usage:
    Run the tests with pytest:
        pytest -v tests/test_pdbtools.py

Requirements:
    - Python 3.x
    - pytest
    - pdbfixer and OpenMM (for testing clean_pdb, optional)
    
Author: DY
Date: 2025-02-27
===============================================================================
"""

import os
import tempfile
import shutil
import pytest
from reforge.pdbtools import *

TEST_PDB = "workflows/structures/dsRNA.pdb"

# ---------------------------------------------------------------------------
# Tests for the Atom class
# ---------------------------------------------------------------------------

def test_atom_from_pdb_line():
    """Test parsing a PDB line into an Atom instance using from_pdb_line."""
    pdb_line = "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00 20.00           C  \n"
    atom = Atom.from_pdb_line(pdb_line)
    assert atom.record == "ATOM"
    assert atom.atid == 1
    assert atom.name == "CA"
    assert atom.resname == "ALA"
    assert atom.chid == "A"
    assert atom.resid == 1
    assert abs(atom.x - 11.104) < 1e-3
    assert abs(atom.y - 13.207) < 1e-3
    assert abs(atom.z - 10.000) < 1e-3
    assert atom.occupancy == 1.00
    assert atom.bfactor == 20.00
    assert atom.element == "C"

def test_atom_to_pdb_line():
    """Test converting an Atom instance to a properly formatted PDB line."""
    atom = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                11.104, 13.207, 10.000, 1.00, 20.00, "SEG1", "C", "")
    pdb_line = atom.to_pdb_line()
    assert "CA" in pdb_line
    assert "ALA" in pdb_line
    assert "11.104" in pdb_line
    assert "13.207" in pdb_line
    assert "10.000" in pdb_line

def test_atom_repr():
    """Test the __repr__ method of the Atom class returns a meaningful string."""
    atom = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                11.104, 13.207, 10.000, 1.00, 20.00, "SEG1", "C", "")
    rep = repr(atom)
    assert "ATOM" in rep
    assert "ALA" in rep
    assert "A1" in rep

# ---------------------------------------------------------------------------
# Fixtures for AtomList and AtomListCollection tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_atoms():
    """Fixture that returns an AtomList containing two sample Atom instances."""
    atom1 = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 2, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    return AtomList([atom1, atom2])

@pytest.fixture
def sample_atomlist_collection():
    """Fixture that returns an AtomListCollection made from two separate AtomLists."""
    atom1 = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 2, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    alist1 = AtomList([atom1])
    alist2 = AtomList([atom2])
    return AtomListCollection([alist1, alist2])

# ---------------------------------------------------------------------------
# Tests for the AtomList class
# ---------------------------------------------------------------------------

def test_atomlist_add(sample_atoms):
    """Test adding two AtomLists using the __add__ method."""
    atom3 = Atom("ATOM", 3, "CG", "", "ALA", "A", 1, "",
                 3.0, 4.0, 5.0, 1.0, 10.0, "SEG1", "C", "")
    list2 = AtomList([atom3])
    combined = sample_atoms + list2
    assert len(combined) == len(sample_atoms) + 1
    assert combined[-1].name == "CG"

def test_atomlist_properties(sample_atoms):
    """Test that all property getters of AtomList return expected values."""
    assert sample_atoms.names == ["CA", "CB"]
    assert sample_atoms.atids == [1, 2]
    assert sample_atoms.alt_locs == ["", ""]
    assert sample_atoms.resnames == ["ALA", "ALA"]
    assert sample_atoms.chids == ["A", "A"]
    assert sample_atoms.resids == [1, 1]
    assert sample_atoms.icodes == ["", ""]
    assert sample_atoms.xs == [1.0, 2.0]
    assert sample_atoms.ys == [2.0, 3.0]
    assert sample_atoms.zs == [3.0, 4.0]
    assert sample_atoms.occupancies == [1.0, 1.0]
    assert sample_atoms.bfactors == [10.0, 10.0]
    assert sample_atoms.segids == ["SEG1", "SEG1"]
    assert sample_atoms.elements == ["C", "C"]
    assert sample_atoms.charges == ["", ""]
    assert sample_atoms.vecs == [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)]

def test_atomlist_setters(sample_atoms):
    """Test that property setters of AtomList properly update the atoms."""
    sample_atoms.names = ["N", "CA"]
    assert sample_atoms.names == ["N", "CA"]
    sample_atoms.xs = [10.0, 20.0]
    assert sample_atoms.xs == [10.0, 20.0]
    for atom, x_val in zip(sample_atoms, [10.0, 20.0]):
        assert atom.vec[0] == x_val

def test_atomlist_sort():
    """Test sorting of an AtomList using the default key."""
    atom1 = Atom("ATOM", 3, "CA", "", "ALA", "B", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 1, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    atom3 = Atom("ATOM", 2, "CG", "", "ALA", "A", 1, "",
                 3.0, 4.0, 5.0, 1.0, 10.0, "SEG1", "C", "")
    atoms = AtomList([atom1, atom2, atom3])
    atoms.sort()
    sorted_ids = [atom.atid for atom in atoms]
    assert sorted_ids == [1, 2, 3]

def test_atomlist_mask(sample_atoms):
    """Test the mask method of AtomList to select atoms based on name."""
    masked = sample_atoms.mask("CA", mode="name")
    assert len(masked) == 1
    assert masked[0].name == "CA"

def test_atomlist_mask_out(sample_atoms):
    """Test the mask_out method of AtomList to filter out atoms by name."""
    masked_out = sample_atoms.mask_out("CA", mode="name")
    assert len(masked_out) == 1
    assert masked_out[0].name == "CB"

def test_atomlist_renum():
    """Test renumbering of atoms in an AtomList using renum method."""
    atom1 = Atom("ATOM", 10, "CA", "", "ALA", "A", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 20, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    atoms = AtomList([atom1, atom2])
    atoms.renum()
    # In this version, renum sets atids starting from 0.
    assert atoms.atids == [0, 1]

def test_atomlist_remove_atoms():
    """Test removal of specific atoms from an AtomList."""
    atom1 = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 2, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    atoms = AtomList([atom1, atom2])
    atoms.remove_atoms([atom1])
    assert len(atoms) == 1
    assert atoms[0].name == "CB"

def test_atomlist_write_read(tmp_path):
    """Test writing an AtomList to a file and reading it back."""
    file_path = tmp_path / "atoms_test.pdb"
    atom1 = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 2, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    atoms = AtomList([atom1, atom2])
    atoms.write_pdb(str(file_path))
    new_atoms = AtomList()
    new_atoms.read_pdb(str(file_path))
    assert len(new_atoms) == 2
    assert new_atoms.names == atoms.names

def test_atomlist_write_ndx(tmp_path):
    """Test writing an NDX file from an AtomList."""
    file_path = tmp_path / "atoms_test.ndx"
    atom1 = Atom("ATOM", 1, "CA", "", "ALA", "A", 1, "",
                 1.0, 2.0, 3.0, 1.0, 10.0, "SEG1", "C", "")
    atom2 = Atom("ATOM", 2, "CB", "", "ALA", "A", 1, "",
                 2.0, 3.0, 4.0, 1.0, 10.0, "SEG1", "C", "")
    atoms = AtomList([atom1, atom2])
    atoms.write_ndx(str(file_path), header="[ TestGroup ]", append=False, wrap=1)
    with open(str(file_path), "r") as f:
        content = f.read()
    assert "[ TestGroup ]" in content
    assert "1" in content
    assert "2" in content

# ---------------------------------------------------------------------------
# Tests for the AtomListCollection class
# ---------------------------------------------------------------------------

def test_collection_names(sample_atomlist_collection):
    """Test that the names property of AtomListCollection returns a list of lists."""
    names = sample_atomlist_collection.names
    assert names == [["CA"], ["CB"]]

def test_collection_atids(sample_atomlist_collection):
    """Test that the atids property of AtomListCollection returns a list of lists."""
    atids = sample_atomlist_collection.atids
    assert atids == [[1], [2]]

def test_collection_vecs(sample_atomlist_collection):
    """Test that the vecs property of AtomListCollection returns a list of lists of vectors."""
    vecs = sample_atomlist_collection.vecs
    assert vecs == [[(1.0, 2.0, 3.0)], [(2.0, 3.0, 4.0)]]

def test_collection_properties(sample_atomlist_collection):
    """Test other properties of AtomListCollection (records, alt_locs, resnames, chids)."""
    coll = sample_atomlist_collection
    for rec_list in coll.records:
        assert all(r == "ATOM" for r in rec_list)
    for alt_list in coll.alt_locs:
        assert all(alt == "" for alt in alt_list)
    for res_list in coll.resnames:
        assert res_list == ["ALA"]
    for chid_list in coll.chids:
        assert chid_list == ["A"]

# ---------------------------------------------------------------------------
# Tests for higher-level pdbtools functions using dsRNA.pdb
# ---------------------------------------------------------------------------

def test_pdb2system():
    """Test that pdb2system successfully parses a PDB file and returns a non-empty system."""
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    assert len(atoms) > 0, "pdb2system returned an empty atom list."

def test_pdb2atomlist():
    """Test that pdb2atomlist returns a non-empty AtomList."""
    atoms = pdb2atomlist(TEST_PDB)
    assert len(atoms) > 0, "pdb2atomlist returned an empty list."

def test_save_system():
    """Test writing a system to a PDB file."""
    test_out = "test_system.pdb"
    if os.path.exists(test_out):
        os.remove(test_out)
    system = pdb2system(TEST_PDB)
    system.write_pdb(test_out)
    assert os.path.exists(test_out), "System file was not created."
    os.remove(test_out)

def test_save_atoms():
    """Test writing atoms to a PDB file."""
    test_out = "test_atoms.pdb"
    if os.path.exists(test_out):
        os.remove(test_out)
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    atoms.write_pdb(test_out)
    assert os.path.exists(test_out), "Atom file was not created."
    os.remove(test_out)

def test_chain():
    """Test that a chain can be correctly retrieved from a parsed system."""
    system = pdb2system(TEST_PDB)
    model = system.models[1]
    chids = list(model.chains.keys())
    chain = model.chains[chids[0]]
    chain_list = model.select_chains(chids)
    assert chain in chain_list, "Chain not found in selected chains."

def test_vecs():
    """Test that the vecs property of the system's atoms matches manually computed vectors."""
    system = pdb2system(TEST_PDB)
    model = system.models[1]
    all_atoms = []
    for chain in model:
        all_atoms += chain.atoms
    atoms_list = AtomList(all_atoms)
    vecs = [atom.vec for atom in atoms_list]
    assert atoms_list.vecs == vecs, "Vector lists do not match."

def test_segids():
    """Test that setting segids equal to chain IDs works correctly."""
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    atoms.segids = atoms.chids
    segids = set(atom.segid for atom in atoms)
    chids = set(atoms.chids)
    assert segids == chids, "Segment IDs do not match chain IDs."

def test_sort():
    """Test that sorting the system's atoms results in the expected order."""
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    atoms.sort()  
    sorted_atids = sorted(atom.atid for atom in atoms)
    current_atids = [atom.atid for atom in atoms]
    assert current_atids == sorted_atids, "Atoms are not sorted correctly."

def test_mask():
    """Test that masking atoms by chain id returns only atoms with the specified id."""
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    chids = list(set(atoms.chids))
    chid = chids[0]
    masked_atoms = atoms.mask(chid, mode="chid")
    test_chid = list(set(masked_atoms.chids))[0]
    assert chid == test_chid, "Masked atoms do not have the expected chain id."

def test_remove():
    """Test that removing atoms using remove_atoms works as expected."""
    mask_vals = ["P", "C3'"]
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    initial_len = len(atoms)
    masked_atoms = atoms.mask(mask_vals)
    atoms.remove_atoms(masked_atoms)
    assert initial_len - len(atoms) == len(masked_atoms), "Atom removal did not work as expected."

def test_sort_pdb():
    """Test that sort_pdb creates a sorted output PDB file."""
    out_pdb = 'test_sort.pdb'
    if os.path.exists(out_pdb):
        os.remove(out_pdb)
    sort_pdb(TEST_PDB, out_pdb)
    assert os.path.exists(out_pdb), "Sorted PDB file was not created."
    os.remove(out_pdb)

def test_write_ndx():
    """Test that an NDX file can be written from the system's atoms."""
    out_ndx = 'test.ndx'
    if os.path.exists(out_ndx):
        os.remove(out_ndx)
    system = pdb2system(TEST_PDB)
    atoms = system.atoms
    atoms.write_ndx(out_ndx, header='[ System ]', append=False, wrap=15)
    assert os.path.exists(out_ndx), "NDX file was not created."
    os.remove(out_ndx)

def test_rename_chain_in_pdb():
    """Test that rename_chain_in_pdb updates all chain identifiers to the specified value."""
    temp_pdb = "temp_rename.pdb"
    shutil.copy(TEST_PDB, temp_pdb)
    new_chain = "X"
    rename_chain_in_pdb(temp_pdb, new_chain)
    atoms = pdb2atomlist(temp_pdb)
    unique_chids = set(atoms.chids)
    assert unique_chids == {new_chain}, "Chain renaming failed."
    os.remove(temp_pdb)

def test_rename_chain_and_histidines_in_pdb():
    """
    Test that rename_chain_and_histidines_in_pdb updates chain identifiers and
    renames histidine residue names as specified.
    """
    temp_pdb = "temp_rename_hist.pdb"
    with open(temp_pdb, "w") as f:
        f.write("ATOM      1  CA  HSD A   1      11.104  13.207  10.000  1.00 20.00           C  \n")
        f.write("ATOM      2  CA  HSE A   2      12.104  14.207  11.000  1.00 20.00           C  \n")
        f.write("END\n")
    new_chain = "Y"
    rename_chain_and_histidines_in_pdb(temp_pdb, new_chain)
    atoms = pdb2atomlist(temp_pdb)
    unique_chids = set(atoms.chids)
    assert unique_chids == {new_chain}, "Chain renaming (with histidine update) failed."
    resnames = set(atoms.resnames)
    assert "HIS" in resnames, "HSD was not renamed to HIS."
    assert "HIE" in resnames, "HSE was not renamed to HIE`."
    os.remove(temp_pdb)

# Optional: Test clean_pdb if pdbfixer is available.
def test_clean_pdb():
    """Test that clean_pdb creates a non-empty cleaned PDB file."""
    tmp_dir = tempfile.mkdtemp()
    try:
        try:
            from pdbfixer import pdbfixer  # noqa: F401
        except ImportError:
            pytest.skip("pdbfixer not installed, skipping clean_pdb test")
        temp_in = os.path.join(tmp_dir, "temp_in.pdb")
        temp_out = os.path.join(tmp_dir, "temp_out.pdb")
        shutil.copy(TEST_PDB, temp_in)
        clean_pdb(str(temp_in), str(temp_out), add_missing_atoms=False, add_hydrogens=False)
        assert os.path.exists(str(temp_out)), "clean_pdb did not create the output file."
        with open(str(temp_out), "r") as f:
            contents = f.read()
        assert len(contents) > 0, "clean_pdb output file is empty."
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
