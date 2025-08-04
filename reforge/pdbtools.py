# pylint: disable=too-many-lines, too-many-instance-attributes, too-many-arguments, too-many-locals, missing-function-docstring, too-many-public-methods, unnecessary-lambda-assignment, unspecified-encoding, broad-exception-caught, undefined-variable, invalid-name, import-outside-toplevel, f-string-without-interpolation, too-few-public-methods, unused-import

"""Classes and functions for parsing and manipulating PDB atoms

Description:
    This module provides utilities for parsing, manipulating, and writing PDB files.
    It defines classes for representing individual atoms (Atom), groups of atoms (AtomList),
    as well as hierarchical representations of residues, chains, models, and entire systems.
    In addition, helper functions are provided to read and write PDB files and GROMACS index
    (NDX) files, and to perform common operations such as sorting and cleaning PDB files.

Usage:
    from pdbtools import pdb2system, pdb2atomlist, sort_pdb, clean_pdb
    system = pdb2system("input.pdb")
    atoms = pdb2atomlist("input.pdb")

Requirements:
    - Python 3.x
    - pathlib and typing (standard library)
    - pdbfixer and OpenMM (for cleaning PDB files, optional)
    
Author: DY
Date: 2025-02-27
"""

import sys
from pathlib import Path
from typing import List

###################################
## Classes and Functions ##
###################################

class Atom:
    """Represents an ATOM or HETATM record from a PDB file.
    
    This class intentionally contains many attributes to capture all fields from a PDB record.
    """
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        record,
        atid,
        name,
        alt_loc,
        resname,
        chid,
        resid,
        icode,
        x,
        y,
        z,
        occupancy,
        bfactor,
        segid,
        element,
        charge,
    ):
        self.record = record  # "ATOM" or "HETATM"
        self.atid = atid      # Atom serial number
        self.name = name      # Atom name
        self.alt_loc = alt_loc  # Alternate location indicator
        self.resname = resname  # Residue name
        self.chid = chid      # Chain identifier
        self.resid = resid    # Residue sequence number
        self.icode = icode    # Insertion code
        self.x = x            # x coordinate
        self.y = y            # y coordinate
        self.z = z            # z coordinate
        self.occupancy = occupancy  # Occupancy
        self.bfactor = bfactor      # Temperature factor
        self.segid = segid          # Segment identifier
        self.element = element      # Element symbol
        self.charge = charge        # Charge on the atom
        self.vec = (self.x, self.y, self.z)

    @classmethod
    def from_pdb_line(cls, line):
        """Parse a PDB line and return an Atom instance."""
        record = line[0:6].strip()
        atid = int(line[6:11])
        name = line[12:16].strip()
        alt_loc = line[16].strip()
        resname = line[17:20].strip()
        chid = line[21].strip()
        resid = int(line[22:26])
        icode = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occupancy_str = line[54:60].strip()
        occupancy = float(occupancy_str) if occupancy_str else 0.0
        bfactor_str = line[60:66].strip()
        bfactor = float(bfactor_str) if bfactor_str else 0.0
        segid = line[72:76].strip()
        element = line[76:78].strip()
        charge = line[78:80].strip()
        return cls(record, atid, name, alt_loc, resname, chid, resid, icode,
                   x, y, z, occupancy, bfactor, segid, element, charge)

    def __repr__(self):
        return ("<Atom {} {} {} {}{} seg:{} "
                "({:.3f}, {:.3f}, {:.3f})>").format(
                    self.record, self.atid, self.name,
                    self.resname, f"{self.chid}{self.resid}",
                    self.segid, self.x, self.y, self.z)

    def to_pdb_line(self):
        """Format the Atom as a fixed-width PDB line."""
        name_string = f"{self.name:^4}"
        if len(self.name) == 3:
            name_string = f" {self.name}"
        line = (
            f"{self.record:<6}"
            f"{self.atid:>5} "
            + name_string +
            f"{self.alt_loc:1}"
            f"{self.resname:>3} "
            f"{self.chid:1}"
            f"{self.resid:>4}"
            f"{self.icode:1}   "
            f"{self.x:>8.3f}"
            f"{self.y:>8.3f}"
            f"{self.z:>8.3f}"
            f"{self.occupancy:>6.2f}"
            f"{self.bfactor:>6.2f}"
            f"{'':6}"
            f"{self.segid:>4}"
            f"{self.element:>2}"
            f"{self.charge:>2}"
        )
        return line


class AtomList(list):
    """A list of Atom objects with convenient attribute access."""
    # Disable warning about lambda assignment
    key_funcs = {
        "record": lambda atom: atom.record,
        "atid": lambda atom: atom.atid,
        "name": lambda atom: atom.name,
        "alt_loc": lambda atom: atom.alt_loc,
        "resname": lambda atom: atom.resname,
        "chid": lambda atom: atom.chid,
        "resid": lambda atom: atom.resid,
        "icode": lambda atom: atom.icode,
        "x": lambda atom: atom.x,
        "y": lambda atom: atom.y,
        "z": lambda atom: atom.z,
        "occupancy": lambda atom: atom.occupancy,
        "bfactor": lambda atom: atom.bfactor,
        "segid": lambda atom: atom.segid,
        "element": lambda atom: atom.element,
        "charge": lambda atom: atom.charge,
    }

    def __add__(self, other):
        return AtomList(super().__add__(other))

    @property
    def records(self):
        return [atom.record for atom in self]

    @records.setter
    def records(self, new_records):
        if len(new_records) != len(self):
            raise ValueError("Length mismatch for records.")
        for i, rec in enumerate(new_records):
            self[i].record = rec

    @property
    def atids(self):
        return [atom.atid for atom in self]

    @atids.setter
    def atids(self, new_atids):
        if len(new_atids) != len(self):
            raise ValueError("Length mismatch for atids.")
        for i, aid in enumerate(new_atids):
            self[i].atid = aid

    @property
    def names(self):
        return [atom.name for atom in self]

    @names.setter
    def names(self, new_names):
        if len(new_names) != len(self):
            raise ValueError("Length mismatch for names.")
        for i, name in enumerate(new_names):
            self[i].name = name

    @property
    def alt_locs(self):
        return [atom.alt_loc for atom in self]

    @alt_locs.setter
    def alt_locs(self, new_alt_locs):
        if len(new_alt_locs) != len(self):
            raise ValueError("Length mismatch for alt_locs.")
        for i, alt in enumerate(new_alt_locs):
            self[i].alt_loc = alt

    @property
    def resnames(self):
        return [atom.resname for atom in self]

    @resnames.setter
    def resnames(self, new_resnames):
        if len(new_resnames) != len(self):
            raise ValueError("Length mismatch for resnames.")
        for i, rn in enumerate(new_resnames):
            self[i].resname = rn

    @property
    def chids(self):
        return [atom.chid for atom in self]

    @chids.setter
    def chids(self, new_chids):
        if len(new_chids) != len(self):
            raise ValueError("Length mismatch for chids.")
        for i, cid in enumerate(new_chids):
            self[i].chid = cid

    @property
    def resids(self):
        return [atom.resid for atom in self]

    @resids.setter
    def resids(self, new_resids):
        if len(new_resids) != len(self):
            raise ValueError("Length mismatch for resids.")
        for i, rid in enumerate(new_resids):
            self[i].resid = rid

    @property
    def icodes(self):
        return [atom.icode for atom in self]

    @icodes.setter
    def icodes(self, new_icodes):
        if len(new_icodes) != len(self):
            raise ValueError("Length mismatch for icodes.")
        for i, code in enumerate(new_icodes):
            self[i].icode = code

    @property
    def xs(self):
        return [atom.x for atom in self]

    @xs.setter
    def xs(self, new_xs):
        if len(new_xs) != len(self):
            raise ValueError("Length mismatch for x coordinates.")
        for i, x_val in enumerate(new_xs):
            self[i].x = x_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def ys(self):
        return [atom.y for atom in self]

    @ys.setter
    def ys(self, new_ys):
        if len(new_ys) != len(self):
            raise ValueError("Length mismatch for y coordinates.")
        for i, y_val in enumerate(new_ys):
            self[i].y = y_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def zs(self):
        return [atom.z for atom in self]

    @zs.setter
    def zs(self, new_zs):
        if len(new_zs) != len(self):
            raise ValueError("Length mismatch for z coordinates.")
        for i, z_val in enumerate(new_zs):
            self[i].z = z_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def occupancies(self):
        return [atom.occupancy for atom in self]

    @occupancies.setter
    def occupancies(self, new_occ):
        if len(new_occ) != len(self):
            raise ValueError("Length mismatch for occupancies.")
        for i, occ in enumerate(new_occ):
            self[i].occupancy = occ

    @property
    def bfactors(self):
        return [atom.bfactor for atom in self]

    @bfactors.setter
    def bfactors(self, new_bfactors):
        if len(new_bfactors) != len(self):
            raise ValueError("Length mismatch for bfactors.")
        for i, bf in enumerate(new_bfactors):
            self[i].bfactor = bf

    @property
    def segids(self):
        return [atom.segid for atom in self]

    @segids.setter
    def segids(self, new_segids):
        if len(new_segids) != len(self):
            raise ValueError("Length mismatch for segids.")
        for i, seg in enumerate(new_segids):
            self[i].segid = seg

    @property
    def elements(self):
        return [atom.element for atom in self]

    @elements.setter
    def elements(self, new_elements):
        if len(new_elements) != len(self):
            raise ValueError("Length mismatch for elements.")
        for i, elem in enumerate(new_elements):
            self[i].element = elem

    @property
    def charges(self):
        return [atom.charge for atom in self]

    @charges.setter
    def charges(self, new_charges):
        if len(new_charges) != len(self):
            raise ValueError("Length mismatch for charges.")
        for i, ch in enumerate(new_charges):
            self[i].charge = ch

    @property
    def vecs(self):
        return [atom.vec for atom in self]

    @vecs.setter
    def vecs(self, new_vecs):
        if len(new_vecs) != len(self):
            raise ValueError("Length mismatch for vectors.")
        for i, nvec in enumerate(new_vecs):
            self[i].vec = nvec

    @property
    def residues(self):
        """Group atoms by residue and return an AtomListCollection."""
        new_residue = AtomList()
        residues = []
        resid = self.resids[0]
        for atom in self:
            if atom.resid != resid:
                residues.append(new_residue)
                new_residue = AtomList()
                resid = atom.resid
            new_residue.append(atom)
        if new_residue:
            residues.append(new_residue)
        return AtomListCollection(residues)

    @property
    def chains(self):
        """Group atoms by chain identifier and return an AtomListCollection."""
        new_chain = AtomList()
        chains = []
        chid = self.chids[0]
        for atom in self:
            if atom.chid != chid:
                chains.append(new_chain)
                new_chain = AtomList()
                chid = atom.chid
            new_chain.append(atom)
        if new_chain:
            chains.append(new_chain)
        return AtomListCollection(chains)

    @property
    def segments(self):
        """Group atoms by segid identifier and return an AtomListCollection."""
        new_segment = AtomList()
        segment = []
        segid = self.segids[0]
        for atom in self:
            if atom.segid != segid:
                segment.append(new_segment)
                new_segment = AtomList()
                segid = atom.segid
            new_segment.append(atom)
        if new_segment:
            segment.append(new_segment)
        return AtomListCollection(segment)

    def renum(self):
        """Renumber atom IDs starting from 0."""
        self.atids = list(range(len(self)))

    def sort(self, key=None, reverse=False):
        """Sort the AtomList in place."""
        if key is None:
            key = lambda atom: (sort_uld(atom.chid), atom.resid, atom.icode, atom.atid)
        super().sort(key=key, reverse=reverse)

    def mask(self, mask_vals, mode="name"):
        """Return a new AtomList with atoms matching the given mask."""
        if isinstance(mask_vals, str):
            mask_vals = {mask_vals}
        if mode not in self.key_funcs:
            raise ValueError("Invalid mode '{}'.".format(mode))
        mask_vals = set(mask_vals)
        return AtomList([atom for atom in self if self.key_funcs[mode](atom) in mask_vals])

    def mask_out(self, mask_vals, mode="name"):
        """Return a new AtomList with atoms not matching the given mask."""
        if isinstance(mask_vals, str):
            mask_vals = {mask_vals}
        if mode not in self.key_funcs:
            raise ValueError("Invalid mode '{}'.".format(mode))
        mask_vals = set(mask_vals)
        return AtomList([atom for atom in self if self.key_funcs[mode](atom) not in mask_vals])

    def renumber(self):
        """Renumber atoms starting from 1."""
        new_atids = [atid % 99999 for atid in range(1, len(self) + 1)]
        self.atids = new_atids

    def remove_atoms(self, atoms_to_remove):
        """Remove specified atoms from the AtomList."""
        removal_set = set(atoms_to_remove)
        for atom in list(self):
            if atom in removal_set:
                self.remove(atom)

    def read_pdb(self, in_pdb):
        """Read a PDB file and populate the AtomList."""
        with open(in_pdb, "r", encoding="utf-8") as file:
            for line in file:
                record_type = line[0:6].strip()
                if record_type == "MODEL":
                    try:
                        current_model = int(line[10:14].strip())
                    except ValueError:
                        current_model = 1
                elif record_type in ("ATOM", "HETATM"):
                    try:
                        atom = Atom.from_pdb_line(line)
                        self.append(atom)
                    except Exception as e:
                        print("Error parsing line: {} -> {}".format(line.strip(), e))
                elif record_type == "ENDMDL":
                    current_model = 1

    def write_pdb(self, out_pdb, append=False):
        """Write the AtomList to a PDB file."""
        mode = "a" if append else "w"
        with open(out_pdb, mode, encoding="utf-8") as f:
            for atom in self:
                f.write(atom.to_pdb_line() + "\n")

    def write_ndx(self, filename, header="[ group ]", append=False, wrap=15): 
        # TODO: works for now but need a fix for very large systems
        """Write the atom IDs to a GROMACS .ndx file."""
        mode = "a" if append else "w"
        # idx_0 = self.atids[0]
        # num_idx = len(self.atids)
        # atids = [str(idx_0 + idx) for idx in range(num_idx)]
        atids = [str(atid) for atid in self.atids]
        if len(self.atids) > 99999:
            atids = [str(idx + 1) for idx in range(len(self.atids))]
        else:
            atids = [str(atid) for atid in self.atids]
        with open(filename, mode, encoding="utf-8") as f:
            f.write(header + "\n")
            for i in range(0, len(self), wrap):
                line = " ".join(atids[i : i + wrap])
                f.write(line + "\n")
            f.write("\n")


class AtomListCollection(list):
    """A collection of AtomList objects."""
    @property
    def records(self) -> List[List]:
        return [alist.records for alist in self]

    @property
    def atids(self) -> List[List]:
        return [alist.atids for alist in self]

    @property
    def names(self) -> List[List]:
        return [alist.names for alist in self]

    @property
    def alt_locs(self) -> List[List]:
        return [alist.alt_locs for alist in self]

    @property
    def resnames(self) -> List[List]:
        return [alist.resnames for alist in self]

    @property
    def chids(self) -> List[List]:
        return [alist.chids for alist in self]

    @property
    def resids(self) -> List[List]:
        return [alist.resids for alist in self]

    @property
    def icodes(self) -> List[List]:
        return [alist.icodes for alist in self]

    @property
    def xs(self) -> List[List]:
        return [alist.xs for alist in self]

    @property
    def ys(self) -> List[List]:
        return [alist.ys for alist in self]

    @property
    def zs(self) -> List[List]:
        return [alist.zs for alist in self]

    @property
    def occupancies(self) -> List[List]:
        return [alist.occupancies for alist in self]

    @property
    def bfactors(self) -> List[List]:
        return [alist.bfactors for alist in self]

    @property
    def segids(self) -> List[List]:
        return [alist.segids for alist in self]

    @property
    def elements(self) -> List[List]:
        return [alist.elements for alist in self]

    @property
    def charges(self) -> List[List]:
        return [alist.charges for alist in self]

    @property
    def vecs(self) -> List[List]:
        return [alist.vecs for alist in self]


class Residue:
    """Represents a residue containing Atom objects."""
    def __init__(self, resname, resid, icode):
        self.resname = resname
        self.resid = resid
        self.icode = icode
        self._atoms = AtomList()

    def add_atom(self, atom):
        self._atoms.append(atom)

    @property
    def atoms(self):
        return self._atoms

    def __iter__(self):
        return iter(self._atoms)

    def __repr__(self):
        return "<Residue {} {}{} with {} atom(s)>".format(
            self.resname, self.resid, self.icode, len(self._atoms)
        )


class Chain:
    """Represents a chain containing multiple residues."""
    def __init__(self, chid):
        self.chid = chid
        self.residues = {}

    def add_atom(self, atom):
        key = (atom.resid, atom.icode)
        if key not in self.residues:
            self.residues[key] = Residue(atom.resname, atom.resid, atom.icode)
        self.residues[key].add_atom(atom)

    @property
    def atoms(self):
        all_atoms = []
        for residue in self.residues.values():
            all_atoms.extend(residue.atoms)
        return AtomList(all_atoms)

    def __iter__(self):
        return iter(self.residues.values())

    def __repr__(self):
        return "<Chain {} with {} residue(s)>".format(self.chid, len(self.residues))


class Model:
    """Represents a model containing multiple chains."""
    def __init__(self, modid):
        self.modid = modid
        self.chains = {}

    def __iter__(self):
        return iter(self.chains.values())

    def __repr__(self):
        return "<Model {} with {} chain(s)>".format(self.modid, len(self.chains))

    def add_atom(self, atom):
        chid = atom.chid if atom.chid else " "
        if chid not in self.chains:
            self.chains[chid] = Chain(chid)
        self.chains[chid].add_atom(atom)

    @property
    def atoms(self):
        all_atoms = []
        for chain in self.chains.values():
            all_atoms.extend(chain.atoms)
        return AtomList(all_atoms)

    def select_chains(self, chids):
        return [chain for chid, chain in self.chains.items() if chid in chids]


class System:
    """Represents an entire system, potentially with multiple models."""
    def __init__(self):
        self.models = {}

    def __iter__(self):
        return iter(self.models.values())

    def __repr__(self):
        return "<System with {} model(s)>".format(len(self.models))

    def add_atom(self, atom, modid=1):
        if modid not in self.models:
            self.models[modid] = Model(modid)
        self.models[modid].add_atom(atom)

    def add_atoms(self, atoms, modid=1):
        if modid not in self.models:
            self.models[modid] = Model(modid)
        for atom in atoms:
            self.models[modid].add_atom(atom)

    @property
    def atoms(self):
        all_atoms = []
        for model in self.models.values():
            all_atoms.extend(model.atoms)
        return AtomList(all_atoms)

    def residues(self):
        for model in self.models.values():
            for chain in sorted(model.chains.values(), key=lambda c: c.chid):
                yield from sorted(chain.residues.values(), key=lambda r: (r.resid, r.icode))

    def chains(self):
        for model in self.models.values():
            yield from sorted(model.chains.values(), key=lambda c: c.chid)

    def write_pdb(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            sorted_modids = sorted(self.models.keys())
            multiple_models = len(sorted_modids) > 1
            for modid in sorted_modids:
                model = self.models[modid]
                if multiple_models:
                    f.write("MODEL     %d\n" % modid)
                for chain in sorted(model.chains.values(), key=lambda c: c.chid):
                    for residue in sorted(chain.residues.values(), key=lambda r: (r.resid, r.icode)):
                        for atom in residue.atoms:
                            f.write(atom.to_pdb_line() + "\n")
                if multiple_models:
                    f.write("ENDMDL\n")
            f.write("END\n")


class PDBParser:
    """Parses a PDB file and builds a System object."""
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file

    def parse(self):
        system = System()
        current_model = 1
        with open(self.pdb_file, "r", encoding="utf-8") as file:
            for line in file:
                record_type = line[0:6].strip()
                if record_type == "MODEL":
                    try:
                        current_model = int(line[10:14].strip())
                    except ValueError:
                        current_model = 1
                elif record_type in ("ATOM", "HETATM"):
                    try:
                        atom = Atom.from_pdb_line(line)
                        system.add_atom(atom, modid=current_model)
                    except Exception as e:
                        print("Error parsing line: {} -> {}".format(line.strip(), e))
                elif record_type == "ENDMDL":
                    current_model = 1
        return system


###################################
## Higher Level Functions ##
###################################

def pdb2system(pdb_path) -> System:
    """Parse a PDB file into a System object."""
    parser = PDBParser(pdb_path)
    return parser.parse()


def pdb2atomlist(pdb_path) -> AtomList:
    """Read a PDB file and return an AtomList of its atoms."""
    atoms = AtomList()
    atoms.read_pdb(pdb_path)
    return atoms


def sort_chains_atoms(atoms):
    """Sort an AtomList and renumber atom IDs."""
    atoms.sort()
    new_atids = [atid % 99999 for atid in range(1, len(atoms) + 1)]
    atoms.atids = new_atids


def rename_chains_for_gromacs(atoms):
    """
    Rename chains in an AtomList in a predefined order: uppercase, then lowercase, then digits.
    """
    import string
    new_chids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    curr_chid = atoms[0].chid
    counter = 0
    for atom in atoms:
        if atom.chid != curr_chid:
            curr_chid = atom.chid
            counter += 1
        atom.chid = new_chids[counter]


def sort_pdb(in_pdb, out_pdb):
    """Sort a PDB file and save the result."""
    atoms = pdb2atomlist(in_pdb)
    sort_chains_atoms(atoms)
    rename_chains_for_gromacs(atoms)
    atoms.write_pdb(out_pdb)
    print("Chains and atoms sorted, renamed and saved to %s" % out_pdb)


def clean_pdb(in_pdb, out_pdb, add_missing_atoms=False, add_hydrogens=False, pH=7.0):
    """Clean a PDB file using PDBFixer via OpenMM."""
    try:
        from pdbfixer.pdbfixer import PDBFixer  # pylint: disable=import-outside-toplevel
        from openmm.app import PDBFile         # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("PDBFixer or OpenMM not available") from e
    print("Processing %s" % in_pdb, file=sys.stderr)
    pdb = PDBFixer(filename=str(in_pdb))
    print("Removing heterogens and checking for missing residues...", file=sys.stderr)
    pdb.removeHeterogens(False)
    pdb.findMissingResidues()
    print("Replacing non-standard residues...", file=sys.stderr)
    pdb.findNonstandardResidues()
    pdb.replaceNonstandardResidues()
    if add_missing_atoms:
        print("Adding missing atoms...", file=sys.stderr)
        pdb.findMissingAtoms()
        pdb.addMissingAtoms()
    if add_hydrogens:
        print("Adding missing hydrogens...", file=sys.stderr)
        pdb.addMissingHydrogens(pH)
    topology = pdb.topology
    positions = pdb.positions
    with open(out_pdb, "w", encoding="utf-8") as outfile:
        PDBFile.writeFile(topology, positions, outfile)
    print("Written cleaned PDB to %s" % out_pdb, file=sys.stderr)


def rename_chain_in_pdb(in_pdb, new_chain_id):
    """Rename all chain identifiers in a PDB file."""
    atoms = pdb2atomlist(in_pdb)
    atoms.chids = [new_chain_id for _ in atoms]
    atoms.write_pdb(in_pdb)


def rename_chain_and_histidines_in_pdb(in_pdb, new_chain_id):
    """Rename chain identifiers and update histidine names in a PDB file."""
    atoms = pdb2atomlist(in_pdb)
    atoms.chids = [new_chain_id for _ in atoms]
    for atom in atoms:
        if atom.resname == "HSD":
            atom.resname = "HIS"
        if atom.resname == "HSE":
            atom.resname = "HIE"
    atoms.write_pdb(in_pdb)


def write_ndx(atoms, fpath="system.ndx", backbone_atoms=("CA", "P", "C1'")):
    """Write a GROMACS index file based on an AtomList."""
    # Note: The following uses a test PDB file; adjust as needed.
    in_pdb = "test.pdb"
    atoms = pdb2atomlist(in_pdb)
    atoms.write_ndx(fpath, header="[ System ]", append=False, wrap=15)
    backbone = atoms.mask(backbone_atoms, mode="name")
    backbone.write_ndx(fpath, header="[ Backbone ]", append=True, wrap=15)
    chids = sorted(set(atoms.chids))
    for chid in chids:
        selected_atoms = atoms.mask(chid, mode="chid")
        selected_atoms.write_ndx(fpath, header="[ chain_%s ]" % chid, append=True, wrap=15)
        print("Written index file to %s" % fpath, file=sys.stderr)


def update_bfactors(in_pdb, out_pdb, bfactors):
    """(Incomplete) Update the B-factors in a PDB file.
    Implementation is missing; please provide definitions for read_b_factors and update_pdb_b_factors.
    """
    # Undefined variables; implementation needed.
    # b_factors = read_b_factors(b_factor_file)
    # update_pdb_b_factors(pdb_file, b_factors, output_file)
    print("update_bfactors: Function not implemented.")


def sort_uld(alist):
    """Sort a list of characters: uppercase, then lowercase, then digits."""
    return sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))


def label_segments(in_pdb, out_pdb,):
    """Label segments based on something"""
    pass


AA_CODE_CONVERTER = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
