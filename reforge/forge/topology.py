#!/usr/bin/env python3
"""Topology Module

Description:
    This module provides classes and functions to construct a coarse-grained
    topology from force field parameters. It defines the Topology class along with
    a helper BondList class. Methods are provided to process atoms, bonds, and
    connectivity information, and to generate topology files for coarse-grained
    simulations.

Usage Example:
    >>> from topology import Topology
    >>> from reforge.forcefield import NucleicForceField
    >>> ff = NucleicForceField(...)  # Initialize the force field instance
    >>> topo = Topology(ff, sequence=['A', 'T', 'G', 'C'])
    >>> topo.from_sequence(['A', 'T', 'G', 'C'])
    >>> topo.write_to_itp('output.itp')

Requirements:
    - Python 3.x
    - NumPy
    - reForge utilities and force field modules

Author: DY
Date: YYYY-MM-DD
"""

import logging
from pathlib import Path
import numpy as np
from . import itpio

############################################################
# Helper class for working with bonds
############################################################

class BondList(list):
    """BondList Class

    Description:
        A helper subclass of the built-in list for storing bond information.
        Each element represents a bond in the form [connectivity, parameters, comment].

    Attributes
    ----------
    (inherited from list)
        Holds individual bond representations.

    Usage Example:
        >>> bonds = BondList([['C1-C2', [1.0, 1.5], 'res1 bead1'],
        ...                   ['C2-O1', [1.1, 1.6], 'res2 bead2']])
        >>> print(bonds.conns)
        ['C1-C2', 'C2-O1']
        >>> bonds.conns = ['C1-C2_mod', 'C2-O1_mod']
        >>> print(bonds.conns)
        ['C1-C2_mod', 'C2-O1_mod']
    """

    def __add__(self, other):
        """Implement addition of two BondList objects.

        Returns
        -------
        BondList
            A new BondList containing the bonds from self and other.
        """
        return BondList(super().__add__(other))

    @property
    def conns(self):
        """Get connectivity values from each bond.

        Returns
        -------
        list
            List of connectivity values (index 0 of each bond).
        """
        return [bond[0] for bond in self]

    @conns.setter
    def conns(self, new_conns):
        """Set new connectivity values for each bond.

        Parameters
        ----------
        new_conns : iterable
            A list-like object of new connectivity values. Must match the length of the BondList.
        """
        if len(new_conns) != len(self):
            raise ValueError("Length of new connectivity list must match the number of bonds")
        for i, new_conn in enumerate(new_conns):
            bond = list(self[i])
            bond[0] = new_conn
            self[i] = bond

    @property
    def params(self):
        """Get parameters from each bond.

        Returns
        -------
        list
            List of parameter values (index 1 of each bond).
        """
        return [bond[1] for bond in self]

    @params.setter
    def params(self, new_params):
        """Set new parameter values for each bond.

        Parameters
        ----------
        new_params : iterable
            A list-like object of new parameter values. Must match the length of the BondList.
        """
        if len(new_params) != len(self):
            raise ValueError("Length of new parameters list must match the number of bonds")
        for i, new_param in enumerate(new_params):
            bond = list(self[i])
            bond[1] = new_param
            self[i] = bond

    @property
    def comms(self):
        """Get comments from each bond.

        Returns
        -------
        list
            List of comments (index 2 of each bond).
        """
        return [bond[2] for bond in self]

    @comms.setter
    def comms(self, new_comms):
        """Set new comments for each bond.

        Parameters
        ----------
        new_comms : iterable
            A list-like object of new comment values. Must match the length of the BondList.
        """
        if len(new_comms) != len(self):
            raise ValueError("Length of new comments list must match the number of bonds")
        for i, new_comm in enumerate(new_comms):
            bond = list(self[i])
            bond[2] = new_comm
            self[i] = bond

    @property
    def measures(self):
        """Get measure values from each bond.

        Returns
        -------
        list
            List of measures extracted from the second element of the bond parameters.
        """
        return [bond[1][1] for bond in self]

    @measures.setter
    def measures(self, new_measures):
        """Set new measure values for each bond.

        Parameters
        ----------
        new_measures : iterable
            A list-like object of new measure values. Must match the length of the BondList.
        """
        if len(new_measures) != len(self):
            raise ValueError("Length of new measures list must match the number of bonds")
        for i, new_measure in enumerate(new_measures):
            bond = list(self[i])
            param = list(bond[1])
            param[1] = new_measure
            bond[1] = param
            self[i] = bond

    def categorize(self):
        """Categorize bonds based on their comments.

        Description:
            Uses the stripped comment (index 2) as a key to group bonds into a dictionary.

        Returns
        -------
        dict
            A dictionary mapping each unique comment to a BondList of bonds.
        """
        keys = sorted({comm.strip() for comm in self.comms})
        adict = {key: BondList() for key in keys}
        for bond in self:
            key = bond[2].strip()
            adict[key].append(bond)
        return adict

    def filter(self, condition, bycomm=True):
        """Filter bonds based on a provided condition.

        Description:
            Selects bonds for which the condition (a callable) returns True.
            By default, the condition is applied to the comment field.

        Parameters
        ----------
        condition : callable
            A function that takes a bond (or its comment) as input and returns True if the bond should be included.
        bycomm : bool, optional
            If True, the condition is applied to the comment (default is True).

        Returns
        -------
        BondList
            A new BondList containing the bonds that meet the condition.
        """
        if bycomm:
            return BondList([bond for bond in self if condition(bond[2])])
        return BondList([bond for bond in self if condition(bond)])

############################################################
# Topology Class
############################################################

class Topology:
    """Topology Class

    Description:
        Constructs a coarse-grained topology from force field parameters.
        Provides methods for processing atoms, bonds, and connectivity, and for
        generating a topology file for coarse-grained simulations.

    Attributes
    ----------
    ff : object
        Force field instance.
    sequence : list
        List of residue names.
    name : str
        Molecule name.
    nrexcl : int
        Exclusion parameter.
    atoms : list
        List of atom records.
    bonds, angles, dihs, cons, excls, pairs, vs3s, posres, elnet : BondList
        BondList instances for various bonded interactions.
    blist : list
        List containing all bond-type BondLists.
    secstruct : list
        Secondary structure as a list of characters.
    natoms : int
        Total number of atoms.
    """

    def __init__(self, forcefield=None, sequence: list | None = None, secstruct: list | None = None, **kwargs) -> None:
        """Initialize a Topology instance.

        Description:
            Initialize a topology with force field parameters, sequence, and optional secondary
            structure. Atom records and bond lists are initialized here.

        Parameters
        ----------
        forcefield : object, optional
            An instance of the nucleic force field class. Can be None when loading from ITP.
        sequence : list, optional
            List of residue names.
        secstruct : list, optional
            Secondary structure as a list of characters. If not provided, defaults to 'F' for each residue.
        **kwargs :
            Additional keyword arguments. Recognized options include:
                molname : str
                    Molecule name (default: "molecule").
                nrexcl : int
                    Exclusion parameter (default: 1).
        """
        molname = kwargs.pop("molname", "molecule")
        nrexcl = kwargs.pop("nrexcl", 1)
        self.ff = forcefield
        self.sequence = sequence if sequence is not None else []
        self.name = molname
        self.nrexcl = nrexcl
        self.atoms: list = []
        self.bonds = BondList()
        self.angles = BondList()
        self.dihs = BondList()
        self.cons = BondList()
        self.excls = BondList()
        self.pairs = BondList()
        self.vs3s = BondList()
        self.posres = BondList()
        self.elnet = BondList()
        self.mapping: list = []
        self.natoms = len(self.atoms)
        self.blist = [self.bonds, self.angles, self.dihs, self.cons, self.excls, self.pairs, self.vs3s]
        self.secstruct = secstruct if secstruct is not None else ["F"] * len(self.sequence) if self.sequence else []

    def __iadd__(self, other) -> "Topology":
        """Implement in-place addition of another Topology instance.

        Description:
            Merges another Topology into this one by updating atom numbers and connectivity.

        Parameters
        ----------
        other : Topology
            Another Topology instance to merge with.

        Returns
        -------
        Topology
            The merged topology (self).
        """
        def update_atom(atom, atom_shift, residue_shift):
            atom[0] += atom_shift  # Update atom id
            atom[2] += residue_shift  # Update residue id
            atom[5] += atom_shift  # Update charge group number
            return atom

        def update_bond(bond, atom_shift):
            conn = bond[0]
            conn = [idx + atom_shift for idx in conn]
            return [conn, bond[1], bond[2]]

        atom_shift = self.natoms
        residue_shift = len(self.sequence)
        new_atoms = [update_atom(atom, atom_shift, residue_shift) for atom in other.atoms]
        self.atoms.extend(new_atoms)
        for self_attrib, other_attrib in zip(self.blist, other.blist):
            updated_bonds = [update_bond(bond, atom_shift) for bond in other_attrib]
            self_attrib.extend(updated_bonds)
        return self

    def __add__(self, other) -> "Topology":
        """Implement addition of two Topology objects.

        Description:
            Returns a new Topology that is the merger of self and other.

        Parameters
        ----------
        other : Topology
            Another Topology instance.

        Returns
        -------
        Topology
            A new Topology instance resulting from the merge.
        """
        new_top = self
        new_top += other
        return new_top

    def lines(self) -> list:
        """Generate the topology file as a list of lines.

        Returns
        -------
        list
            A list of strings, each representing a line in the topology file.
        """
        forcefield_name = self.ff.name if self.ff is not None else ""
        lines = itpio.format_header(molname=self.name, forcefield=forcefield_name, arguments="")
        lines += itpio.format_sequence_section(self.sequence, self.secstruct)
        lines += itpio.format_moleculetype_section(molname=self.name, nrexcl=self.nrexcl)
        lines += itpio.format_atoms_section(self.atoms)
        lines += itpio.format_bonded_section("bonds", self.bonds)
        lines += itpio.format_bonded_section("angles", self.angles)
        lines += itpio.format_bonded_section("dihedrals", self.dihs)
        lines += itpio.format_bonded_section("constraints", self.cons)
        lines += itpio.format_bonded_section("exclusions", self.excls)
        lines += itpio.format_bonded_section("pairs", self.pairs)
        lines += itpio.format_bonded_section("virtual_sites3", self.vs3s)
        lines += itpio.format_bonded_section("bonds", self.elnet)
        lines += itpio.format_posres_section(self.atoms)
        logging.info("Created coarsegrained topology")
        return lines

    def write_to_itp(self, filename: str):
        """Write the topology to an ITP file.

        Parameters
        ----------
        filename : str
            The output file path.
        """
        with open(filename, "w", encoding="utf-8") as file:
            for line in self.lines():
                file.write(line)


    def elastic_network(self, atoms, anames: list[str] | None = None, el: float = 0.5, eu: float = 1.1, ef: float = 500):
        """Construct an elastic network between selected atoms.

        Parameters
        ----------
        atoms : list
            List of PDB atom objects.
        anames : list of str, optional
            Atom names to include (default: ["BB1", "BB3"]).
        el : float, optional
            Lower distance cutoff.
        eu : float, optional
            Upper distance cutoff.
        ef : float, optional
            Force constant.
        """
        if anames is None:
            anames = ["BB1", "BB3"]
        def get_distance(v1, v2):
            return 0.1 * np.linalg.norm(np.array(v1) - np.array(v2)) 
        selected = [atom for atom in atoms if atom.name in anames]
        for a1 in selected:
            for a2 in selected:
                if a2.atid - a1.atid > 3:
                    v1 = a1.vec
                    v2 = a2.vec
                    d = get_distance(v1, v2)
                    if el < d < eu:
                        comment = f"{a1.resname}{a1.atid}-{a2.resname}{a2.atid}"
                        self.elnet.append([[a1.atid, a2.atid], [6, d, ef], comment])

    def from_sequence(self, sequence, secstruc=None):
        """Build topology from a given sequence.

        Parameters
        ----------
        sequence : list
            Nucleic acid sequence (list of residue names).
        secstruc : list, optional
            Secondary structure. If not provided, defaults to all 'F'.
        """
        self.sequence = sequence
        self.process_atoms()
        self.process_bb_bonds()
        self.process_sc_bonds()

    def from_chain(self, chain, secstruc=None):
        """Build topology from a chain instance.

        Parameters
        ----------
        chain : object
            Chain object containing residues.
        secstruc : list, optional
            Secondary structure.
        """
        sequence = [residue.resname for residue in chain]
        self.from_sequence(sequence, secstruc=secstruc)

    @classmethod
    def from_itp(cls, itp_file: Path | str, molname: str | None = None) -> "Topology":
        """Create a Topology instance from an ITP file.
        
        Parameters
        ----------
        itp_file : Path or str
            Path to the ITP file to read
        molname : str, optional
            Molecule name. If not provided, will be read from moleculetype section
            
        Returns
        -------
        Topology
            A new Topology instance with data from the ITP file
        """
        itp_file = Path(itp_file)
        if not itp_file.exists():
            raise FileNotFoundError(f"ITP file not found: {itp_file}")
        
        # Read moleculetype section manually to get name and nrexcl
        nrexcl = 1
        with itp_file.open("r") as f:
            in_moleculetype = False
            for line in f:
                if line.strip().startswith("[ moleculetype ]"):
                    in_moleculetype = True
                    continue
                if in_moleculetype:
                    if line.strip() and not line.strip().startswith(";"):
                        parts = line.split()
                        if len(parts) >= 2 and not molname:
                            molname = parts[0]
                            nrexcl = int(parts[1])
                        break
        
        if not molname:
            molname = "molecule"
        
        # Create empty topology
        topo = cls(forcefield=None, sequence=[], molname=molname, nrexcl=nrexcl)
        
        # Read atoms section manually (not handled well by read_itp)
        with itp_file.open("r") as f:
            in_atoms = False
            for line in f:
                if line.strip().startswith("[ atoms ]"):
                    in_atoms = True
                    continue
                if in_atoms:
                    if line.strip().startswith("["):
                        break
                    if line.strip() and not line.strip().startswith(";"):
                        parts = line.split(";")
                        data = parts[0].split()
                        comment = parts[1].strip() if len(parts) > 1 else ""
                        if data:
                            # Format: nr type resnr residue atom cgnr charge mass [comment]
                            atom = [
                                int(data[0]),      # atom id
                                data[1],           # type
                                int(data[2]),      # residue id
                                data[3],           # residue name
                                data[4],           # atom name
                                int(data[5]),      # charge group
                                float(data[6]),    # charge
                            ]
                            if len(data) > 7:
                                atom.append(float(data[7]))  # mass
                            if comment:
                                atom.append(comment)
                            topo.atoms.append(atom)
        
        # Use read_itp for bonded interactions
        itp_data = itpio.read_itp(str(itp_file))
        
        # Map section names to BondList attributes
        section_map = {
            "bonds": topo.bonds,
            "angles": topo.angles,
            "dihedrals": topo.dihs,
            "constraints": topo.cons,
            "exclusions": topo.excls,
            "pairs": topo.pairs,
            "virtual_sites3": topo.vs3s,
        }
        
        # Populate BondLists from itp_data
        for section_name, bondlist in section_map.items():
            if section_name in itp_data:
                for entry in itp_data[section_name]:
                    # Convert tuples to lists for mutability
                    conn = list(entry[0])
                    params = list(entry[1]) if entry[1] else []
                    comment = entry[2]
                    bondlist.append([conn, params, comment])
        
        topo.natoms = len(topo.atoms)
        logging.info(f"Loaded topology from {itp_file.name}: {topo.natoms} atoms")
        return topo

    @staticmethod
    def merge_topologies(topologies):
        """Merge multiple Topology instances into one.

        Parameters
        ----------
        topologies : list
            List of Topology objects.

        Returns
        -------
        Topology
            A merged Topology instance.
        """
        top = topologies.pop(0)
        if topologies:
            for new_top in topologies:
                top += new_top
        return top
