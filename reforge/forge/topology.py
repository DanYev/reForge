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
    bonds, angles, dihs, cons, excls, pairs, vs2s, vs3s, vs4s, vsn, posres, elnet : BondList
        BondList instances for various bonded interactions.
    blist : list
        List containing all bond-type BondLists.
    secstruct : list
        Secondary structure as a list of characters.
    natoms : int
        Total number of atoms.
    """

    def __init__(self, **kwargs) -> None:
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
        self.header = []
        self.molname = "molecule"
        self.nrexcl = 1
        self.atoms: list = []
        self.bonds = BondList()
        self.angles = BondList()
        self.dihs = BondList()
        self.cons = BondList()
        self.excls = BondList()
        self.pairs = BondList()
        self.vs2s = BondList()
        self.vs3s = BondList()
        self.vs4s = BondList()
        self.vsn = BondList()
        self.posres = BondList()
        self.elnet = BondList()
        self.natoms = len(self.atoms)
        self.blist = [self.bonds, self.angles, self.dihs, self.cons, self.excls, self.pairs, 
            self.vs2s, self.vs3s, self.vs4s, self.vsn, self.posres, self.elnet]

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
            atom = atom.copy()
            atom[0] += atom_shift  # Update atom id
            atom[2] += residue_shift  # Update residue id
            atom[5] += atom_shift  # Update charge group number
            return atom

        def update_bond(bond, atom_shift):
            bond = bond.copy()
            conn = bond[0]
            conn = [idx + atom_shift for idx in conn]
            return [conn, bond[1], bond[2]]

        # # Store initial lengths before merge for verification
        # initial_self_lengths = [len(lst) for lst in self.blist]
        
        atom_shift = self.natoms
        # Calculate residue_shift from actual max residue ID in atoms, not sequence length
        if self.atoms:
            residue_shift = max(atom[2] for atom in self.atoms)
        else:
            residue_shift = 0
        
        new_atoms = [update_atom(atom, atom_shift, residue_shift) for atom in other.atoms]
        self.atoms.extend(new_atoms)
        self.natoms = len(self.atoms)  # Update natoms after extending
        for self_attrib, other_attrib in zip(self.blist, other.blist):
            updated_bonds = [update_bond(bond, atom_shift) for bond in other_attrib]
            self_attrib.extend(updated_bonds)
        
        # # Verify that each bonded list has the correct length
        # self._check_iadd(other, initial_self_lengths)
        return self
    
    def _check_iadd(self, other, initial_self_lengths):
        """Check that bonded lists have correct lengths after addition.
        
        Parameters
        ----------
        other : Topology
            The other topology that was added to self.
        initial_self_lengths : list of int
            The initial lengths of self.blist before merge.
        """
        blist_names = ["bonds", "angles", "dihedrals", "constraints", "exclusions", 
                      "pairs", "vs2s", "vs3s", "vs4s", "vsn", "posres", "elnet"]
        
        for i, (self_list, other_list, self_init) in enumerate(zip(self.blist, other.blist, initial_self_lengths)):
            expected_length = self_init + len(other_list)
            actual_length = len(self_list)
            if actual_length != expected_length:
                raise ValueError(
                    f"Length mismatch in {blist_names[i]}: "
                    f"expected {expected_length} (self={self_init} + other={len(other_list)}), "
                    f"got {actual_length}"
                )

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
        lines = format_header(self.header)
        lines += format_moleculetype_section(molname=self.molname, nrexcl=self.nrexcl)
        lines += format_atoms_section(self.atoms)
        lines += format_bonded_section("bonds", self.bonds)
        lines += format_bonded_section("angles", self.angles)
        lines += format_bonded_section("dihedrals", self.dihs)
        lines += format_bonded_section("constraints", self.cons)
        lines += format_bonded_section("exclusions", self.excls)
        lines += format_bonded_section("pairs", self.pairs)
        lines += format_bonded_section("virtual_sites2", self.vs2s)
        lines += format_bonded_section("virtual_sites3", self.vs3s)
        lines += format_bonded_section("virtual_sites4", self.vs4s)
        lines += format_bonded_section("virtual_sitesn", self.vsn)
        lines += format_bonded_section("bonds", self.elnet)
        lines += format_posres_section(self.posres)
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

    @classmethod
    def from_itp(cls, itp_file: Path | str, **kwargs) -> "Topology":
        """Create a Topology instance from an ITP file.
        
        Parameters
        ----------
        itp_file : Path or str
            Path to the ITP file to read
        molname : str, optional
            Molecule name. If not provided, will be read from moleculetype section
        define : list[str], optional
            List of macro names that are defined. Controls which #ifdef/#ifndef blocks are included.
            If None, defaults to empty list (no macros defined).
            
        Returns
        -------
        Topology
            A new Topology instance with data from the ITP file
        """
        itp_file = Path(itp_file)
        if not itp_file.exists():
            raise FileNotFoundError(f"ITP file not found: {itp_file}")
        
        # Read ITP file
        itp_data = read_itp(str(itp_file), **kwargs)    
        
        # Create empty topology
        topo = cls()
        
        # Header 
        topo.header = itp_data.get("header", [])
        
        # Read moleculetype section to get name and nrexcl
        nrexcl = 1
        if "moleculetype" in itp_data:
            for entry in itp_data["moleculetype"]:
                if len(entry) >= 2:
                    topo.molname = entry[0]
                    topo.nrexcl = int(entry[1])
                    break

        # Read atoms section
        if "atoms" in itp_data:
            for entry in itp_data["atoms"]:
                # Format: nr type resnr residue atom cgnr charge [mass] [comment]
                if len(entry) >= 7:
                    atom = [
                        int(entry[0]),      # atom id
                        entry[1],           # type
                        int(entry[2]),      # residue id
                        entry[3],           # residue name
                        entry[4],           # atom name
                        int(entry[5]),      # charge group
                        float(entry[6]),    # charge
                    ]
                    if len(entry) > 7:
                        try:
                            atom.append(float(entry[7]))  # mass
                        except ValueError:
                            # It's a comment, not mass
                            atom.append("")
                            if entry[7]:
                                atom.append(entry[7])
                    if len(entry) > 8 and entry[8]:
                        # Comment in last position
                        if len(atom) == 7:
                            atom.append("")  # No mass
                        atom.append(entry[8])
                    topo.atoms.append(atom)
        
        # Map section names to BondList attributes
        section_map = {
            "bonds": topo.bonds,
            "angles": topo.angles,
            "dihedrals": topo.dihs,
            "constraints": topo.cons,
            "exclusions": topo.excls,
            "pairs": topo.pairs,
            "virtual_sites2": topo.vs2s,
            "virtual_sites3": topo.vs3s,
            "virtual_sites4": topo.vs4s,
            "virtual_sitesn": topo.vsn,
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
        
        # Handle position_restraints section (parsed as generic data)
        if "position_restraints" in itp_data:
            for entry in itp_data["position_restraints"]:
                # Format: atom_id func_type fc_x fc_y fc_z [comment]
                if len(entry) >= 5:
                    atom_id = int(entry[0])
                    func_type = int(entry[1])
                    # Store as connectivity=atom_id, parameters=[func_type, fc_x, fc_y, fc_z]
                    params = [func_type, entry[2], entry[3], entry[4]]
                    comment = entry[5] if len(entry) > 5 else ""
                    topo.posres.append([[atom_id], params, comment])
        
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

###################################
# ITP PARSING UTILITIES
###################################

def read_itp(fpath, define: list[str] = ["POSRES"]):
    """Read a Gromacs ITP file and organize its contents by section.

    Parameters
    ----------
    fpath : Path or str
        The path to the ITP file.
    define : list[str], optional
        List of macro names that are defined. Controls which #ifdef/#ifndef blocks are included.
        If None, defaults to empty list (no macros defined).

    Returns
    -------
    dict
        A dictionary where keys are section names and values are lists of entries.
        For bonded sections, each entry is [connectivity, parameters, comment].
        For other sections, each entry is a list of parsed line data.
        Header lines are stored in itp_data["header"].
        
    Notes
    -----
    - #ifdef blocks in the header are preserved
    - #ifdef MACRO: included if MACRO in define list, excluded otherwise
    - #ifndef MACRO: included if MACRO not in define list, excluded otherwise
    - #else flips the inclusion logic within a block
    """
    # Define which sections should be parsed as bonded interactions
    bonded_sections = {"bonds", "angles", "dihedrals", "constraints", 
                      "pairs", "exclusions", "virtual_sites2", "virtual_sites3", 
                      "virtual_sites4", "virtual_sitesn"}
    
    itp_data = {"header": []}
    current_section = None
    inside_ifdef = False
    ifdef_allowed = False  # Whether to include content from current preprocessor block
    
    with open(fpath, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            
            # Handle #ifdef and #ifndef blocks
            if stripped.startswith("#ifdef") or stripped.startswith("#ifndef"):
                if current_section is None:
                    # In header - keep it
                    itp_data["header"].append(line)
                else:
                    # Extract macro name from directive
                    parts = stripped.split()
                    macro_name = parts[1] if len(parts) > 1 else ""
                    
                    inside_ifdef = True
                    if stripped.startswith("#ifdef"):
                        # #ifdef MACRO: include if MACRO is defined
                        ifdef_allowed = macro_name in define
                    else:
                        # #ifndef MACRO: include if MACRO is NOT defined
                        ifdef_allowed = macro_name not in define
                continue
            
            # Handle #else
            if stripped.startswith("#else"):
                if current_section is None:
                    # In header - keep it
                    itp_data["header"].append(line)
                elif inside_ifdef:
                    # Flip the allowed state
                    ifdef_allowed = not ifdef_allowed
                continue
            
            # Handle #endif
            if stripped.startswith("#endif"):
                if current_section is None:
                    # In header - keep it
                    itp_data["header"].append(line)
                else:
                    inside_ifdef = False
                    ifdef_allowed = False
                continue
            
            # Skip content inside ignored ifdef blocks
            if inside_ifdef and not ifdef_allowed:
                continue
            
            # Handle #define and other preprocessor directives in header
            if current_section is None and stripped.startswith("#"):
                itp_data["header"].append(line)
                continue
            
            # Skip any other preprocessor directives in sections (shouldn't parse them as data)
            if current_section is not None and stripped.startswith("#"):
                continue
            
            # Store header lines (before first section)
            if current_section is None:
                if stripped.startswith("[") and stripped.endswith("]"):
                    # First section found
                    current_section = stripped[1:-1].strip()
                    itp_data[current_section] = []
                else:
                    # Store as header
                    itp_data["header"].append(line)
                continue
            
            # Skip empty lines
            if stripped == "":
                continue
            
            # Check for new section header
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1].strip()
                itp_data[current_section] = []
                continue
            
            # Skip comment lines within sections
            if stripped.startswith(";"):
                continue
            
            # Parse section content
            if current_section in bonded_sections:
                # Parse as bonded interaction
                connectivity, parameters, comment = line_to_bond(line, current_section)
                itp_data[current_section].append([connectivity, parameters, comment])
            else:
                # Parse as generic line data
                parts = line.split(";", 1)
                data = parts[0].split()
                comment = parts[1].strip() if len(parts) > 1 else ""
                if data:  # Only add non-empty data
                    itp_data[current_section].append(data + ([comment] if comment else []))
    
    return itp_data


def line_to_bond(line, section):
    """Parse a line from an ITP file and return connectivity, parameters, and comment.

    Parameters
    ----------
    line : str
        A line from the ITP file.
    section : str
        The section (e.g. 'bonds', 'angles', etc.).

    Returns
    -------
    tuple
        A tuple (connectivity, parameters, comment) where connectivity is a tuple of ints,
        parameters is a tuple of numbers (first as int, rest as floats), and comment is a string.
    """
    data, _, comment = line.partition(";")
    data = data.split()
    comment = comment.strip()
    if section == "bonds" or section == "constraints":
        connectivity = data[:2]
        parameters = data[2:]
    elif section == "virtual_sites2":
        connectivity = data[:3]  # virtual site + 2 constructing atoms
        parameters = data[3:]
    elif section == "angles":
        connectivity = data[:3]
        parameters = data[3:]
    elif section == "dihedrals" or section == "virtual_sites3":
        connectivity = data[:4]
        parameters = data[4:]
    elif section == "virtual_sites4":
        connectivity = data[:5]  # virtual site + 4 constructing atoms
        parameters = data[5:]
    elif section == "virtual_sitesn":
        # Format: vsite func_type at1 at2 ... atN
        # connectivity: vsite + all constructing atoms
        # parameters: just func_type (integer)
        if len(data) >= 2:
            vsite = [data[0]]
            func_type = [data[1]]
            constructing_atoms = data[2:]  # All remaining atoms
            connectivity = vsite + constructing_atoms
            parameters = func_type
        else:
            connectivity = data
            parameters = []
    else:
        connectivity = data
        parameters = []
    if parameters:
        parameters[0] = int(parameters[0])
        parameters[1:] = [float(i) for i in parameters[1:]]
    connectivity = tuple(int(i) for i in connectivity)
    parameters = tuple(parameters)
    return connectivity, parameters, comment


def bond_to_line(connectivity=None, parameters="", comment=""):
    """Format a bond entry into a string for a Gromacs ITP file.

    Parameters
    ----------
    connectivity : tuple, optional
        Connectivity indices.
    parameters : tuple, optional
        Bond parameters.
    comment : str, optional
        Optional comment.

    Returns
    -------
    str
        A formatted string representing the bond entry.
    """
    connectivity_str = "   ".join(f"{int(atom):5d}" for atom in connectivity)
    type_str = ""
    parameters_str = ""
    if parameters:
        type_str = f"{int(parameters[0]):2d}"
        parameters_str = "   ".join(f"{float(param):7.4f}" for param in parameters[1:])
    line = connectivity_str + "   " + type_str + "   " + parameters_str
    if comment:
        line += " ; " + comment
    line += "\n"
    return line

###################################
# ITP FORMATTING
###################################

def format_header(lines=None) -> list[str]:
    """Format the header section."""
    if lines is None:
        return []
    return lines


def format_moleculetype_section(molname="molecule", nrexcl=1) -> list[str]:
    """Format the moleculetype section."""
    lines = ["\n[ moleculetype ]\n"]
    lines.append("; Name         Exclusions\n")
    lines.append(f"{molname:<15s} {nrexcl:3d}\n")
    return lines


def format_atoms_section(atoms: list[tuple]) -> list[str]:
    """Format the atoms section for a Gromacs ITP file.

    Parameters
    ----------
    atoms : list[tuple]
        List of atom records.

    Returns
    -------
    list[str]
        A list of formatted lines.
    """
    lines = ["\n[ atoms ]\n"]
    for atom in atoms:
        atom = tuple(atom)
        if len(atom) == 9:
            # Format with mass + comment: nr type resnr residue atom cgnr charge mass comment
            if atom[8] and isinstance(atom[8], str) and atom[8].strip():
                line = "%5d %5s %5d %5s %5s %5d %7.4f %7.4f ; %s" % atom
            else:
                line = "%5d %5s %5d %5s %5s %5d %7.4f %7.4f" % atom[:8]
        elif len(atom) == 8:
            # Format with mass, no comment: nr type resnr residue atom cgnr charge mass
            if isinstance(atom[7], float):
                line = "%5d %5s %5d %5s %5s %5d %7.4f %7.4f" % atom
            else:
                # atom[7] is a comment string
                line = "%5d %5s %5d %5s %5s %5d %7.4f ; %s" % atom
        else:
            # Format with just 7 values: nr type resnr residue atom cgnr charge
            line = "%5d %5s %5d %5s %5s %5d %7.4f" % atom[:7]
        line += "\n"
        lines.append(line)
    return lines


def format_bonded_section(header: str, bonds: list[list]) -> list[str]:
    """Format a bonded section (e.g., bonds, angles) for a Gromacs ITP file.

    Parameters
    ----------
    header : str
        Section header.
    bonds : list[list]
        List of bond entries.

    Returns
    -------
    list[str]
        A list of formatted lines.
    """
    if not bonds:
        return []  # Don't write empty sections
    lines = [f"\n[ {header} ]\n"]
    
    # Special formatting for virtual_sitesn: all values as integers
    if header == "virtual_sitesn":
        for bond in bonds:
            connectivity, parameters, comment = bond
            # Format: vsite, func_type, then constructing atoms
            # conn[0] + params[0] + conn[1:] + ; comment
            vals = [connectivity[0]] + list(parameters) + list(connectivity[1:])
            line = "".join(f"{int(val):>8d}" for val in vals)
            if comment:
                line += " ; " + comment
            line += "\n"
            lines.append(line)
    else:
        for bond in bonds:
            line = bond_to_line(*bond)
            lines.append(line)
    return lines


def format_posres_section(posres: BondList) -> list[str]:
    """Format the position restraints section.

    Parameters
    ----------
    posres : BondList
        List of position restraint records. Each record has:
        - conn: [atom_id]
        - params: [func_type, fc_x, fc_y, fc_z]
        - comment: comment string

    Returns
    -------
    list[str]
        A list of formatted lines.
    """
    if not posres:
        return []
    
    lines = [
        "\n#ifdef POSRES\n",
        "\n#ifndef POSRES_FC\n",
        "\n#define POSRES_FC 1000\n",
        "\n#endif\n",
        " [ position_restraints ]\n",
    ]
    for restraint in posres:
        atom_id = restraint[0][0]  # conn is [atom_id]
        params = restraint[1]  # [func_type, fc_x, fc_y, fc_z]
        comment = restraint[2]
        
        if params:
            func_type = params[0]
            fc_x = params[1] if len(params) > 1 else "POSRES_FC"
            fc_y = params[2] if len(params) > 2 else "POSRES_FC"
            fc_z = params[3] if len(params) > 3 else "POSRES_FC"
        else:
            func_type = 1
            fc_x = fc_y = fc_z = "POSRES_FC"
        
        line = f"  {atom_id:5d}    {func_type}    {fc_x}    {fc_y}    {fc_z}"
        if comment:
            line += f"  ; {comment}"
        line += "\n"
        lines.append(line)
    
    lines.append("#endif\n")
    return lines


def write_itp(filename, lines):
    """Write a list of lines to an ITP file.

    Parameters
    ----------
    filename : str
        Output file path.
    lines : list[str]
        Lines to write.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line)