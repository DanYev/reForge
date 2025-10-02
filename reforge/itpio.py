"""IO for GROMACS topology .itp files.

Description:
    This module provides functions for reading, parsing, formatting, and writing
    GROMACS ITP files. It includes utilities to extract and format different sections
    (such as bonds, angles, atoms, etc.) and higher-level functions tailored for
    Martini RNA and ion topologies.

Requirements:
    - Python 3.x
    - Standard library modules: shutil, typing
    - (Optional) Additional modules for extended functionality.

Author: DY
Date: 2025-02-27
"""

import shutil as sh
from typing import List, Tuple

###################################
# Generic functions
###################################

def read_itp(filename):
    """Read a Gromacs ITP file and organize its contents by section.

    Parameters
    ----------
    filename : str
        The path to the ITP file.

    Returns
    -------
    dict
        A dictionary where keys are section names and values are lists of entries,
        each entry being a list of [connectivity, parameters, comment].
    """
    itp_data = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            # Skip comments and empty lines
            if line.strip() == "" or line.strip().startswith(";"):
                continue
            # Detect section headers; break long condition into multiple lines.
            if (line.startswith("[") and line.endswith("]\n")):
                tag = line.strip()[2:-2]
                itp_data[tag] = []
            else:
                connectivity, parameters, comment = line2bond(line, tag)
                itp_data[tag].append([connectivity, parameters, comment])
    return itp_data


def line2bond(line, tag):
    """Parse a line from an ITP file and return connectivity, parameters, and comment.

    Parameters
    ----------
    line : str
        A line from the ITP file.
    tag : str
        The section tag (e.g. 'bonds', 'angles', etc.).

    Returns
    -------
    tuple
        A tuple (connectivity, parameters, comment) where connectivity is a tuple of ints,
        parameters is a tuple of numbers (first as int, rest as floats), and comment is a string.
    """
    data, _, comment = line.partition(";")
    data = data.split()
    comment = comment.strip()
    if tag == "bonds" or tag == "constraints":
        connectivity = data[:2]
        parameters = data[2:]
    elif tag == "angles":
        connectivity = data[:3]
        parameters = data[3:]
    elif tag == "dihedrals":
        connectivity = data[:4]
        parameters = data[4:]
    elif tag == "virtual_sites3":
        connectivity = data[:4]
        parameters = data[4:]
    else:
        connectivity = data
        parameters = []
    if parameters:
        parameters[0] = int(parameters[0])
        parameters[1:] = [float(i) for i in parameters[1:]]
    connectivity = tuple(int(i) for i in connectivity)
    parameters = tuple(parameters)
    return connectivity, parameters, comment


def bond2line(connectivity=None, parameters="", comment=""):
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


def format_header(molname="molecule", forcefield="", arguments="", timestamp="") -> List[str]:
    """Format the header of the topology file.

    Parameters
    ----------
    molname : str, optional
        Molecule name. Default is "molecule".
    forcefield : str, optional
        Force field identifier.
    arguments : str, optional
        Parsed arguments with their values including defaults.
    timestamp : str, optional
        Timestamp when file was generated.

    Returns
    -------
    List[str]
        A list of header lines.
    """
    lines = [f'; MARTINI ({forcefield}) Coarse Grained topology file for "{molname}"\n']
    if timestamp:
        lines.append(f"; Generated on: {timestamp}\n")
    lines.append("; Created using the following options:\n")
    if arguments:
        lines.append(f"; {arguments}\n")
    lines.append("; " + "#" * 100 + "\n")
    return lines


def format_sequence_section(sequence, secstruct) -> List[str]:
    """Format the sequence section.

    Parameters
    ----------
    sequence : iterable
        Sequence characters.
    secstruct : iterable
        Secondary structure characters.

    Returns
    -------
    List[str]
        Formatted lines for the sequence section.
    """
    sequence_str = "".join(sequence)
    secstruct_str = "".join(secstruct)
    lines = ["; Sequence:\n"]
    lines.append(f"; {sequence_str}\n")
    lines.append("; Secondary Structure:\n")
    lines.append(f"; {secstruct_str}\n")
    return lines


def format_moleculetype_section(molname="molecule", nrexcl=1) -> List[str]:
    """Format the moleculetype section.

    Parameters
    ----------
    molname : str, optional
        Molecule name. Default is "molecule".
    nrexcl : int, optional
        Number of exclusions. Default is 1.

    Returns
    -------
    List[str]
        Formatted lines for the moleculetype section.
    """
    lines = ["\n[ moleculetype ]\n"]
    lines.append("; Name         Exclusions\n")
    lines.append(f"{molname:<15s} {nrexcl:3d}\n")
    return lines


def format_atoms_section(atoms: List[Tuple]) -> List[str]:
    """Format the atoms section for a Gromacs ITP file.

    Parameters
    ----------
    atoms : List[Tuple]
        List of atom records.

    Returns
    -------
    List[str]
        A list of formatted lines.
    """
    lines = ["\n[ atoms ]\n"]
    for atom in atoms:
        atom = tuple(atom)
        if len(atom) == 9:
            # Format with 8 values + comment
            if atom[8] and atom[8].strip():  # If there's a non-empty comment
                line = "%5d %5s %5d %5s %5s %5d %7.4f %7.4f ; %s" % atom
            else:
                line = "%5d %5s %5d %5s %5s %5d %7.4f %7.4f" % atom[:8]
        else:
            # Format with 7 values + comment
            if len(atom) > 7 and atom[7] and atom[7].strip():  # If there's a non-empty comment
                line = "%5d %5s %5d %5s %5s %5d %7.4f ; %s" % atom
            else:
                line = "%5d %5s %5d %5s %5s %5d %7.4f" % atom[:7]
        line += "\n"
        lines.append(line)
    return lines


def format_bonded_section(header: str, bonds: List[List]) -> List[str]:
    """Format a bonded section (e.g., bonds, angles) for a Gromacs ITP file.

    Parameters
    ----------
    header : str
        Section header.
    bonds : List[List]
        List of bond entries.

    Returns
    -------
    List[str]
        A list of formatted lines.
    """
    lines = [f"\n[ {header} ]\n"]
    for bond in bonds:
        line = bond2line(*bond)
        lines.append(line)
    return lines


def format_posres_section(atoms: List[Tuple], posres_fc=500, 
                          selection: List[str] = None) -> List[str]:
    """Format the position restraints section.

    Parameters
    ----------
    atoms : List[Tuple]
        List of atom records.
    posres_fc : float, optional
        Force constant for restraints. Default is 500.
    selection : List[str], optional
        Atom names to select. Defaults to ["BB1", "BB2", "BB3", "SC1"] if not provided.

    Returns
    -------
    List[str]
        A list of formatted lines.
    """
    if selection is None:
        selection = ["BB1", "BB2", "BB3", "SC1"]
    lines = [
        "\n#ifdef POSRES\n",
        f"#define POSRES_FC {posres_fc:.2f}\n",
        " [ position_restraints ]\n",
    ]
    for atom in atoms:
        if atom[4] in selection:
            lines.append(f"  {atom[0]:5d}    1    POSRES_FC    POSRES_FC    POSRES_FC\n")
    lines.append("#endif")
    return lines


def write_itp(filename, lines):
    """Write a list of lines to an ITP file.

    Parameters
    ----------
    filename : str
        Output file path.
    lines : List[str]
        Lines to write.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line)


###################################
# HL functions for martini_rna
###################################

def make_in_terms(input_file, output_file, dict_of_names):
    """Generate a Martini ITP file using input terms and a dictionary of names.

    Parameters
    ----------
    input_file : str
        Path to the input ITP file.
    output_file : str
        Path to the output ITP file.
    dict_of_names : dict
        Dictionary mapping keys to desired names.
    """
    tag = None
    pairs = []

    def get_sigma(b1, b2):
        list_of_pairs_1 = {("TA4", "TU3"), ("TA5", "TU4"),
                           ("TG3", "TY2"), ("TG4", "TY3"), ("TG5", "TY4")}
        list_of_pairs_2 = {("TA4", "TU2"), ("TA4", "TU4"),
                           ("TA5", "TU3"), ("TG3", "TY3"), ("TG4", "TY2"),
                           ("TG4", "TY4"), ("TG5", "TY3")}
        list_of_pairs_3 = {("TA4", "TY3"), ("TA5", "TY4"), ("TA4", "TY2"),
                           ("TA4", "TY4"), ("TA5", "TY3"), ("TG3", "TU2"),
                           ("TG4", "TU3"), ("TG5", "TU4"), ("TG3", "TU3"),
                           ("TG4", "TU2"), ("TG4", "TU4"), ("TG5", "TU3")}
        if (b1, b2) in list_of_pairs_1 or (b2, b1) in list_of_pairs_1:
            sigma = "2.75000e-01"
        elif (b1, b2) in list_of_pairs_2 or (b2, b1) in list_of_pairs_2:
            sigma = "2.750000e-01"
        elif (b1, b2) in list_of_pairs_3 or (b2, b1) in list_of_pairs_3:
            sigma = "2.750000e-01"
        else:
            sigma = "3.300000e-01"
        return sigma

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("[ atomtypes ]\n")
        dict_of_vdw = {
            "TA1": ("3.250000e-01", "1.000000e-01"),
            "TA2": ("3.250000e-01", "1.000000e-01"),
            "TA3": ("3.250000e-01", "1.000000e-01"),
            "TA4": ("2.800000e-01", "1.368000e-01"),
            "TA5": ("2.800000e-01", "1.368000e-01"),
            "TA6": ("3.250000e-01", "1.000000e-01"),
            "TY1": ("3.250000e-01", "1.000000e-01"),
            "TY2": ("2.800000e-01", "1.000000e-01"),
            "TY3": ("2.800000e-01", "1.000000e-01"),
            "TY4": ("2.800000e-01", "1.000000e-01"),
            "TY5": ("3.250000e-01", "1.000000e-01"),
            "TG1": ("3.250000e-01", "1.000000e-01"),
            "TG2": ("3.250000e-01", "1.000000e-01"),
            "TG3": ("2.800000e-01", "1.368000e-01"),
            "TG4": ("2.800000e-01", "1.368000e-01"),
            "TG5": ("2.800000e-01", "1.368000e-01"),
            "TG6": ("3.250000e-01", "1.000000e-01"),
            "TG7": ("0.400000e-01", "1.368000e-01"),
            "TG8": ("3.250000e-01", "1.000000e-01"),
            "TU1": ("3.250000e-01", "1.000000e-01"),
            "TU2": ("2.800000e-01", "1.000000e-01"),
            "TU3": ("2.800000e-01", "1.368000e-01"),
            "TU4": ("2.800000e-01", "1.000000e-01"),
            "TU5": ("3.250000e-01", "1.000000e-01"),
            "TU6": ("0.400000e-01", "1.368000e-01"),
            "TU7": ("3.250000e-01", "1.000000e-01"),
        }
        for key in dict_of_names.keys():
            file.write(f"{key}  45.000  0.000  A  {dict_of_vdw[key][0]}  {dict_of_vdw[key][1]}\n")
        file.write("\n[ nonbond_params ]\n")

    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    with open(output_file, "a", encoding="utf-8") as file:
        for line in lines:
            if line.startswith(";") or len(line.split()) < 2:
                continue
            parts = line.split()
            atom_name_1 = parts[0].strip()
            atom_name_2 = parts[1].strip()
            if (atom_name_1 in dict_of_names.values() and 
                atom_name_2 in dict_of_names.values()):
                keys_1 = [key for key, value in dict_of_names.items() if value == atom_name_1]
                keys_2 = [key for key, value in dict_of_names.items() if value == atom_name_2]
                for key_1 in keys_1:
                    for key_2 in keys_2:
                        if (key_1, key_2) in pairs or (key_2, key_1) in pairs:
                            continue
                        pairs.append((key_1, key_2))
                        parts[0] = key_1
                        parts[1] = key_2
                        parts[3] = get_sigma(key_1, key_2)
                        file.write("  ".join(parts) + "\n")


def make_cross_terms(input_file, output_file, old_name, new_name):
    """Append cross-term entries to an ITP file by replacing an old name with a new one.

    Parameters
    ----------
    input_file : str
        Path to the input ITP file.
    output_file : str
        Path to the output ITP file.
    old_name : str
        The name to be replaced.
    new_name : str
        The replacement name.
    """
    switch = False
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    with open(output_file, "a", encoding="utf-8") as file:
        for line in lines:
            if line.startswith(";") or len(line.split()) < 2:
                continue
            if line.startswith("[ nonbond_params ]"):
                switch = True
            if switch:
                parts = line.split()
                atom_name_1 = parts[0].strip()
                atom_name_2 = parts[1].strip()
                if atom_name_1 == old_name:
                    parts[0] = new_name
                    file.write("  ".join(parts) + "\n")
                    continue
                if atom_name_2 == old_name:
                    parts[1] = new_name
                    file.write("  ".join(parts) + "\n")
                    continue
            else:
                continue


def make_marnatini_itp():
    """Generate and copy a Martini RNA ITP file.

    This high-level function processes a base Martini ITP file using defined name mappings
    and copies the resulting file to several target directories.
    """
    dict_of_names = {
        "TA1": "SC5",
        "TA2": "TN1a",
        "TA3": "TC6",
        "TA4": "TN4r",
        "TA5": "TN4r",
        "TA6": "TN1a",
        "TY1": "SC3",
        "TY2": "TN3",
        "TY3": "TN4",
        "TY4": "TN3",
        "TY5": None,
        "TG1": "SC5",
        "TG2": "TN1a",
        "TG3": "TN3r",
        "TG4": "TN4r",
        "TG5": "TN4r",
        "TG6": "TN1a",
        "TG7": None,
        "TG8": None,
        "TU1": "SC3",
        "TU2": "TN3",
        "TU3": "TN4",
        "TU4": "TN3",
        "TU5": None,
        "TU6": None,
        "TU7": None,
    }
    out_file = "reforge/itp/martini_RNA.itp"
    make_in_terms("reforge/itp/martini.itp", out_file, dict_of_names)
    for new_name, old_name in dict_of_names.items():
        make_cross_terms("reforge/itp/martini.itp", out_file, old_name, new_name)
    sh.copy(out_file, "/scratch/dyangali/reforge/systems/dsRNA/topol/martini_v3.0.0_rna.itp")
    sh.copy(out_file, "/scratch/dyangali/reforge/systems/ssRNA/topol/martini_v3.0.0_rna.itp")
    sh.copy(out_file, "/scratch/dyangali/maRNAtini_sims/dimerization_pmf_us/topol/martini_RNA.itp")
    sh.copy(out_file, "/scratch/dyangali/maRNAtini_sims/angled_dimerization_pmf_us/topol/martini_RNA.itp")


def make_ions_itp():
    """Generate an ITP file for ions.

    This function modifies a base Martini ITP file, adjusts parameters, and writes
    the final Martini ions ITP file.
    """
    import pandas as pd

    dict_of_names = {"TMG": "TD"}
    out_file = "reforge/itp/ions.itp"
    for new_name, old_name in dict_of_names.items():
        make_cross_terms("reforge/itp/martini_v3.0.0.itp", out_file, old_name, new_name)
    df = pd.read_csv(out_file, sep="\\s+", header=None)
    df[3] -= 0.08
    tmp_file = "reforge/itp/ions_tmp.itp"
    df.to_csv(tmp_file, sep=" ", header=None, index=False, float_format="%.6e")
    out_file = "reforge/itp/martini_ions.itp"
    new_lines = [
        "[ atomtypes ]\n",
        "TMG  45.000  0.000  A  0.0  0.0\n\n",
        "[ nonbond_params ]\n",
        "TMG TMG 1 3.580000e-01 1.100000e+00\n",
    ]
    with open(tmp_file, "r", encoding="utf-8") as file:
        original_content = file.readlines()
    with open(out_file, "w+", encoding="utf-8") as file:
        file.writelines(new_lines + original_content)


def count_itp_atoms(file_path):
    """Count the number of atom entries in the [ atoms ] section of an ITP file.

    Parameters
    ----------
    file_path : str
        Path to the ITP file.

    Returns
    -------
    int
        The atom count, or 0 if an error occurs.
    """
    in_atoms_section = False
    atom_count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                if line.startswith("[ atoms ]"):
                    in_atoms_section = True
                    continue
                if in_atoms_section and line.startswith("["):
                    break
                if in_atoms_section:
                    atom_count += 1
        return atom_count
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


if __name__ == "__main__":
    pass
    # make_ions_itp()
    # make_marnatini_itp()
