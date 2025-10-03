"""
Description:
    Provides geometric calculations (distances, angles, dihedrals) for coarse-grained
    systems and functions to compute bond lists from a system (using a reference topology).

Author: DY
"""

import sys
import logging
import numpy as np
from reforge.pdbtools import pdb2system
from reforge.forge import cgmap
from reforge.forge.topology import BondList

logger = logging.getLogger(__name__)


def get_distance(v1, v2):
    """
    Compute the Euclidean distance between two vectors (returned in nanometers).

    Args:
        v1: First vector (list or array-like).
        v2: Second vector (list or array-like).

    Returns:
        float: Distance between v1 and v2 divided by 10.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return 0.1 * np.linalg.norm(v1 - v2) 


def get_angle(v1, v2, v3):
    """
    Calculate the angle at vertex v2 (in degrees) given three points.

    Args:
        v1: First vector (list or array-like).
        v2: Vertex vector (list or array-like).
        v3: Third vector (list or array-like).

    Returns:
        float: Angle in degrees.
    """
    v1, v2, v3 = map(np.array, (v1, v2, v3))
    vec1 = v1 - v2
    vec2 = v3 - v2
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_radians)


def get_dihedral(v1, v2, v3, v4):
    """
    Calculate the dihedral angle (in degrees) defined by four points.

    Args:
        v1: First vector.
        v2: Second vector.
        v3: Third vector.
        v4: Fourth vector.

    Returns:
        float: Dihedral angle in degrees.
    """
    v1, v2, v3, v4 = map(np.array, (v1, v2, v3, v4))
    b1 = v2 - v1
    b2 = v3 - v2
    b3 = v4 - v3
    b2n = b2 / np.linalg.norm(b2)
    n1 = np.cross(b1, b2)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(b2, b3)
    n2 /= np.linalg.norm(n2)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, b2n), n2)
    return np.degrees(np.arctan2(y, x))


def calc_bonds(atoms, bonds):
    """
    Calculate bond distances from a topology bonds object and a list of atoms.

    Args:
        atoms: List of atom objects.
        bonds: Topology bonds object with attributes 'conns', 'params', and 'comms'.

    Returns:
        BondList: A list of bonds with computed distances.
    """
    conns = bonds.conns
    params = bonds.params
    comms = bonds.comms
    pairs = [(atoms[i - 1], atoms[j - 1]) for i, j in conns]
    vecs_list = [(a1.vec, a2.vec) for a1, a2 in pairs]
    distances = [get_distance(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2 in pairs]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, distances)]
    comms = [f"{resname} {comm}" for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms))
    return BondList(result)


def calc_angles(atoms, angles):
    """
    Calculate bond angles from a topology angles object and a list of atoms.

    Args:
        atoms: List of atom objects.
        angles: Topology angles object with attributes 'conns', 'params', and 'comms'.

    Returns:
        BondList: A list of angles with computed values.
    """
    conns = angles.conns
    params = angles.params
    comms = angles.comms
    triplets = [(atoms[i - 1], atoms[j - 1], atoms[k - 1]) for i, j, k in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec) for a1, a2, a3 in triplets]
    angle_values = [get_angle(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2, a3 in triplets]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, angle_values)]
    comms = [f"{resname} {comm}" for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms))
    return BondList(result)


def calc_dihedrals(atoms, dihs):
    """
    Calculate dihedral angles from a topology dihedrals object and a list of atoms.

    Args:
        atoms: List of atom objects.
        dihs: Topology dihedrals object with attributes 'conns', 'params', and 'comms'.

    Returns:
        BondList: A list of dihedrals with computed values.
    """
    conns = dihs.conns
    params = dihs.params
    comms = dihs.comms
    quads = [(atoms[i - 1], atoms[j - 1], atoms[k - 1], atoms[l - 1])
             for i, j, k, l in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec, a4.vec) for a1, a2, a3, a4 in quads]
    dihedrals = [get_dihedral(*vecs) for vecs in vecs_list]
    resnames = [a2.resname for a1, a2, a3, a4 in quads]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, dihedrals)]
    comms = [f"{resname} {comm}" for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms))
    return BondList(result)


def get_cg_bonds(inpdb, top):
    """
    Calculate bonds, angles, and dihedrals for a coarse-grained system.

    Args:
        inpdb (str): Input PDB file for the coarse-grained system.
        top: Topology object containing bond, angle, and dihedral definitions.

    Returns:
        tuple: Three BondList objects: (bonds, angles, dihedrals).
    """
    logger.info("Calculating bonds, angles and dihedrals from %s...", inpdb)
    system = pdb2system(inpdb)
    bonds = BondList()
    angles = BondList()
    dihs = BondList()
    for model in system:
        bonds.extend(calc_bonds(model.atoms, top.bonds + top.cons))
        angles.extend(calc_angles(model.atoms, top.angles))
        dihs.extend(calc_dihedrals(model.atoms, top.dihs))
    print("Done!", file=sys.stderr)
    return bonds, angles, dihs


def get_aa_bonds(inpdb, ff, top):
    """
    Calculate bonds, angles, and dihedrals for an all-atom system.

    Args:
        inpdb (str): Input PDB file for the all-atom system.
        ff: Force field object.
        top: Topology object containing reference bonds, angles, and dihedrals.

    Returns:
        tuple: Three BondList objects: (bonds, angles, dihedrals).
    """
    logger.info("Calculating bonds, angles and dihedrals from %s...", inpdb)
    system = pdb2system(inpdb)
    cgmap.move_o3(system)
    bonds = BondList()
    angles = BondList()
    dihs = BondList()
    for model in system:
        mapped_model = cgmap.map_model(model, ff, atid=1)
        bonds.extend(calc_bonds(mapped_model, top.bonds + top.cons))
        angles.extend(calc_angles(mapped_model, top.angles))
        dihs.extend(calc_dihedrals(mapped_model, top.dihs))
    print("Done!", file=sys.stderr)
    return bonds, angles, dihs
