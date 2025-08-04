"""Collection of CG force fields

Description:
    This module defines force fields for Martini RNA and nucleic acids.
    It provides classes for reading ITP files and organizing force-field parameters
    for coarse-grained simulations.

Author: DY    
"""

import importlib.resources
import os
from reforge import itpio

# Split each argument into a list of tokens.
def nsplit(*x):
    return [i.split() for i in x]

# List of available force fields
forcefields = ["martini30rna", "martini31nucleic"]
RNA_SYSTEM = "test"


###################################
## FORCE FIELDS BASE CLASS ##
###################################

class NucleicForceField:
    """
    Base class for nucleic acid force fields.
    """

    @staticmethod
    def read_itp(resname, directory, mol, version):
        """
        Read an ITP file for a given residue.

        Args:
            resname (str): Residue name.
            directory (str): Directory where force field files are stored.
            mol (str): Molecule name.
            version (str): Version string.

        Returns:
            dict: Parsed ITP data.
        """
        itpdir = importlib.resources.files("reforge") / "forge" / "forcefields"
        file_path = os.path.join(itpdir, directory, f"{mol}_{resname}_{version}.itp")
        itp_data = itpio.read_itp(file_path)
        return itp_data

    @staticmethod
    def read_itps(mol, ff_type, version):
        """
        Read ITP files for nucleobases A, C, G, and U.

        Args:
            mol (str): Molecule name.
            ff_type (str): Force-field type (e.g. "polar").
            version (str): Version string.

        Returns:
            tuple: (a_itp, c_itp, g_itp, u_itp)
        """
        a_itp = NucleicForceField.read_itp("A", ff_type, mol, version)
        c_itp = NucleicForceField.read_itp("C", ff_type, mol, version)
        g_itp = NucleicForceField.read_itp("G", ff_type, mol, version)
        u_itp = NucleicForceField.read_itp("U", ff_type, mol, version)
        return a_itp, c_itp, g_itp, u_itp

    @staticmethod
    def itp_to_indata(itp_data):
        """
        Convert ITP data into individual parameter lists.

        Args:
            itp_data (dict): Parsed ITP file data.

        Returns:
            tuple: (sc_bonds, sc_angles, sc_dihs, sc_cons, sc_excls, sc_pairs, sc_vs3s)
        """
        sc_bonds = itp_data["bonds"]
        sc_angles = itp_data["angles"]
        sc_dihs = itp_data["dihedrals"]
        sc_cons = itp_data["constraints"]
        sc_excls = itp_data["exclusions"]
        sc_pairs = itp_data["pairs"]
        sc_vs3s = itp_data["virtual_sites3"]
        return sc_bonds, sc_angles, sc_dihs, sc_cons, sc_excls, sc_pairs, sc_vs3s

    @staticmethod
    def parameters_by_resname(resnames, directory, mol, version):
        """
        Obtain parameters for each residue.

        Args:
            resnames (list): List of residue names.
            directory (str): Directory containing ITP files.
            mol (str): Molecule name.
            version (str): Version string.

        Returns:
            dict: Mapping of residue name to its parameters.
        """
        params = []
        for resname in resnames:
            itp_data = NucleicForceField.read_itp(resname, directory, mol, version)
            param = NucleicForceField.itp_to_indata(itp_data)
            params.append(param)
        return dict(zip(resnames, params))

    def __init__(self, directory, mol, version):
        """
        Initialize the nucleic force field.

        Args:
            directory (str): Directory with force field files.
            mol (str): Molecule name.
            version (str): Version string.
        """
        self.directory = directory
        self.mol = mol
        self.version = version
        self.resdict = self.parameters_by_resname(self.resnames, directory, mol, version)
        self.elastic_network = False
        self.el_bond_type = 6  # Elastic network bond type

    def sc_bonds(self, resname):
        return self.resdict[resname][0]

    def sc_angles(self, resname):
        return self.resdict[resname][1]

    def sc_dihs(self, resname):
        return self.resdict[resname][2]

    def sc_cons(self, resname):
        return self.resdict[resname][3]

    def sc_excls(self, resname):
        return self.resdict[resname][4]

    def sc_pairs(self, resname):
        return self.resdict[resname][5]

    def sc_vs3s(self, resname):
        return self.resdict[resname][6]

    def sc_blist(self, resname):
        """
        Get all bonded parameters for a residue.

        Args:
            resname (str): Residue name.

        Returns:
            list: List of bonded parameter lists.
        """
        return [
            self.sc_bonds(resname),
            self.sc_angles(resname),
            self.sc_dihs(resname),
            self.sc_cons(resname),
            self.sc_excls(resname),
            self.sc_pairs(resname),
            self.sc_vs3s(resname),
        ]

    @staticmethod
    def update_non_standard_mapping(mapping):
        mapping.update({"RA3":mapping["A"],
                        "RA5":mapping["A"],
                        "A3":mapping["A"],
                        "A5":mapping["A"],
                        "2MA":mapping["A"],
                        "6MA":mapping["A"],
                        "RAP":mapping["A"],
                        "DMA":mapping["A"],
                        "DHA":mapping["A"],
                        "SPA":mapping["A"],
                        "RC3":mapping["C"],
                        "RC5":mapping["C"],
                        "C3":mapping["C"],
                        "C5":mapping["C"],
                        "5MC":mapping["C"],
                        "3MP":mapping["C"],
                        "MRC":mapping["C"],
                        "NMC":mapping["C"],
                        "RG3":mapping["G"],
                        "RG5":mapping["G"],
                        "G3":mapping["G"],
                        "G5":mapping["G"],
                        "1MG":mapping["G"],
                        "2MG":mapping["G"],
                        "7MG":mapping["G"],
                        "MRG":mapping["G"],
                        "RU3":mapping["U"],
                        "RU5":mapping["U"],
                        "U3":mapping["U"],
                        "U5":mapping["U"],
                        "4SU":mapping["U"],
                        "DHU":mapping["U"],
                        "PSU":mapping["U"],
                        "5MU":mapping["U"],
                        "3MU":mapping["U"],
                        "3MP":mapping["U"],
                        "MRU":mapping["U"],
        })


###################################
## Martini 3.0 RNA Force Field ##
###################################

class Martini30RNA(NucleicForceField):
    """
    Force field for Martini 3.0 RNA.
    """
    resnames = ["A", "C", "G", "U"]
    alt_resnames = ["ADE", "CYT", "GUA", "URA"]

    bb_mapping = {
        "BB1": ("P", "OP1", "OP2", "O5'", "O3'", "O1P", "O2P"),
        "BB2": ("C5'", "1H5'", "2H5'", "H5'", "H5''", "C4'", "H4'", "O4'", "C3'", "H3'"),
        "BB3": ("C1'", "C2'", "O2'", "O4'"),
    }
    a_mapping = {
        "SC1": ("N9", "C8", "H8"),
        "SC2": ("N3", "C4"),
        "SC3": ("N1", "C2", "H2"),
        "SC4": ("N6", "C6", "H61", "H62"),
        "SC5": ("N7", "C5"),
    }
    c_mapping = {
        "SC1": ("N1", "C5", "C6"),
        "SC2": ("C2", "O2"),
        "SC3": ("N3",),
        "SC4": ("N4", "C4", "H41", "H42"),
    }
    g_mapping = {
        "SC1": ("C8", "H8", "N9"),
        "SC2": ("C4", "N3"),
        "SC3": ("C2", "N2", "H21", "H22",),
        "SC4": ("N1",),
        "SC5": ("C6", "O6"),
        "SC6": ("C5", "N7"),
    }
    u_mapping = {
        "SC1": ("N1", "C5", "C6"),
        "SC2": ("C2", "O2"),
        "SC3": ("N3",),
        "SC4": ("C4", "O4"),
    }
    mapping = { "A": {**bb_mapping, **a_mapping},
        "ADE": {**bb_mapping, **a_mapping},       
        "C": {**bb_mapping, **c_mapping},
        "CYT": {**bb_mapping, **c_mapping},
        "G": {**bb_mapping, **g_mapping},
        "GUA": {**bb_mapping, **g_mapping},
        "U": {**bb_mapping, **u_mapping},
        "URA": {**bb_mapping, **u_mapping},
    }

    def __init__(self, directory="rna_reg", mol=RNA_SYSTEM, version="new"):
        super().__init__(directory, mol, version)
        self.name = "martini30rna"

        # RNA backbone atoms: tuple of (atom id, type, name, charge group, charge, mass)
        self.bb_atoms = [
            (0, "Q1n", "BB1", 1, -1, 72),
            (1, "N1", "BB2", 1, 0, 60),
            (2, "N3", "BB3", 1, 0, 60),
        ]
        self.bb_bonds = [
            [(0, 1), (1, 0.351, 18000), ("BB1-BB2")],
            [(1, 2), (1, 0.238, 18000), ("BB2-BB3")],
            [(1, 0), (1, 0.375, 12000), ("BB2-BB1n")],
            [(2, 0), (1, 0.414, 12000), ("BB3-BB1n")],
        ]
        self.bb_angles = [
            [(0, 1, 0), (10, 106.0,  50), ("BB1-BB2-BB1n")],
            [(1, 0, 1), (10, 123.0, 150), ("BB2-BB1n-BB2n")],
            [(0, 1, 2), (10, 142.0, 300), ("BB1-BB2-BB3")],
        ]
        # self.bb_dihs = [
        #     [( 0, 1, 0, 1), (9,  180.0,  8.0, 1), ("BB1-BB2-BB1n-BB2n")],  # 3/29/2025 3.18 good version
        #     [( 0, 1, 0, 1), (9,    0.0,  1.7, 2), ("BB1-BB2-BB1n-BB2n")],  
        #     [( 0, 1, 0, 1), (9,    0.0,  1.0, 4), ("BB1-BB2-BB1n-BB2n")],  
        #     [( 0, 1, 0, 1), (9,    0.0,  2.2, 5), ("BB1-BB2-BB1n-BB2n")],  
        #     [(-2, 0, 1, 0), (1,   30.0,  7.0, 1), ("BB2p-BB1-BB2-BB1n")],
        #     [(-2, 0, 1, 2), (1, -100.0, 11.0, 1), ("BB2p-BB1-BB2-BB3")],
        # ]
        self.bb_dihs = [
            # [(0, 1, 0, 1), (1, 0.0, 12.0, 1), ("BB1-BB2-BB1n-BB2n")],  (3, 13, -7, -25, -6, 25, -2),
            [(0, 1, 0, 1), (3, 13, -7, -25, -6, 25, 2), ("BB1-BB2-BB1n-BB2n")],  # 2, 2.5 or 0 stable
            [(-2, 0, 1, 0), (1, 0.0, 7.0, 1), ("BB2p-BB1-BB2-BB1n")],
            [(-2, 0, 1, 2), (1, -105.0, 10.0, 1), ("BB2p-BB1-BB2-BB3")],
        ]
        self.bb_cons = []
        self.bb_excls = [[(0, 2), (), ("BB1-BB3")], [(2, 0), (), ("BB3-BB1n")]]
        self.bb_pairs = []
        self.bb_vs3s = []
        self.bb_blist = [
            self.bb_bonds,
            self.bb_angles,
            self.bb_dihs,
            self.bb_cons,
            self.bb_excls,
            self.bb_pairs,
            self.bb_vs3s,
        ]

        # Side-chain atom definitions for each base.
        a_atoms = [
            (3, "TA0", "SC1", 2, 0, 45),
            (4, "TA1", "SC2", 2, 0, 0),
            (5, "TA2", "SC3", 2, 0, 45),
            (6, "TA3", "SC4", 2, 0, 45),
            (7, "TA4", "SC5", 2, 0, 0),
        ]
        c_atoms = [
            (3, "TY0", "SC1", 2, 0, 37),
            (4, "TY1", "SC2", 2, 0, 37),
            (5, "TY2", "SC3", 2, 0, 0),
            (6, "TY3", "SC4", 2, 0, 37),
        ]
        g_atoms = [
            (3, "TG0", "SC1", 2, 0, 50),
            (4, "TG1", "SC2", 2, 0, 0),
            (5, "TG2", "SC3", 2, 0, 50),
            (6, "TG3", "SC4", 2, 0, 0),
            (7, "TG4", "SC5", 2, 0, 50),
            (8, "TG5", "SC6", 2, 0, 0),
        ]
        u_atoms = [
            (3, "TU0", "SC1", 2, 0, 37),
            (4, "TU1", "SC2", 2, 0, 37),
            (5, "TU2", "SC3", 2, 0, 0),
            (6, "TU3", "SC4", 2, 0, 37),
        ]
        sc_atoms = (a_atoms, c_atoms, g_atoms, u_atoms)
        self.mapdict = dict(zip(self.resnames, sc_atoms))

        NucleicForceField.update_non_standard_mapping(self.mapping)

    def sc_atoms(self, resname):
        """Return side-chain atoms for the given residue."""
        return self.mapdict[resname]


class Martini31Nucleic(NucleicForceField):
    """
    Force field for Martini 3.1 nucleic acids.
    """
    bb_mapping = nsplit(
        "P OP1 OP2 O5' O3' O1P O2P",
        "C5' 1H5' 2H5' H5' H5'' C4' H4' O4' C3' H3'",
        "C1' C2' O2' O4'",
    )
    mapping = {
        "A": {**dict(zip(["SC1"], nsplit("TA1 TA2 TA3 TA4 TA5 TA6"))), **{}},
        "C": {**dict(zip(["SC1"], nsplit("TY1 TY2 TY3 TY4 TY5"))), **{}},
        "G": {**dict(zip(["SC1"], nsplit("TG1 TG2 TG3 TG4 TG5 TG6 TG7 TG8"))), **{}},
        "U": {**dict(zip(["SC1"], nsplit("TU1 TU2 TU3 TU4 TU5 TU6 TU7"))), **{}},
    }

    def __init__(self):
        super().__init__(directory="rna_pol", mol=RNA_SYSTEM, version="new")
        self.name = "martini31nucleic"
        charges = {
            "TDU": 0.5,
            "TA1": 0.4,
            "TA2": -0.3,
            "TA3": 0.5,
            "TA4": -0.8,
            "TA5": 0.6,
            "TA6": -0.4,
            "TY1": 0.0,
            "TY2": -0.5,
            "TY3": -0.6,
            "TY4": 0.6,
            "TY5": 0.5,
            "TG1": 0.3,
            "TG2": 0.0,
            "TG3": 0.3,
            "TG4": -0.3,
            "TG5": -0.5,
            "TG6": -0.6,
            "TG7": 0.3,
            "TG8": 0.5,
            "TU1": 0.0,
            "TU2": -0.5,
            "TU3": -0.5,
            "TU4": -0.5,
            "TU5": 0.5,
            "TU6": 0.5,
            "TU7": 0.5,
        }
        self.charges = {key: value * 1.8 for key, value in charges.items()}
        self.bbcharges = {"BB1": -1}
        self.use_bbs_angles = False
        self.use_bbbb_dihedrals = False

        self.dna_bb = {
            "atom": nsplit("Q0 SN0 SC2"),
            "bond": [],
            "angle": [],
            "dih": [],
            "excl": [],
            "pair": [],
        }
        self.dna_con = {
            "bond": [],
            "angle": [],
            "dih": [],
            "excl": [],
            "pair": [],
        }

        self.bases = {}
        self.base_connectivity = {}

        self.rna_bb = {
            "atom": nsplit("Q1 N4 N6"),
            "bond": [
                (1, 0.349, 18000),
                (1, 0.377, 12000),
                (1, 0.240, 18000),
                (1, 0.412, 12000),
            ],
            "angle": [(10, 119.0, 27), (10, 118.0, 140), (10, 138.0, 180)],
            "dih": [
                (3, 13, -7, -25, -6, 25, -2),
                (1, 0, 6, 1),
                (1, -112.0, 15, 1),
            ],
            "excl": [(), (), ()],
            "pair": [],
        }
        self.rna_con = {
            "bond": [(0, 1), (1, 0), (1, 2), (2, 0)],
            "angle": [(0, 1, 0), (1, 0, 1), (0, 1, 2)],
            "dih": [(0, 1, 0, 1), (1, 0, 1, 0), (1, 0, 1, 2)],
            "excl": [(2, 0), (0, 2)],
            "pair": [],
        }

        a_itp, c_itp, g_itp, u_itp = NucleicForceField.read_itps(RNA_SYSTEM, "polar", "new")

        mapping_a = nsplit("TA1 TA2 TA3 TA4 TA5 TA6")
        connectivity_a, itp_params_a = self.itp_to_indata(a_itp)
        self.update_adenine(mapping_a, connectivity_a, itp_params_a)

        mapping_c = nsplit("TY1 TY2 TY3 TY4 TY5")
        connectivity_c, itp_params_c = self.itp_to_indata(c_itp)
        self.update_cytosine(mapping_c, connectivity_c, itp_params_c)

        mapping_g = nsplit("TG1 TG2 TG3 TG4 TG5 TG6 TG7 TG8")
        connectivity_g, itp_params_g = self.itp_to_indata(g_itp)
        self.update_guanine(mapping_g, connectivity_g, itp_params_g)

        mapping_u = nsplit("TU1 TU2 TU3 TU4 TU5 TU6 TU7")
        connectivity_u, itp_params_u = self.itp_to_indata(u_itp)
        self.update_uracil(mapping_u, connectivity_u, itp_params_u)

        super().__init__()

    def sc_atoms(self, resname):
        """Return side-chain atoms for a given residue."""
        return self.mapdict[resname]

    # The update_* methods are currently commented out.
    # They can be implemented if needed.
    # def update_adenine(self, mapping, connectivity, itp_params):
    #     pass
    # def update_cytosine(self, mapping, connectivity, itp_params):
    #     pass
    # def update_guanine(self, mapping, connectivity, itp_params):
    #     pass
    # def update_uracil(self, mapping, connectivity, itp_params):
    #     pass


    # def update_adenine(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"A": parameters})
    #     self.base_connectivity.update({"A": connectivity})
    #     self.bases.update({"RA3": parameters})
    #     self.base_connectivity.update({"RA3": connectivity})
    #     self.bases.update({"RA5": parameters})
    #     self.base_connectivity.update({"RA5": connectivity})
    #     self.bases.update({"2MA": parameters})
    #     self.base_connectivity.update({"2MA": connectivity})
    #     self.bases.update({"DMA": parameters})
    #     self.base_connectivity.update({"DMA": connectivity})
    #     self.bases.update({"SPA": parameters})
    #     self.base_connectivity.update({"SPA": connectivity})
    #     self.bases.update({"RAP": parameters})
    #     self.base_connectivity.update({"RAP": connectivity})
    #     self.bases.update({"6MA": parameters})
    #     self.base_connectivity.update({"6MA": connectivity})

    # def update_cytosine(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"C": parameters})
    #     self.base_connectivity.update({"C": connectivity})
    #     self.bases.update({"RC3": parameters})
    #     self.base_connectivity.update({"RC3": connectivity})
    #     self.bases.update({"RC5": parameters})
    #     self.base_connectivity.update({"RC5": connectivity})
    #     self.bases.update({"MRC": parameters})
    #     self.base_connectivity.update({"MRC": connectivity})
    #     self.bases.update({"5MC": parameters})
    #     self.base_connectivity.update({"5MC": connectivity})
    #     self.bases.update({"NMC": parameters})
    #     self.base_connectivity.update({"NMC": connectivity})

    # def update_guanine(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"G": parameters})
    #     self.base_connectivity.update({"G": connectivity})
    #     self.bases.update({"RG3": parameters})
    #     self.base_connectivity.update({"RG3": connectivity})
    #     self.bases.update({"RG5": parameters})
    #     self.base_connectivity.update({"RG5": connectivity})
    #     self.bases.update({"MRG": parameters})
    #     self.base_connectivity.update({"MRG": connectivity})
    #     self.bases.update({"1MG": parameters})
    #     self.base_connectivity.update({"1MG": connectivity})
    #     self.bases.update({"2MG": parameters})
    #     self.base_connectivity.update({"2MG": connectivity})
    #     self.bases.update({"7MG": parameters})
    #     self.base_connectivity.update({"7MG": connectivity})

    # def update_uracil(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"U": parameters})
    #     self.base_connectivity.update({"U": connectivity})
    #     self.bases.update({"RU3": parameters})
    #     self.base_connectivity.update({"RU3": connectivity})
    #     self.bases.update({"RU5": parameters})
    #     self.base_connectivity.update({"RU5": connectivity})
    #     self.bases.update({"MRU": parameters})
    #     self.base_connectivity.update({"MRU": connectivity})
    #     self.bases.update({"DHU": parameters})
    #     self.base_connectivity.update({"DHU": connectivity})
    #     self.bases.update({"PSU": parameters})
    #     self.base_connectivity.update({"PSU": connectivity})
    #     self.bases.update({"3MP": parameters})
    #     self.base_connectivity.update({"3MP": connectivity})
    #     self.bases.update({"3MU": parameters})
    #     self.base_connectivity.update({"3MU": connectivity})
    #     self.bases.update({"4SU": parameters})
    #     self.base_connectivity.update({"4SU": connectivity})
    #     self.bases.update({"5MU": parameters})
    #     self.base_connectivity.update({"5MU": connectivity})

    # @staticmethod
    # def update_non_standard_mapping(mapping):
    #     mapping.update({"RA3":mapping["A"],
    #                     "RA5":mapping["A"],
    #                     "2MA":mapping["A"],
    #                     "6MA":mapping["A"],
    #                     "RAP":mapping["A"],
    #                     "DMA":mapping["A"],
    #                     "DHA":mapping["A"],
    #                     "SPA":mapping["A"],
    #                     "RC3":mapping["C"],
    #                     "RC5":mapping["C"],
    #                     "5MC":mapping["C"],
    #                     "3MP":mapping["C"],
    #                     "MRC":mapping["C"],
    #                     "NMC":mapping["C"],
    #                     "RG3":mapping["G"],
    #                     "RG5":mapping["G"],
    #                     "1MG":mapping["G"],
    #                     "2MG":mapping["G"],
    #                     "7MG":mapping["G"],
    #                     "MRG":mapping["G"],
    #                     "RU3":mapping["U"],
    #                     "RU5":mapping["U"],
    #                     "4SU":mapping["U"],
    #                     "DHU":mapping["U"],
    #                     "PSU":mapping["U"],
    #                     "5MU":mapping["U"],
    #                     "3MU":mapping["U"],
    #                     "3MP":mapping["U"],
    #                     "MRU":mapping["U"],
    #     })
