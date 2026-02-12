"""File: mdsystem.py

Description:
    This module provides classes and functions for setting up, running, and
    analyzing molecular dynamics (MD) simulations.  The main
    classes include:

Usage:
  
Requirements:
    - Python 3.x
    - MDAnalysis
    - NumPy
    - Pandas
    - The reforge package and its dependencies

Author: DY
Date: 2025-02-27
"""

import importlib.resources
from pathlib import Path
import sys
import shutil
import subprocess as sp
import tempfile
import logging
import MDAnalysis as mda
import numpy as np
from collections import defaultdict
from openmm.app import PDBFile
from pdbfixer.pdbfixer import PDBFixer
from reforge import cli, mdm, pdbtools, io
from reforge.utils import cd, clean_dir
from reforge.martini import getgo, martini_tools
from reforge.forge.topology import Topology

logger = logging.getLogger(__name__)

################################################################################
# GMX system class
################################################################################

class MDSystem:
    """
    Most attributes are paths to files and directories needed to set up
    and run the MD simulation.
    """
    NUC_RESNAMES = ["A", "C", "G", "U",
                    "RA3", "RA5", "RC3", "RC5", 
                    "RG3", "RG5", "RU3", "RU5",]

    def __init__(self, sysdir, sysname):
        """Initializes the MD system with required directories and file paths.

        Parameters
        ----------
            sysdir (str): Base directory for collection of MD systems
            sysname (str): Name of the MD system.

        Sets up paths for various files required for coarse-grained MD simulation.
        """
        self.sysname = sysname
        self.sysdir = Path(sysdir).resolve()
        self.root = self.sysdir / sysname
        self.inpdb = self.root / "inpdb.pdb"
        self.solupdb = self.root / "solute.pdb"
        self.prodir = self.root / "proteins"
        self.nucdir = self.root / "nucleotides"
        self.iondir = self.root / "ions"
        self.ligdir = self.root / "ligands"
        self.ionpdb = self.iondir / "ions.pdb"
        self.topdir = self.root / "topol"
        self.mddir = self.root / "mdruns"
        self.datdir = self.root / "data"
        self.pngdir = self.root / "png"
        self.pdbdir = self.root / "pdb"
        self.sysxml = self.root / "system.xml"
        self.systop = self.root / "system.top"
        self.sysgro = self.root / "system.gro"
        self.syspdb = self.root / "system.pdb"

    @property
    def chains(self):
        """Retrieves and returns a sorted list of chain identifiers from the
        input PDB.

        Returns:
            list: Sorted chain identifiers extracted from the PDB file.
        """
        atoms = io.pdb2atomlist(self.inpdb)
        chains = pdbtools.sort_uld(set(atoms.chids))
        return chains

    @property
    def segments(self):
        """Same as for chains but for segments IDs"""
        atoms = io.pdb2atomlist(self.inpdb)
        segments = pdbtools.sort_uld(set(atoms.segids))
        return segments

    def prepare_files(self, pour_martini=False):
        """Prepares the simulation by creating necessary directories and copying input files.

        The method:
          - Creates directories for proteins, nucleotides, topologies, maps, mdp files,
            coarse-grained PDBs, GRO files, MD runs, data, and PNG outputs.
          - Copies 'water.gro' and 'atommass.dat' from the master data directory.
          - Copies .itp files from the master ITP directory to the system topology directory.
        """
        logger.info("Preparing files and directories")
        self.prodir.mkdir(parents=True, exist_ok=True)
        self.nucdir.mkdir(parents=True, exist_ok=True)
        self.datdir.mkdir(parents=True, exist_ok=True)
        self.pngdir.mkdir(parents=True, exist_ok=True)
        if pour_martini:
            # Delegate Martini-specific setup to the Martini mixin if present.
            # This keeps MDSystem generic while still supporting the legacy
            # `pour_martini=True` flag.
            prepare_martini = getattr(self, "prepare_martini_files", None)
            if callable(prepare_martini):
                prepare_martini()
            else:
                raise AttributeError(
                    "pour_martini=True requires a class providing Martini behavior "
                    "(e.g., inherit from MartiniMixin which defines prepare_martini_files())."
                )

    def sort_input_pdb(self, in_pdb="inpdb.pdb"):
        """Sorts and renames atoms and chains in the input PDB file.

        Parameters
        ----------
            in_pdb (str): Path to the input PDB file (default: "inpdb.pdb").

        Uses pdbtools to perform sorting and renaming, saving the result to self.inpdb.
        """
        with cd(self.root):
            pdbtools.sort_pdb(in_pdb, self.inpdb)

    def clean_pdb_mm(self, pdb_file, 
                    add_missing_atoms=False, 
                    add_hydrogens=False, 
                    remove_heterogens=True, 
                    keep_water=False,
                    pH=7.0, **kwargs):
        """Clean the starting PDB file using PDBfixer by OpenMM.

        Parameters
        ----------
        pdb_file : str
            Path to the input PDB file.
        add_missing_atoms : bool, optional
            Whether to add missing atoms (default: False).
        add_hydrogens : bool, optional
            Whether to add missing hydrogens (default: False).
        pH : float, optional
            pH value for adding hydrogens (default: 7.0).
        **kwargs : dict, optional
            Additional keyword arguments (ignored).
        """
        logger.info("Cleaning the PDB with PDBfixer...")
        logger.info(f"Processing {pdb_file}")
        pdb = PDBFixer(filename=str(pdb_file))
        logger.info("Removing heterogens and checking for missing residues...")
        if remove_heterogens:
            pdb.removeHeterogens(keep_water)
        pdb.findMissingResidues()
        logger.info("Replacing non-standard residues...")
        pdb.findNonstandardResidues()
        pdb.replaceNonstandardResidues()
        if add_missing_atoms:
            logger.info("Adding missing atoms...")
            pdb.findMissingAtoms()
            pdb.addMissingAtoms()
        if add_hydrogens:
            logger.info("Adding missing hydrogens...")
            pdb.addMissingHydrogens(pH)  
        topology = pdb.topology
        positions = pdb.positions
        with open(self.inpdb, "w", encoding="utf-8") as outfile:
            PDBFile.writeFile(topology, positions, outfile)
        logger.info(f"Written cleaned PDB to {self.inpdb}")

    def clean_pdb_gmx(self, pdb_file, **kwargs):
        """Cleans the PDB file using GROMACS pdb2gmx tool.

        Parameters
        ----------
            in_pdb (str, optional): Input PDB file to clean. If None, uses self.inpdb.
            kwargs: Additional keyword arguments for the GROMACS command.

        After running pdb2gmx, cleans up temporary files (e.g., "topol*" and "posre*").
        """
        logger.info(f"Cleaning the PDB with GROMACS pdb2gmx...")
        logger.info(f"Processing {pdb_file}")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_pdb:
            temp_pdb_path = temp_pdb.name
            with open(pdb_file, 'r') as infile:
                for line in infile:
                    if line.startswith('ATOM'):
                        temp_pdb.write(line)
        try:
            with cd(self.root):
                cli.gmx("pdb2gmx", f=temp_pdb_path, o=self.inpdb, **kwargs)
        finally:
            Path(temp_pdb_path).unlink(missing_ok=True)  # Remove the temporary PDB file
        clean_dir(self.root, "topol*")
        clean_dir(self.root, "posre*")

    def split_chains(self):
        """Splits the input PDB file into separate files for each chain.

        Nucleotide chains are saved to self.nucdir, while protein chains are saved to self.prodir.
        The determination is based on the residue names.
        """
        def it_is_nucleotide(atoms):
            # Check if the chain is nucleotide based on residue name.
            return atoms.resnames[0] in self.NUC_RESNAMES
        logger.info("Splitting chains from the input PDB...")
        system = pdbtools.pdb2system(self.inpdb)
        for chain in system.chains():
            atoms = chain.atoms
            if it_is_nucleotide(atoms):
                out_pdb = self.nucdir / f"chain_{chain.chid}.pdb"
            else:
                out_pdb = self.prodir / f"chain_{chain.chid}.pdb"
            atoms.write_pdb(out_pdb)

    def clean_chains_mm(self, **kwargs):
        """Cleans chain-specific PDB files using PDBfixer (OpenMM).

        Kwargs are passed to pdbtools.clean_pdb. Also renames chain IDs based on the file name.
        """
        kwargs.setdefault("add_missing_atoms", True)
        kwargs.setdefault("add_hydrogens", True)
        kwargs.setdefault("pH", 7.0)
        logger.info("Cleaning chain PDBs using OpenMM...")
        files = list(self.prodir.iterdir())
        files += list(self.nucdir.iterdir())
        files = sorted(files, key=lambda p: p.name)
        for file in files:
            pdbtools.clean_pdb(file, file, **kwargs)
            new_chain_id = file.name.split("chain_")[1][0]
            pdbtools.rename_chain_in_pdb(file, new_chain_id)

    def clean_chains_gmx(self, **kwargs):
        """Cleans chain-specific PDB files using GROMACS pdb2gmx tool.

        Parameters
        ----------
            kwargs: Additional keyword arguments for the GROMACS command.

        Processes all files in the protein and nucleotide directories, renaming chains
        and cleaning temporary files afterward.
        """
        logger.info("Cleaning chain PDBs using GROMACS pdb2gmx...")
        files = [p for p in self.prodir.iterdir() if not p.name.startswith("#")]
        files += [p for p in self.nucdir.iterdir() if not p.name.startswith("#")]
        files = sorted(files, key=lambda p: p.name)
        with cd(self.root):
            for file in files:
                new_chain_id = file.name.split("chain_")[1][0]
                cli.gmx("pdb2gmx", f=file, o=file, **kwargs)
                pdbtools.rename_chain_and_histidines_in_pdb(file, new_chain_id)
            clean_dir(self.prodir)
            clean_dir(self.nucdir)
        clean_dir(self.root, "topol*")
        clean_dir(self.root, "posre*")

    def make_ions_pdb(self, inpdb=None, outpdb=None, ions=["MG", "ZN", "CA", "K"], resname="ION"):
        """Creates a PDB file for the ion species with custom residue name.
        
        Ions are written to the output file grouped by type in the order specified
        in the ions parameter.
        
        Parameters
        ----------
        inpdb : str or Path, optional
            Input PDB file to read ions from (default: self.inpdb)
        outpdb : str or Path, optional
            Output PDB file to write ions to (default: self.ionpdb)
        ions : list, optional
            List of ion names to extract (default: ["MG", "ZN", "CA", "K"])
        resname : str, optional
            Residue name to assign to all ions in output (default: "ION")
        """
        logger.info("Creating ion PDB file...")
        if inpdb is None:
            inpdb = self.inpdb
        if outpdb is None:
            outpdb = self.ionpdb
        # ensure ion directory exists
        Path.mkdir(self.iondir, exist_ok=True, parents=True)
        # Ensure resname is exactly 3 characters (right-padded with spaces if shorter)
        resname_formatted = f"{resname:<3}"[:3]
        # Collect ions grouped by type to maintain order
        ions_by_type = {ion: [] for ion in ions}
        with open(inpdb, 'r', encoding='utf-8') as rin:
            for line in rin:
                # Only consider HETATM records which commonly hold resolved ions
                if line.startswith('HETATM'):
                    # atom name/resname typically in cols 13-16 (1-based) and 17-20 for resname
                    # using same slicing as elsewhere in the file (0-based indices)
                    atom_name = line[12:16].strip()
                    original_resname = line[17:20].strip()
                    # Check either atom name or residue name matches ion list
                    for ion in ions:
                        if original_resname == ion or atom_name == ion:
                            # convert record type to ATOM and replace residue name
                            # PDB format: cols 1-6 record, 7-11 serial, 12-16 atom, 17 altLoc, 18-20 resname, 22 chain...
                            new_line = 'HETATM' + line[6:17] + resname_formatted + line[20:]
                            ions_by_type[ion].append(new_line)
                            break
        # Write ions in order: all of first type, then all of second type, etc.
        written = 0
        with open(outpdb, 'w', encoding='utf-8') as wout:
            for ion in ions:
                for line in ions_by_type[ion]:
                    wout.write(line)
                    written += 1
        logger.info(f"Ion PDB file created: {outpdb} (wrote {written} ions with resname '{resname_formatted.strip()}')")



    def count_resolved_ions(self, ions=("MG", "ZN", "CA", "K")):
        """Counts the number of resolved ions in the system PDB file.

        Parameters
        ----------
        ions (list, optional): 
            List of ion names to count (default: ["MG", "ZN", "CA", "K"]).

        Returns
        -------  
        dict: 
            A dictionary mapping ion names to their counts.
        """
        counts = {ion: 0 for ion in ions}
        with open(self.ionpdb, "r", encoding='utf-8') as file:
            for line in file:
                if line.startswith("HETATM"):
                    current_ion = line[12:16].strip()
                    if current_ion in ions:
                        counts[current_ion] += 1
        # Log the ion counts
        total_ions = sum(counts.values())
        if total_ions > 0:
            logger.info(f"Found {total_ions} resolved ions in {self.ionpdb}:")
            for ion, count in counts.items():
                if count > 0:
                    logger.info(f"  {ion}: {count}")
        else:
            logger.info(f"No resolved ions found in {self.ionpdb}")
        return counts

    def get_mean_sem(self, pattern="dfi*.npy"):
        """Calculates the mean and standard error of the mean (SEM) from numpy files.

        Parameters
        ----------
            pattern (str, optional): Filename pattern to search for (default: "dfi*.npy").

        Saves the calculated averages and errors as numpy files in the data directory.
        """
        logger.info("Calculating averages and errors from %s", pattern)
        files = io.pull_files(self.mddir, pattern)
        datas = [np.load(file) for file in files]
        mean = np.average(datas, axis=0)
        sem = np.std(datas, axis=0) / np.sqrt(len(datas))
        file_mean = self.datdir / f"{pattern.split('*')[0]}_av.npy"
        file_err = self.datdir / f"{pattern.split('*')[0]}_err.npy"
        np.save(file_mean, mean)
        np.save(file_err, sem)

    def get_td_averages(self, pattern):
        """Calculates time-dependent averages from a set of numpy files.
        
        Parameters
        ----------
        fname : str
            Filename pattern to pull files from the MD runs directory.
        loop : bool, optional
            If True, processes files sequentially (default: True).
            
        Returns
        -------
        numpy.ndarray
            The time-dependent average.
        """
        def slicer(shape): # Slice object to crop arrays to min_shape
            return tuple(slice(0, s) for s in shape)
        logger.info("Getting time-dependent averages")
        files = io.pull_files(self.mddir, pattern)
        if files:
            logger.info("Processing %s", files[0])
            average = np.load(files[0])
            min_shape = average.shape
            count = 1
            if len(files) > 1:
                for f in files[1:]:
                    logger.info("Processing %s", f)
                    arr = np.load(f)
                    min_shape = tuple(min(s1, s2) for s1, s2 in zip(min_shape, arr.shape))
                    s = slicer(min_shape)
                    average[s] += arr[s]  # ← in-place addition
                    count += 1
                average = average[s] 
                average /= count
            out_file = self.datdir / f"{pattern.split('*')[0]}_av.npy"     
            np.save(out_file, average)
            logger.info("Done!")
            return average
        else:
            logger.info('Could not find files matching given pattern: %s. Maybe you forgot "*"?', pattern)


class MartiniMixin:
    MDATDIR = importlib.resources.files("reforge") / "martini" / "datdir"
    MITPDIR = importlib.resources.files("reforge") / "martini" / "itp"

    """Martini coarse-grained helpers.

    This is intended to be used as a *mixin* alongside :class:`MDSystem`.
    Methods in this class may assume the host class defines core MDSystem
    attributes like ``self.root`` and ``self.prodir``.
    """

    @property
    def mapdir(self):
        """Directory for GO/contact maps (Martini workflows)."""
        return self.root / "maps"

    @property
    def cgdir(self):
        """Directory for coarse-grained PDBs (Martini workflows)."""
        return self.root / "cgpdb"

    @property
    def molecules(self):
        """Directory for coarse-grained molecules (Martini workflows)."""
        if not hasattr(self, "_molecules"):
            self._molecules = {}
        return self._molecules

    def __init__(self, *args, **kwargs):
        # Keep this cooperative so the mixin doesn't constrain the host class.
        super().__init__(*args, **kwargs)

    def prepare_martini_files(self):
        """Prepare Martini-specific folders and copy shared Martini data.

        This is called by :meth:`MDSystem.prepare_files` when invoked with
        ``pour_martini=True``.
        """
        self.cgdir.mkdir(parents=True, exist_ok=True)
        self.mapdir.mkdir(parents=True, exist_ok=True)
        self.topdir.mkdir(parents=True, exist_ok=True)
        # Copy water.gro and atommass.dat from master data directory
        shutil.copy(self.MDATDIR / "water.gro", self.root)
        shutil.copy(self.MDATDIR / "atommass.dat", self.root)
        # Copy .itp files from the master ITP directory to the system topology directory
        for file in self.MITPDIR.iterdir():
            if file.name.endswith(".itp"):
                outpath = self.topdir / file.name
                shutil.copy(file, outpath)

    def get_go_maps(self, append=False):
        """Retrieves GO contact maps for proteins using the RCSU server.
        
        http://info.ifpan.edu.pl/~rcsu/rcsu/index.html

        Parameters
        ----------
            append (bool, optional): If True, filters out maps that already exist in self.mapdir.
        """
        print("Getting GO-maps", file=sys.stderr)
        pdbs = sorted([self.prodir / f.name for f in self.prodir.iterdir()])
        map_names = [f.name.replace("pdb", "map") for f in self.prodir.iterdir()]
        if append:
            pdbs = [pdb for pdb, amap in zip(pdbs, map_names)
                    if amap not in [f.name for f in self.mapdir.iterdir()]]
        if pdbs:
            getgo.get_go(self.mapdir, pdbs)
        else:
            print("Maps already there", file=sys.stderr)

    def martinize_proteins_go(self, append=False, **kwargs):
        """Performs virtual site-based GoMartini coarse-graining on protein PDBs.

        Uses Martinize2 from https://github.com/marrink-lab/vermouth-martinize.
        All keyword arguments are passed directly to Martinize2. 
        Run `martinize2 -h` to see the full list of parameters.

        Parameters
        ----------
            append (bool, optional): If True, only processes proteins for 
                which corresponding topology files do not already exist.
            kwargs: Additional parameters for the martinize_go function.

        Generates .itp files and cleans temporary directories after processing.
        """
        logger.info("Working on proteins (GoMartini)...")
        pdb_files = sorted([p.name for p in self.prodir.iterdir() if p.is_file() and p.suffix == '.pdb'])
        if not pdb_files:
            logger.warning(f"No PDB files found in protein directory: {self.prodir}")
            return
        # For topology book keeping
        for f in pdb_files:
            mol_name = f.split(".")[0]
            self.molecules[mol_name] = int(1)
        logger.info(f"Found {len(pdb_files)} protein PDB files to process")
        itp_files = [f.replace("pdb", "itp") for f in pdb_files]
        # Filter unprocessed files
        if append:
            pdb_files = [pdb for pdb, itp in zip(pdb_files, itp_files) if not (self.topdir / itp).exists()]
            if not pdb_files:
                logger.info("All protein topology files already exist, skipping processing")
                return
        else:
            clean_dir(self.topdir, "go_*.itp")
        atomtypes_file = self.topdir / "go_atomtypes.itp"
        if not atomtypes_file.is_file():
            with open(atomtypes_file, "w", encoding='utf-8') as f:
                f.write("[ atomtypes ]\n")
        nbparams_file = self.topdir / "go_nbparams.itp"
        if not nbparams_file.is_file():
            with open(nbparams_file, "w", encoding='utf-8') as f:
                f.write("[ nonbond_params ]\n")
        for pdb_file in pdb_files:
            input_pdb = self.prodir / pdb_file
            output_pdb = self.cgdir / pdb_file
            mol_name = pdb_file.split(".")[0]
            go_map = self.mapdir / f"{mol_name}.map"
            martini_tools.run_martinize_go(self.root, self.topdir, input_pdb, output_pdb, name=mol_name, **kwargs)
        clean_dir(self.cgdir)
        clean_dir(self.root)
        clean_dir(self.root, "*.itp")

    def martinize_proteins_en(self, append=False, **kwargs):
        """Generates an elastic network for proteins using the Martini elastic network model.

        Uses Martinize2 from https://github.com/marrink-lab/vermouth-martinize.
        All keyword arguments are passed directly to Martinize2. 
        Run `martinize2 -h` to see the full list of parameters.

        Parameters
        ----------
            append (bool, optional): If True, processes only proteins that do not 
                already have corresponding topology files.
            kwargs: Elastic network parameters (e.g., elastic bond force constants, cutoffs).

        Modifies the generated ITP files by replacing the default molecule name 
        with the actual protein name and cleans temporary files.
        """
        logger.info("Working on proteins (Elastic Network)...")
        pdb_files = sorted([p.name for p in self.prodir.iterdir() if p.is_file() and p.suffix == '.pdb'])
        if not pdb_files:
            logger.warning(f"No PDB files found in protein directory: {self.prodir}")
            return
        # For topology book keeping
        for f in pdb_files:
            mol_name = f.split(".")[0]
            self.molecules[mol_name] = int(1)
        logger.info(f"Found {len(pdb_files)} protein PDB files to process")
        itp_files = [f.replace("pdb", "itp") for f in pdb_files]
        # Filter unprocessed files
        if append:
            pdb_files = [pdb for pdb, itp in zip(pdb_files, itp_files) if not (self.topdir / itp).exists()]
            if not pdb_files:
                logger.info("All protein topology files already exist, skipping processing")
                return
        for pdb_file in pdb_files:
            input_pdb = self.prodir / pdb_file
            output_pdb = self.cgdir / pdb_file
            temp_itp = self.root / "molecule_0.itp"
            final_itp = self.topdir / pdb_file.replace("pdb", "itp")
            temp_top = self.root / "protein.top"
            martini_tools.run_martinize_en(self.root, input_pdb, output_pdb, **kwargs)
            with open(temp_itp, "r", encoding="utf-8") as f:
                content = f.read()
            mol_name = pdb_file[:-4]
            updated_content = content.replace("molecule_0", mol_name, 1)
            with open(final_itp, "w", encoding="utf-8") as f:
                f.write(updated_content)
            temp_top.unlink()
        clean_dir(self.cgdir)
        clean_dir(self.root)

    def martinize_nucleotides(self, **kwargs):
        """Performs coarse-graining on nucleotide PDBs using the martinize_nucleotide tool.

        Parameters
        ----------
            append (bool, optional): If True, skips already existing topologies.
            kwargs: Additional parameters for the martinize_nucleotide function.

        After processing, renames files and moves the resulting ITP files to the topology directory.
        """
        logger.info("Working on nucleotides...")
        pdb_files = [f for f in self.nucdir.iterdir() if f.is_file() and f.suffix == '.pdb']
        if not pdb_files:
            logger.warning(f"No PDB files found in nucleotide directory: {self.nucdir}")
            return
        logger.info(f"Found {len(pdb_files)} nucleotide PDB files to process")
        for pdb_file in pdb_files:
            input_pdb = self.nucdir / pdb_file.name
            output_pdb = self.cgdir / pdb_file.name
            martini_tools.run_martinize_nucleotide(self.root, input_pdb, output_pdb, **kwargs)
        nucleic_files = [p.name for p in self.root.iterdir() if p.name.startswith("Nucleic")]
        for nucleic_file in nucleic_files:
            temp_file_path = self.root / nucleic_file
            command = f"sed -i s/Nucleic_/chain_/g {temp_file_path}"
            sp.run(command.split(), check=True)
            final_filename = nucleic_file.replace("Nucleic", "chain")
            shutil.move(temp_file_path, self.topdir / final_filename)
        clean_dir(self.cgdir)
        clean_dir(self.root)

    def martinize_rna(self, append=False, **kwargs):
        """Coarse-grains RNA molecules using the martinize_rna tool.

        Parameters
        ----------
            append (bool, optional): If True, processes only RNA files without existing topologies.
            kwargs: Additional parameters for the martinize_rna function.

        Exits the process with an error message if coarse-graining fails.
        """
        logger.info("Working on RNA molecules...")
        pdb_files = [p.name for p in self.nucdir.iterdir() if p.is_file() and p.suffix == '.pdb']
        if not pdb_files:
            logger.warning(f"No PDB files found in nucleotide directory: {self.nucdir}")
            return
        # For topology book keeping
        for f in pdb_files:
            mol_name = f.split(".")[0]
            self.molecules[mol_name] = int(1)
        logger.info(f"Found {len(pdb_files)} RNA PDB files to process")
        # Filter unprocessed files
        if append:
            pdb_files = [f for f in pdb_files if not (self.topdir / f.replace("pdb", "itp")).exists()]
            if not pdb_files:
                logger.info("All RNA topology files already exist, skipping processing")
                return
        for pdb_file in pdb_files:
            mol_name = pdb_file.split(".")[0]
            input_pdb = self.nucdir / pdb_file
            output_pdb = self.cgdir / pdb_file
            output_itp = self.topdir / f"{mol_name}.itp"
            martini_tools.run_martinize_rna(self.root, 
                f=input_pdb, os=output_pdb, ot=output_itp, mol=mol_name, **kwargs)

    def _map_ligand(self, map_file, ligand_residue):
        """Map an all-atom ligand residue to Martini beads using a `.map` file.

        Parameters
        ----------
        map_file : str | pathlib.Path
            Path to a martini mapping file with an `[ atoms ]` section.
        ligand_residue : MDAnalysis.core.groups.Residue
            Ligand residue to map.

        Writes
        ------
        A PDB file to `self.cgdir / f"{resname}_{resid}.pdb"` containing one ATOM
        per bead at the center-of-geometry of the mapped atoms.
        """
        map_file = Path(map_file)
        if not map_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {map_file}")
        # --- parse mapping file (only need the [ atoms ] section)
        bead_to_atomnames: dict[str, list[str]] = defaultdict(list)
        in_atoms = False
        with map_file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith(";") or line.startswith("#"):
                    continue
                lower = line.lower()
                if lower.startswith("[") and lower.endswith("]"):
                    in_atoms = (lower == "[ atoms ]")
                    continue
                if not in_atoms:
                    continue
                # Expected format: idx  atomName  beadName
                parts = line.split()
                if len(parts) < 3:
                    continue
                atom_name = parts[1]
                bead_name = parts[2]
                bead_to_atomnames[bead_name].append(atom_name)
        if not bead_to_atomnames:
            raise ValueError(f"No [ atoms ] mapping entries found in {map_file}")
        # Use MDAnalysis atom names (atom.name), and average positions equally.
        aa_atoms = ligand_residue.atoms
        aa_by_name: dict[str, list] = defaultdict(list)
        for idx, a in enumerate(aa_atoms):
            # aa_by_name[f"{a.type}{idx+1}"].append(a)
            aa_by_name[f"{a.name}"].append(a)
        beads = []  # list of (bead_name, xyz)
        for bead_name, atom_names in bead_to_atomnames.items():
            bead_atoms = []
            for aname in atom_names:
                bead_atoms.extend(aa_by_name.get(aname, []))
            if not bead_atoms:
                # Mapping references atoms not present in this residue; skip silently.
                continue
            coords = np.array([a.position for a in bead_atoms], dtype=float)
            xyz = coords.mean(axis=0)
            beads.append((bead_name, xyz))
        if not beads:
            raise ValueError(
                f"Mapping produced no beads for residue {ligand_residue.resname} {ligand_residue.resid}. "
                f"Atom names {list(aa_by_name.keys())} should be in the {map_file}."
            )
        # --- write CG PDB
        self.cgdir.mkdir(parents=True, exist_ok=True)
        resname = str(ligand_residue.resname)
        resid = int(ligand_residue.resid)
        outpdb = self.cgdir / f"ligand_{resname}_{resid}.pdb"
        # PDB formatting: keep it simple and GROMACS-friendly.
        # Use one residue, chain A, one atom per bead.
        with outpdb.open("w", encoding="utf-8") as w:
            for i, (bead_name, xyz) in enumerate(beads, start=1):
                x, y, z = xyz
                # atom name field is 4 chars, right/left aligned depends; simplest: right align
                atom_name = f"{bead_name:>4}"[:4]
                w.write(
                    f"ATOM  {i:5d} {atom_name} {resname:>3} A{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
            w.write("END\n")
        logger.info("Wrote mapped ligand CG PDB: %s", outpdb)

    def _merge_ligands_with(self, target: str, ligand: str) -> None:
        """Merge ligand ITP file into target ITP file using Topology class.
        
        Parameters
        ----------
        ligand_itp : Path
            Path to the ligand ITP file to merge
        target_name : str
            Name of the target ITP file (without .itp extension)
        add_bonded_restraints : list of tuple, optional
            List of bonded restraints to add (default: None)
        """
        target_itp = self.topdir / f"{target}.itp"
        ligand_itp = self.topdir / f"ligand_{ligand}.itp"
        if not target_itp.exists():
            raise FileNotFoundError(f"Target ITP file not found: {target_itp}")
        if not ligand_itp.exists():
            raise FileNotFoundError(f"Ligand ITP file not found: {ligand_itp}")
        logger.info(f"Merging {ligand_itp.name} into {target_itp.name} using Topology class")
        # Load both topologies
        target_topo = Topology.from_itp(target_itp)
        ligand_topo = Topology.from_itp(ligand_itp)
        ligand_key = f"ligand_{ligand}"
        for n in range(self.molecules[ligand_key]):
            target_topo += ligand_topo
        del self.molecules[ligand_key]
        target_topo.write_to_itp(target_itp)
        logger.info("Saved merged topology to %s", target_itp)
        
        # if add_bonded_restraints:
        #     for restraint in add_bonded_restraints:
        #         target_topo.bonds.append([
        #         (restraint[0], restraint[1]), 
        #         (6, restraint[2], restraint[3]), 
        #         "BONDED DISTANCE RESTRAINT",
        #         ])

    def martinize_ligands(
        self, 
        input_pdb: Path | None = None, 
        ligands: list[str] | None = None, 
        merge_with: str | None = None,
        out_itp: Path | None = None
    ) -> None:
        logger.info("Working on ligands...")
        if not input_pdb:
            input_pdb = self.inpdb
        u = mda.Universe(input_pdb)
        for ligand in ligands:
            ligand_residues = u.select_atoms(f"resname {ligand}").residues
            map_file = self.ligdir / ligand / f"{ligand.lower()}.map"
            itp_file = self.ligdir / ligand / f"{ligand.lower()}.itp"
            if not ligand_residues:
                raise ValueError(f"No residues found for ligand: {ligand}, check the ligand list or the PDB file.")
            for ligand_residue in ligand_residues:
                self._map_ligand(map_file, ligand_residue)
                self.molecules[f"ligand_{ligand}"] = len(ligand_residues)
                shutil.copy(itp_file, self.topdir / f"ligand_{ligand}.itp")
            if merge_with:
                self._merge_ligands_with(merge_with, ligand)


    def make_cg_structure(self, add_resolved_ions=False, **kwargs):
        """Merges coarse-grained PDB files into a single solute PDB file.
        
        Parameters
        ----------
        add_resolved_ions : bool, optional
            If True, adds resolved ions from ionpdb to the structure (default: False)
        **kwargs : dict
            Additional keyword arguments (currently unused)
        """
        logger.info("Merging CG PDB files into a single solute PDB...")
        cg_pdb_files = [p.name for p in self.cgdir.iterdir()]
        # cg_pdb_files = pdbtools.sort_uld(cg_pdb_files)
        cg_pdb_files = sort_for_gmx(cg_pdb_files)
        cg_pdb_files = [self.cgdir / fname for fname in cg_pdb_files]
        all_atoms = pdbtools.AtomList()
        for file in cg_pdb_files:
            atoms = pdbtools.pdb2atomlist(file)
            all_atoms.extend(atoms)
        # Add resolved ions if requested (already sorted by type in ionpdb)
        if add_resolved_ions:
            logger.info("Adding resolved ions to structure...")
            if self.ionpdb.exists():
                ion_atoms = pdbtools.pdb2atomlist(self.ionpdb)
                # Convert HETATM to ATOM records
                for atom in ion_atoms:
                    atom.record = 'ATOM'
                all_atoms.extend(ion_atoms)
                logger.info(f"Added {len(ion_atoms)} ions from {self.ionpdb}")
            else:
                logger.warning(f"Ion PDB file not found: {self.ionpdb}")
        all_atoms.renumber()
        all_atoms.write_pdb(self.solupdb)

    def make_cg_topology(self, add_resolved_ions=False, prefix="chain"):
        """Creates the system topology file by including all relevant ITP files and
        defining the system and molecule sections.

        Parameters
        ----------
            add_resolved_ions (bool, optional): If True, counts and includes resolved ions.
            prefix (str, optional): Prefix for ITP files to include (default: "chain").

        Writes the topology file (self.systop) with include directives and molecule counts.
        """
        logger.info("Writing system topology...")
        molecules = sort_for_gmx(self.molecules.keys())
        itp_files = [f"{molecule}.itp" for molecule in molecules]
        # itp_files = [p.name for p in self.topdir.glob(f'{prefix}*itp')]
        with self.systop.open("w") as f:
            # Include section
            f.write('#define GO_VIRT"\n')
            f.write("#define RUBBER_BANDS\n")
            f.write('#include "topol/martini_v3.0.0.itp"\n')
            f.write('#include "topol/martini_v3.0.0_rna.itp"\n')
            f.write('#include "topol/martini_ions.itp"\n')
            if any(p.name == "go_atomtypes.itp" for p in self.topdir.iterdir()):
                f.write('#include "topol/go_atomtypes.itp"\n')
                f.write('#include "topol/go_nbparams.itp"\n')
            f.write('#include "topol/martini_v3.0.0_solvents_v1.itp"\n')
            f.write('#include "topol/martini_v3.0.0_phospholipids_v1.itp"\n')
            f.write('#include "topol/martini_v3.0.0_ions_v1.itp"\n')
            f.write("\n")
            for filename in itp_files:
                f.write(f'#include "topol/{filename}"\n')
            # System name and molecule count
            f.write("\n[ system ]\n")
            f.write(f"Martini system for {self.sysname}\n")
            f.write("\n[molecules]\n")
            f.write("; name\t\tnumber\n")
            for molecule in molecules:
                count = self.molecules[molecule]
                if molecule.startswith("ligand_"):
                    molecule = molecule.replace("ligand_", "")
                f.write(f"{molecule}\t\t{count}\n")
            # for filename in itp_files:
            #     molecule_name = Path(filename).stem
            #     f.write(f"{molecule_name}\t\t1\n")
            # Add resolved ions if requested.
            if add_resolved_ions:
                ions = self.count_resolved_ions()
                for ion, count in ions.items():
                    if count > 0:
                        f.write(f"{ion}          {count}\n")

    def insert_membrane(self, **kwargs):
        """Insert CG lipid membrane using INSANE."""
        with cd(self.root):
            martini_tools.insert_membrane(**kwargs)

    def make_gro_file(self, d=1.25, bt="dodecahedron"):
        """Generates the final GROMACS GRO file from coarse-grained PDB files.

        Parameters
        ----------
            d (float, optional): Distance parameter for the editconf command (default: 1.25).
            bt (str, optional): Box type for the editconf command (default: "dodecahedron").

        Converts PDB files to GRO files, merges them, and adjusts the system box.
        """
        with cd(self.root):
            cg_pdb_files = pdbtools.sort_uld([p.name for p in self.cgdir.iterdir()])
            for file in cg_pdb_files:
                if file.endswith(".pdb"):
                    pdb_file = self.cgdir / file
                    # Replace suffix and directory component as in the original string manipulation
                    gro_file_str = str(pdb_file).replace(".pdb", ".gro").replace("cgpdb", "gro")
                    gro_file = Path(gro_file_str)
                    command = f"gmx_mpi editconf -f {pdb_file} -o {gro_file}"
                    sp.run(command.split(), check=True)
            # Merge all .gro files.
            gro_files = sorted([p.name for p in self.grodir.iterdir()])
            total_count = 0
            for filename in gro_files:
                if filename.endswith(".gro"):
                    filepath = self.grodir / filename
                    with filepath.open("r") as in_f:
                        lines = in_f.readlines()
                        atom_count = int(lines[1].strip())
                        total_count += atom_count
            with self.sysgro.open("w") as out_f:
                out_f.write(f"{self.sysname} \n")
                out_f.write(f"  {total_count}\n")
                for filename in gro_files:
                    if filename.endswith(".gro"):
                        filepath = self.grodir / filename
                        with filepath.open("r") as in_f:
                            lines = in_f.readlines()[2:-1]
                            for line in lines:
                                out_f.write(line)
                out_f.write("10.00000   10.00000   10.00000\n")
            command = f"gmx_mpi editconf -f {self.sysgro} -d {d} -bt {bt}  -o {self.sysgro}"
            sp.run(command.split(), check=True)

class MDRun(MDSystem):
    """Subclass of MDSystem for executing molecular dynamics (MD) simulations
    and performing post-processing analyses.
    """

    def __init__(self, sysdir, sysname, runname):
        """Initializes the MD run environment with additional directories for analysis.

        Parameters
        ----------
            sysdir (str): Base directory for the system.
            sysname (str): Name of the MD system.
            runname (str): Name for the MD run.
        """
        super().__init__(sysdir, sysname)
        self.runname = runname
        self.rundir = self.mddir / self.runname
        self.rmsdir = self.rundir / "rms_analysis"
        self.covdir = self.rundir / "cov_analysis"
        self.lrtdir = self.rundir / "lrt_analysis"
        self.cludir = self.rundir / "clusters"
        self.pngdir = self.rundir / "png"

    def prepare_files(self):
        """Creates necessary directories for the MD run and copies essential files."""
        self.mddir.mkdir(parents=True, exist_ok=True)
        self.rundir.mkdir(parents=True, exist_ok=True)
        self.rmsdir.mkdir(parents=True, exist_ok=True)
        self.cludir.mkdir(parents=True, exist_ok=True)
        self.covdir.mkdir(parents=True, exist_ok=True)
        self.lrtdir.mkdir(parents=True, exist_ok=True)
        self.pngdir.mkdir(parents=True, exist_ok=True)
        src = self.root / "atommass.dat"
        if src.exists():
            shutil.copy(src, self.rundir)

        
    def get_covmats(self, u, ag, dtype=np.float32, **kwargs):
        """Calculates covariance matrices by splitting the trajectory into chunks.

        Parameters
        ----------
            u (MDAnalysis.Universe, optional): Pre-loaded MDAnalysis Universe; if None, creates one.
            ag (AtomGroup, optional): Atom selection; if None, selects backbone atoms.
            sample_rate (int, optional): Sampling rate for positions.
            b (int, optional): Begin time/frame.
            e (int, optional): End time/frame.
            n (int, optional): Number of covariance matrices to calculate.
            outtag (str, optional): Tag prefix for output files.
        """
        b = kwargs.pop('b', None)
        e = kwargs.pop('e', None)
        n = kwargs.pop('n', 1)
        sample_rate = kwargs.pop('sample_rate', 1)
        outtag = kwargs.pop('outtag', 'covmat')
        logger.info("Calculating covariance matrices...")
        positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e)
        mdm.calc_and_save_covmats(positions, outdir=self.covdir, n=n, outtag=outtag, dtype=dtype)
        logger.info("Finished calculating covariance matrices!")

    def get_pertmats(self, intag="covmat", outtag="pertmat", iso=False, dtype=np.float32):
        """Calculates perturbation matrices from the covariance matrices.

        Parameters
        ----------
            intag (str, optional): Input file tag for covariance matrices.
            outtag (str, optional): Output file tag for perturbation matrices.
        """
        with cd(self.covdir):
            cov_files = [p.name for p in self.covdir.iterdir() if p.name.startswith(intag)]
            cov_files = sorted(cov_files)
            for cov_file in cov_files:
                logger.info("  Processing covariance matrix %s", cov_file)
                covmat = np.load(self.covdir / cov_file)
                logger.info("  Calculating perturbation matrix")
                if iso:
                    pertmat = mdm.perturbation_matrix_iso(covmat, dtype=dtype)
                else:
                    pertmat = mdm.perturbation_matrix(covmat, dtype=dtype)
                pert_file = cov_file.replace(intag, outtag)
                logger.info("  Saving perturbation matrix at %s", pert_file)
                np.save(self.covdir / pert_file, pertmat)
        logger.info("Finished calculating perturbation matrices!")

    def get_dfi(self, intag="pertmat", outtag="dfi", old_norm=False):
        """Calculates Dynamic Flexibility Index (DFI) from perturbation matrices.

        Parameters
        ----------
            intag (str, optional): Input file tag for perturbation matrices.
            outtag (str, optional): Output file tag for DFI values.
        """
        with cd(self.covdir):
            pert_files = [p.name for p in self.covdir.iterdir() if p.name.startswith(intag)]
            pert_files = sorted(pert_files)
            for pert_file in pert_files:
                logger.info("  Processing perturbation matrix %s", pert_file)
                pertmat = np.load(self.covdir / pert_file)
                logger.info("  Calculating DFI")
                dfi_val = mdm.dfi(pertmat, old_norm=old_norm)
                dfi_file = pert_file.replace(intag, outtag)
                dfi_file_path = self.covdir / dfi_file
                np.save(dfi_file_path, dfi_val)
                logger.info("  Saved DFI at %s", dfi_file_path)
        logger.info("Finished calculating DFIs!")

    def get_dci(self, intag="pertmat", outtag="dci", asym=False):
        """Calculates the Dynamic Coupling Index (DCI) from perturbation matrices.

        Parameters
        ----------
            intag (str, optional): Input file tag for perturbation matrices.
            outtag (str, optional): Output file tag for DCI values.
            asym (bool, optional): If True, calculates asymmetric DCI.
        """
        with cd(self.covdir):
            pert_files = [p.name for p in self.covdir.iterdir() if p.name.startswith(intag)]
            pert_files = sorted(pert_files)
            for pert_file in pert_files:
                logger.info("  Processing perturbation matrix %s", pert_file)
                pertmat = np.load(self.covdir / pert_file)
                logger.info("  Calculating DCI")
                dci_file = pert_file.replace(intag, outtag)
                dci_file_path = self.covdir / dci_file
                dci_val = mdm.dci(pertmat, asym=asym)
                np.save(dci_file_path, dci_val)
                logger.info("  Saved DCI at %s", dci_file_path)
        logger.info("Finished calculating DCIs!")

    def get_group_dci(self, groups, labels, **kwargs):
        """Calculates DCI between specified groups based on perturbation matrices.

        Parameters
        ----------
            groups (list): List of groups (atom indices or similar) to compare.
            labels (list): Corresponding labels for the groups.
            asym (bool, optional): If True, computes asymmetric group DCI.
        """
        intag  = kwargs.pop('intag', "pertmat")
        outtag  = kwargs.pop('outtag', "dci")
        asym  = kwargs.pop('asym', False)
        transpose = kwargs.pop('transpose', False)
        with cd(self.covdir):
            pert_files = sorted([f.name for f in Path.cwd().glob("pertmat*")])
            for pert_file in pert_files:
                logger.info("  Processing perturbation matrix %s", pert_file)
                pertmat = np.load(pert_file)
                logger.info("  Calculating group DCI")
                dcis = mdm.group_molecule_dci(pertmat, groups=groups, asym=asym, transpose=transpose)
                for dci_val, group, group_id in zip(dcis, groups, labels):
                    dci_file = pert_file.replace("pertmat", f"g{outtag}_{group_id}")
                    dci_file_path = self.covdir / dci_file
                    np.save(dci_file_path, dci_val)
                    logger.info("  Saved group DCI at %s", dci_file_path)
                ch_dci_file = pert_file.replace("pertmat", f"gg{outtag}")
                ch_dci_file_path = self.covdir / ch_dci_file
                ch_dci = mdm.group_group_dci(pertmat, groups=groups, asym=asym)
                np.save(ch_dci_file_path, ch_dci)
                logger.info("  Saved group-group DCI at %s", ch_dci_file_path)
        logger.info("Finished calculating group DCIs!")


def sort_for_gmx(data):
    return sorted(data, key=lambda x: (
    0 if x.split('_')[1][0].isupper() else 1 if x.split('_')[1][0].islower() else 2,
    x.split('_')[1]
))    