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
import numpy as np
import logging
from openmm.app import PDBFile
from pdbfixer.pdbfixer import PDBFixer
from reforge import cli, mdm, pdbtools, io
from reforge.utils import cd, clean_dir
from reforge.martini import getgo, martini_tools

logger = logging.getLogger(__name__)

################################################################################
# GMX system class
################################################################################

class MDSystem:
    """
    Most attributes are paths to files and directories needed to set up
    and run the MD simulation.
    """
    MDATDIR = importlib.resources.files("reforge") / "martini" / "datdir"
    MITPDIR = importlib.resources.files("reforge") / "martini" / "itp"
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
        self.ionpdb = self.iondir / "ions.pdb"
        self.topdir = self.root / "topol"
        self.mapdir = self.root / "maps"
        self.cgdir = self.root / "cgpdb"
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
            self.cgdir.mkdir(parents=True, exist_ok=True)
            self.mapdir.mkdir(parents=True, exist_ok=True)
            self.topdir.mkdir(parents=True, exist_ok=True)
            # Copy water.gro and atommass.dat from master data directory
            shutil.copy(self.MDATDIR / "water.gro", self.root)
            shutil.copy(self.MDATDIR / "atommass.dat", self.root)
            # Copy .itp files from master ITP directory
            for file in self.MITPDIR.iterdir():
                if file.name.endswith(".itp"):
                    outpath = self.topdir / file.name
                    shutil.copy(file, outpath)

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
        logger.info(f"Found {len(pdb_files)} protein PDB files to process")
        itp_files = [f.replace("pdb", "itp") for f in pdb_files]
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
        logger.info(f"Found {len(pdb_files)} protein PDB files to process")
        itp_files = [f.replace("pdb", "itp") for f in pdb_files]
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
        logger.info(f"Found {len(pdb_files)} RNA PDB files to process")
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

    def insert_membrane(self, **kwargs):
        """Insert CG lipid membrane using INSANE."""
        with cd(self.root):
            martini_tools.insert_membrane(**kwargs)

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
        b = kwargs.pop('b', 50000)
        e = kwargs.pop('e', 1000000)
        n = kwargs.pop('n', 10)
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

