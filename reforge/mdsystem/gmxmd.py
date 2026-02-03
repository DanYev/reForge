"""File: gmxmd.py

Description:
    This module provides classes and functions for setting up, running, and
    analyzing molecular dynamics (MD) simulations using GROMACS. The main
    classes include:

      - GmxSystem: Provides methods to prepare simulation files, process PDB
        files, run GROMACS commands, and perform various analyses on MD data.
      - MDRun: A subclass of GmxSystem dedicated to executing MD simulations and
        performing post-processing tasks (e.g., RMSF, RMSD, covariance analysis).

Usage:
    Import this module and instantiate the GmxSystem or MDRun classes to set up
    and run your MD simulations.

Requirements:
    - Python 3.x
    - MDAnalysis
    - NumPy
    - Pandas
    - GROMACS (with CLI tools such as gmx_mpi, gmx, etc.)
    - The reforge package and its dependencies

Author: DY
Date: 2025-02-27
"""

import importlib.resources
import logging
import os
from pathlib import Path
import shutil
import subprocess as sp
from reforge import cli, pdbtools, io
from reforge.utils import cd, clean_dir
from reforge.mdsystem.mdsystem import MDSystem, MDRun, MartiniMixin

logger = logging.getLogger(__name__)

################################################################################
# GMX system class
################################################################################

class GmxSystem(MDSystem, MartiniMixin):
    """Class to set up and analyze protein-nucleotide-lipid systems for MD
    simulations using GROMACS.

    Most attributes are paths to files and directories needed to set up
    and run the MD simulation.
    """
    MMDPDIR = importlib.resources.files("reforge") / "martini" / "datdir" / "mdp"

    def __init__(self, sysdir, sysname):
        """Initializes the MD system with required directories and file paths.

        Parameters
        ----------
            sysdir (str): Base directory for the system files.
            sysname (str): Name of the MD system.

        Sets up paths for various files required for coarse-grained MD simulation.
        """
        super().__init__(sysdir, sysname)
        self.sysndx = self.root / "system.ndx"
        self.mdpdir = self.root / "mdp"
        self.grodir = self.root / "gro"

    def gmx(self, command="-h", clinput=None, clean_wdir=True, **kwargs):
        """Executes a GROMACS command from the system's root directory. 

        Parameters
        ----------
            command (str): The GROMACS command to run (default: "-h").
            clinput (str, optional): Input to pass to the command's stdin.
            clean_wdir (bool, optional): If True, cleans the working directory after execution.
            kwargs: Additional keyword arguments to pass to gromacs.
        """
        with cd(self.root):
            cli.gmx(command, clinput=clinput, **kwargs)
            if clean_wdir:
                clean_dir()

    def prepare_files(self, *args, **kwargs):
        """Extension for GROMACS system"""
        super().prepare_files(*args, **kwargs)
        self.topdir.mkdir(parents=True, exist_ok=True)
        self.mdpdir.mkdir(parents=True, exist_ok=True)
        # .mdp files
        for file in self.MMDPDIR.iterdir():
            if file.name.endswith(".mdp"):
                outpath = self.mdpdir / file.name
                shutil.copy(file, outpath)

    def make_box(self, **kwargs):
        """Sets up simulation box with GROMACS editconf command.

        Parameters
        ----------
            kwargs: Additional keyword arguments for the GROMACS editconf command. Defaults:
                - d: Distance parameter (default: 1.0)
                - bt: Box type (default: "dodecahedron").
        """
        kwargs.setdefault("d", 1.0)
        kwargs.setdefault("bt", "dodecahedron")
        with cd(self.root):
            cli.gmx("editconf", f=self.solupdb, o=self.solupdb, **kwargs)

    def solvate(self, **kwargs):
        """Solvates the system using GROMACS solvate command.

        Parameters
        ----------
            kwargs: Additional parameters for the solvate command. Defaults:
                - cp: "solute.pdb"
                - cs: "water.gro"
        """
        kwargs.setdefault("cp", "solute.pdb")
        kwargs.setdefault("cs", "water.gro")
        self.gmx("solvate", p=self.systop, o=self.syspdb, **kwargs)

    def add_bulk_ions(self, solvent="W", **kwargs):
        """ Adds bulk ions to neutralize the system using GROMACS genion.

        Parameters
        ----------
        solvent (str, optional): 
            Solvent residue name (default: "W").
        kwargs (dict):
            Additional parameters for genion. Defaults include:
            - conc: 0.15
            - pname: "NA"
            - nname: "CL"
            - neutral: ""
        """
        kwargs.setdefault("conc", 0.15)
        kwargs.setdefault("pname", "NA")
        kwargs.setdefault("nname", "CL")
        kwargs.setdefault("neutral", "")
        self.gmx("grompp",
            f=self.mdpdir / "ions.mdp", c=self.syspdb, p=self.systop, o="ions.tpr")
        self.gmx("genion", clinput=f"{solvent}\n", # clinput=f"{solvent}\n",
            s="ions.tpr", p=self.systop, o=self.syspdb, **kwargs)
        self.gmx("editconf", f=self.syspdb, o=self.sysgro)
        clean_dir(self.root, "ions.tpr")

    def make_system_ndx(self, backbone_atoms=("CA", "P", "C1'"), water_resname='W'):
        """Creates an index (NDX) file for the system, separating solute, 
        backbone, solvent, and individual chains.

        Parameters
        ----------
            backbone_atoms : list, optional
                List of atom names to include in the backbone (default: ["CA", "P", "C1'"]).
        """
        logger.info("Making index file from %s...", self.syspdb)
        system = pdbtools.pdb2atomlist(self.syspdb)
        solute = pdbtools.pdb2atomlist(self.solupdb)
        solvent = AtomList(system[len(solute):])
        backbone = solute.mask(backbone_atoms, mode="name")
        not_water = system.mask_out(water_resname, mode='resname')
        system.write_ndx(self.sysndx, header="[ System ]", append=False, wrap=15)
        solute.write_ndx(self.sysndx, header="[ Solute ]", append=True, wrap=15)
        backbone.write_ndx(self.sysndx, header="[ Backbone ]", append=True, wrap=15)
        solvent.write_ndx(self.sysndx, header="[ Solvent ]", append=True, wrap=15)
        not_water.write_ndx(self.sysndx, header="[ Not_Water ]", append=True, wrap=15)
        chids = self.chains
        for chid in chids:
            chain = solute.mask(chid, mode="chid")
            chain.write_ndx(self.sysndx, header=f"[ chain_{chid} ]", append=True, wrap=15)
        segids = set(solute.segids)
        for segid in segids:
            segment = solute.mask(segid, mode="segid")
            segment.write_ndx(self.sysndx, header=f"[ seg_{segid} ]", append=True, wrap=15)
        logger.info("Written index to %s", self.sysndx)

    def initmd(self, runname):
        """Initializes a new GMX MD run.

        Parameters
        ----------
            runname (str): Name for the MD run.

        Returns:
            MDRun: An instance of the MDRun class for the specified run.
        """
        mdrun = MDRun(self.sysdir, self.sysname, runname)
        return mdrun


class GmxRun(MDRun):
    """Subclass of MDRun for running MD simulations with GROMACS."""

    def __init__(self, sysdir, sysname, runname):
        """Initializes the MD run environment with additional directories for analysis.

        Parameters
        ----------
            sysdir (str): Base directory for the system.
            sysname (str): Name of the MD system.
            runname (str): Name for the MD run.
        """
        super().__init__(sysdir, sysname, runname)
        self.sysgro = self.root / "system.gro"
        self.systop = self.root / "system.top"
        self.sysndx = self.root / "system.ndx"
        self.mdpdir = self.root / "mdp"
        self.str = self.rundir / "mdc.pdb"  # Structure file
        self.trj = self.rundir / "mdc.trr"  # Trajectory file
        self.trj = self.trj if self.trj.exists() else self.rundir / "mdc.xtc"

    def gmx(self, command="-h", clinput=None, clean_wdir=True, **kwargs):
        """Executes a GROMACS command from the run's root directory. 

        Parameters
        ----------
            command (str): The GROMACS command to run (default: "-h").
            clinput (str, optional): Input to pass to the command's stdin.
            clean_wdir (bool, optional): If True, cleans the working directory after execution.
            kwargs: Additional keyword arguments to pass to gromacs.
        """
        with cd(self.rundir):
            cli.gmx(command, clinput=clinput, **kwargs)
            if clean_wdir:
                clean_dir()

    def empp(self, **kwargs):
        """Prepares the energy minimization run using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to .mdp file.
                - c: Structure file.
                - r: Reference structure.
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("em.tpr").
        """
        kwargs.setdefault("f", self.mdpdir / "em_cg.mdp")
        kwargs.setdefault("c", self.sysgro)
        kwargs.setdefault("r", self.sysgro)
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "em.tpr")
        self.gmx('grompp', **kwargs)

    def hupp(self, **kwargs):
        """Prepares the heat-up phase using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to .mdp file
                - c: Starting structure ("em.gro").
                - r: Reference structure ("em.gro").
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("hu.tpr").
        """
        kwargs.setdefault("f", self.mdpdir / "hu_cg.mdp")
        kwargs.setdefault("c", "em.gro")
        kwargs.setdefault("r", "em.gro")
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "hu.tpr")
        self.gmx('grompp', **kwargs)

    def eqpp(self, **kwargs):
        """Prepares the equilibration phase using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to .mdp file.
                - c: Starting structure ("hu.gro").
                - r: Reference structure ("hu.gro").
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("eq.tpr").
        """
        kwargs.setdefault("f", self.mdpdir / "eq_cg.mdp")
        kwargs.setdefault("c", "hu.gro")
        kwargs.setdefault("r", "hu.gro")
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "eq.tpr")
        self.gmx('grompp', **kwargs)

    def mdpp(self, **kwargs):
        """Prepares the production MD run using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to .mdp file.
                - c: Starting structure ("eq.gro").
                - r: Reference structure ("eq.gro").
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("md.tpr").
        """
        kwargs.setdefault("f", self.mdpdir / "md_cg.mdp")
        kwargs.setdefault("c", "eq.gro")
        kwargs.setdefault("r", "eq.gro")
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "md.tpr")
        self.gmx('grompp', **kwargs)

    def mdrun(self, **kwargs):
        """Executes the production MD run using GROMACS mdrun.

        Parameters
        ----------
            kwargs: Additional parameters for mdrun. Defaults include:
                - deffnm: "md"
                - nsteps: "-2"
                - ntomp: "8"
        """
        kwargs.setdefault("deffnm", "md")
        kwargs.setdefault("nsteps", "-2")
        kwargs.setdefault("ntomp", "8")
        kwargs.setdefault("pin", "on")
        kwargs.setdefault("pinstride", "1")
        self.gmx('mdrun', **kwargs)

    def trjconv(self, clinput=None, **kwargs):
        """Converts trajectories using GROMACS trjconv.

        Parameters
        ----------
            clinput (str, optional): Input to be passed to trjconv.
            kwargs: Additional parameters for trjconv.
        """
        self.gmx('trjconv', clinput=clinput, **kwargs)

    def convert_tpr(self, clinput=None, **kwargs):
        """Converts TPR files using GROMACS convert-tpr."""
        self.gmx('convert-tpr', clinput=clinput, **kwargs)   

    def rmsf(self, clinput=None, **kwargs):
        """Calculates RMSF using GROMACS rmsf.

        Parameters
        ----------
            clinput (str, optional): Input for the rmsf command.
            kwargs: Additional parameters for rmsf. Defaults include:
                - s: Structure file.
                - f: Trajectory file.
                - o: Output xvg file.
        """
        xvg_file = self.rmsdir / "rmsf.xvg"
        npy_file = self.rmsdir / "rmsf.npy"
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("o", xvg_file)
        kwargs.setdefault("xvg", "none")
        self.gmx('rmsf', clinput=clinput, **kwargs)
        io.xvg2npy(xvg_file, npy_file, usecols=[1])

    def rmsd(self, clinput=None, **kwargs):
        """Calculates RMSD using GROMACS rms.

        Parameters
        ----------
            clinput (str, optional): Input for the rms command.
            kwargs: Additional parameters for rms. Defaults include:
                - s: Structure file.
                - f: Trajectory file.
                - o: Output xvg file.
        """
        xvg_file = self.rmsdir / "rmsd.xvg"
        npy_file = self.rmsdir / "rmsd.npy"
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("o", xvg_file)
        kwargs.setdefault("xvg", "none")
        self.gmx('rms', clinput=clinput, **kwargs)
        io.xvg2npy(xvg_file, npy_file, usecols=[0, 1])

    def rdf(self, clinput=None, **kwargs):
        """Calculates the radial distribution function using GROMACS rdf.

        Parameters
        ----------
            clinput (str, optional): Input for the rdf command.
            kwargs: Additional parameters for rdf. Defaults include:
                - f: Trajectory file.
                - s: Structure file.
                - n: Index file.
        """
        kwargs.setdefault("f", "mdc.xtc")
        kwargs.setdefault("s", "mdc.pdb")
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("xvg", "none")
        self.gmx('rdf', clinput=clinput, **kwargs)

    def cluster(self, clinput=None, **kwargs):
        """Performs clustering using GROMACS cluster.

        Parameters
        ----------
            clinput (str, optional): Input for the clustering command.
            kwargs: Additional parameters for cluster.
        """
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        with cd(self.cludir):
            cli.gmx('cluster', clinput=clinput, **kwargs)

    def extract_cluster(self, clinput=None, **kwargs):
        """Extracts frames belonging to a cluster using GROMACS extract-cluster.

        Parameters
        ----------
            clinput (str, optional): Input for the extract-cluster command.
            kwargs: Additional parameters for extract-cluster. Defaults include:
            - clusters: "cluster.ndx"
        """
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("clusters", "cluster.ndx")
        with cd(self.cludir):
            cli.gmx('extract-cluster', clinput=clinput, **kwargs)

    def covar(self, clinput=None, **kwargs):
        """Calculates and diagonalizes the covariance matrix using GROMACS covar.

        Parameters
        ----------
            clinput (str, optional): Input for the covar command.
            kwargs: Additional parameters for covar. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - n: Index file.
        """
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("s", self.str)
        with cd(self.covdir):
            cli.gmx('covar', clinput=clinput, **kwargs)

    def anaeig(self, clinput=None, **kwargs):
        r"""Analyzes eigenvectors using GROMACS anaeig.

        Parameters
        ----------
            clinput (str, optional): Input for the anaeig command.
            kwargs: Additional parameters for anaeig. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - v: Output eigenvector file.
        """
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("v", "eigenvec.trr")
        with cd(self.covdir):
            cli.gmx('anaeig', clinput=clinput, **kwargs)

    def make_edi(self, clinput=None, **kwargs):
        """Prepares files for essential dynamics analysis using GROMACS make-edi.

        Parameters
        ----------
            clinput (str, optional): Input for the make-edi command.
            kwargs: Additional parameters for make-edi. Defaults include:
            - f: Eigenvector file.
            - s: Structure file.
        """
        kwargs.setdefault("f", "eigenvec.trr")
        kwargs.setdefault("s", self.str)
        with cd(self.covdir):
            cli.gmx('make_edi', clinput=clinput, **kwargs)

    def get_rmsf_by_chain(self, **kwargs):
        """Calculates RMSF for each chain in the system using GROMACS rmsf.

        Parameters
        ----------
            kwargs: Additional parameters for the rmsf command. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - n: Index file.
            - res: Whether to output per-residue RMSF (default: "no").
            - fit: Whether to fit the trajectory (default: "yes").
        """
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("res", "yes")
        kwargs.setdefault("fit", "yes")
        kwargs.setdefault("xvg", "none")
        for idx, chain in enumerate(self.chains):
            idx = idx + 5
            xvg_file = self.rmsdir / f"crmsf_{chain}.xvg"
            npy_file = self.rmsdir / f"crmsf_{chain}.npy"
            self.gmx('rmsf', clinput=f"{idx}\n{idx}\n", o=xvg_file, **kwargs)
            io.xvg2npy(xvg_file, npy_file, usecols=[1])

    def get_rmsd_by_chain(self, **kwargs):
        """Calculates RMSD for each chain in the system using GROMACS rmsd.

        Parameters
        ----------
            kwargs: Additional parameters for the rmsd command. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - n: Index file.
        """
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("fit", "rot+trans")
        kwargs.setdefault("xvg", "none")
        for idx, chain in enumerate(self.chains):
            idx = idx + 5
            xvg_file = self.rmsdir / f"crmsd_{chain}.xvg"
            npy_file = self.rmsdir / f"crmsd_{chain}.npy"
            self.gmx('rms', clinput=f"{idx}\n{idx}\n", o=xvg_file, **kwargs)
            io.xvg2npy(xvg_file, npy_file, usecols=[1])


#######################
# Detect available CPU cores and set ntomp
def get_ntomp():
    """Detect number of available CPU cores for OpenMP threads"""
    # Check if ntomp is set via environment variable
    env_ntomp = os.environ.get('OMP_NUM_THREADS')
    if env_ntomp:
        try:
            ntomp_val = int(env_ntomp)
            logger.info(f"Using OMP_NUM_THREADS={ntomp_val} from environment")
            return ntomp_val
        except ValueError:
            logger.warning(f"Invalid OMP_NUM_THREADS value: {env_ntomp}")
    
    # Check SLURM environment variables first (most reliable for HPC)
    # Calculate total allocated cores: ntasks × cpus-per-task
    slurm_ntasks = os.environ.get('SLURM_NTASKS')
    slurm_cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
    
    if slurm_ntasks and slurm_cpus_per_task:
        try:
            ntasks = int(slurm_ntasks)
            cpus_per_task = int(slurm_cpus_per_task)
            total_cores = ntasks * cpus_per_task
            logger.info(f"SLURM allocation: {ntasks} tasks × {cpus_per_task} cpus/task = {total_cores} total cores")
            return total_cores
        except ValueError as e:
            logger.warning(f"Invalid SLURM values - ntasks: {slurm_ntasks}, cpus_per_task: {slurm_cpus_per_task}")
    
    # Fallback to individual SLURM variables
    if slurm_cpus_per_task:
        try:
            ntomp_val = int(slurm_cpus_per_task)
            logger.info(f"Using SLURM_CPUS_PER_TASK={ntomp_val} (no ntasks found)")
            return ntomp_val
        except ValueError:
            logger.warning(f"Invalid SLURM_CPUS_PER_TASK value: {slurm_cpus_per_task}")
    
    # Check other SLURM variables
    slurm_nprocs = os.environ.get('SLURM_NPROCS')
    if slurm_nprocs:
        try:
            ntomp_val = int(slurm_nprocs)
            logger.info(f"Using SLURM_NPROCS={ntomp_val} from SLURM allocation")
            return ntomp_val
        except ValueError:
            logger.warning(f"Invalid SLURM_NPROCS value: {slurm_nprocs}")
    
    # Check PBS/Torque environment variables
    pbs_ncpus = os.environ.get('PBS_NCPUS') or os.environ.get('NCPUS')
    if pbs_ncpus:
        try:
            ntomp_val = int(pbs_ncpus)
            logger.info(f"Using PBS_NCPUS/NCPUS={ntomp_val} from PBS allocation")
            return ntomp_val
        except ValueError:
            logger.warning(f"Invalid PBS_NCPUS/NCPUS value: {pbs_ncpus}")
    
    # Try to get available cores from cgroup limits (Docker/container environments)
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            ntomp_val = max(1, quota // period)
            logger.info(f"Using cgroup CPU quota: {ntomp_val} cores available")
            return ntomp_val
    except (FileNotFoundError, ValueError, PermissionError):
        pass  # cgroup not available or not accessible
    
    # Fallback: use total system cores but warn about potential oversubscription
    try:
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        logger.warning(f"No job scheduler detected. Found {total_cores} total system cores.")
        logger.warning("This may cause oversubscription on shared systems!")
        logger.warning("Consider setting OMP_NUM_THREADS explicitly for your allocation.")
        # Default to 1 core to be safe on shared systems
        ntomp_val = 1 
        logger.info(f"Defaulting to ntomp={ntomp_val} for safety on shared systems")
        return ntomp_val
    except Exception as e:
        logger.warning(f"Could not detect CPU cores: {e}, defaulting to ntomp=1")
        return 1
        

def sort_for_gmx(data):
    return sorted(data, key=lambda x: (
    0 if x.split('_')[1][0].isupper() else 1 if x.split('_')[1][0].islower() else 2,
    x.split('_')[1]
))    