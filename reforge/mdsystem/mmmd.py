"""File: mmmd.py

Description:
    This module provides classes and functions for setting up, running, and
    analyzing molecular dynamics (MD) simulations using GROMACS. The main
    classes include:

      - Mm: Provides methods to prepare simulation files, process PDB
        files, run GROMACS commands, and perform various analyses on MD data.
      - MmRun: A subclass of GmxSystem dedicated to executing MD simulations and
        performing post-processing tasks (e.g., RMSF, RMSD, covariance analysis).

Usage:
    Import this module and instantiate the MmSystem or MmRun classes to set up
    and run your MD simulations.

Author: DY
"""

import logging
import os
import sys
import shutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.util import get_ext
from MDAnalysis.lib.mdamath import triclinic_box
import openmm as mm
from openmm import app, unit
import logging
from pdbfixer.pdbfixer import PDBFixer
from reforge import cli, io
from reforge.utils import cd, clean_dir, timeit, memprofit
from reforge.mdsystem.mdsystem import MDSystem, MDRun

logger = logging.getLogger(__name__)


class MmSystem(MDSystem):
    """Subclass for OpenMM"""

    def __init__(self, sysdir, sysname, **kwargs):
        """Initialize the MD system with required directories and file paths."""
        super().__init__(sysdir, sysname)
        
    def prepare_files(self, *args, **kwargs):
        """Extension for OpenMM system"""
        super().prepare_files(*args, **kwargs)

    def clean_pdb(self, pdb_file, add_missing_atoms=False, add_hydrogens=False, pH=7.0, **kwargs):
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
        logger.info("Cleaning the PDB")
        logger.info(f"Processing {pdb_file}")
        pdb = PDBFixer(filename=str(pdb_file))
        logger.info("Removing heterogens and checking for missing residues...")
        pdb.removeHeterogens(False)
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
            app.PDBFile.writeFile(topology, positions, outfile)
        logger.info(f"Written cleaned PDB to {self.inpdb}")


################################################################################
# MmRun class
################################################################################

class MmRun(MDRun):

    def __init__(self, sysdir, sysname, runname):
        """Initializes the MD run environment with additional directories for analysis.

        Parameters
        ----------
            sysdir (str): Base directory for the system.
            sysname (str): Name of the MD system.
            runname (str): Name for the MD run.
        """
        super().__init__(sysdir, sysname, runname)
        self.sysxml = self.root / "system.xml"
        self.systop = self.root / "system.top"
        self.sysndx = self.root / "system.ndx"
        self.mdpdir = self.root / "mdp"
        self.str = self.rundir / "mdc.pdb"  # Structure file
        self.trj = self.rundir / "mdc.trr"  # Trajectory file
        self.trj = self.trj if self.trj.exists() else self.rundir / "mdc.xtc"
        
    def build_modeller(self):
        """Generate a modeller object from the system PDB file.

        Returns
        -------
        openmm.app.Modeller
            The modeller object initialized with the system topology and positions.
        """
        pdb_file = app.PDBFile(str(self.syspdb))
        modeller_obj = app.Modeller(pdb_file.topology, pdb_file.positions)
        return modeller_obj

    def simulation(self, modeller_obj, integrator):
        """Initialize and return a simulation object for the MD run.

        Parameters
        ----------
        modeller_obj : openmm.app.Modeller
            The modeller object with prepared topology and positions.
        integrator : openmm.Integrator
            The integrator for the simulation.

        Returns
        -------
        openmm.app.Simulation
            The initialized simulation object.
        """
        simulation_obj = app.Simulation(modeller_obj.topology, str(self.sysxml), integrator)
        simulation_obj.context.setPositions(modeller_obj.positions)
        return simulation_obj

    def save_state(self, simulation_obj, file_prefix="sim"):
        """Save the current simulation state to XML and PDB files.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        file_prefix : str, optional
            Prefix for the state files (default: 'sim').

        Notes
        -----
        Saves the simulation state as an XML file and writes the current positions to a PDB file.
        """
        pdb_file = os.path.join(self.rundir, file_prefix + "_state.pdb")
        xml_file = os.path.join(self.rundir, file_prefix + ".xml")
        simulation_obj.saveState(xml_file)
        state = simulation_obj.context.getState(getPositions=True)
        positions = state.getPositions()
        with open(pdb_file, "w", encoding="utf-8") as file:
            app.PDBFile.writeFile(simulation_obj.topology, positions, file, keepIds=True)

    def get_std_reporters(self, append, prefix='md', nlog=10000, nchk=10000, **kwargs):
        kwargs.setdefault("step", True)
        kwargs.setdefault("time", True)
        kwargs.setdefault("elapsedTime", True)
        kwargs.setdefault("potentialEnergy", True)
        kwargs.setdefault("temperature", True)
        kwargs.setdefault("density", False)
        log_file = os.path.join(self.rundir, f"{prefix}.log")
        xml_file = os.path.join(self.rundir, f"{prefix}.xml")
        log_reporter = app.StateDataReporter(log_file, nlog, append=append, **kwargs)
        stderr_reporter = app.StateDataReporter(sys.stderr, nlog, append=False, **kwargs)
        xml_reporter = app.CheckpointReporter(xml_file, nchk, writeState=True)
        reporters = [xml_reporter, log_reporter, stderr_reporter]
        return reporters

    @memprofit(level=logging.INFO)
    @timeit(level=logging.INFO, unit='auto')
    def em(self, simulation, tolerance=10, max_iterations=1000):
        """Perform energy minimization for the simulation.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation object.
        tolerance [kJ/nm/mol] : float, optional
            RMSF force tolerance for energy minimization (default: 10).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).

        Notes
        -----
        Minimizes the energy, saves the minimized state, and logs progress.
        """
        logger.info("Minimizing energy...")
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)
        self.save_state(simulation, "em")
        logger.info("Minimization completed.")

    @memprofit(level=logging.INFO)
    @timeit(level=logging.INFO, unit='auto')
    def hu(self, simulation, temperature, 
            n_cycles=100, steps_per_cycle=100, **kwargs):
        """Run equilibration.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of steps for equilibration (default: 10000).
        **kwargs : dict, optional
            Additional keyword arguments.
        Notes
        -----
        Loads the minimized state, runs heatup, and saves the final state.
        """
        logger.info("Heating up the system...")
        in_xml = os.path.join(self.rundir, "em.xml")
        simulation.loadState(in_xml)
        for i in range(n_cycles):
            simulation.integrator.setTemperature(temperature*i/n_cycles)
            simulation.step(steps_per_cycle)
        self.save_state(simulation, "hu")
        logger.info("Heatup completed.")

    @memprofit(level=logging.INFO)
    @timeit(level=logging.INFO, unit='auto')
    def eq(self, simulation, 
            n_cycles=100, steps_per_cycle=100, **kwargs):
        """Run equilibration.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of steps for equilibration (default: 10000).
        **kwargs : dict, optional
            Additional keyword arguments.

        Notes
        -----
        Loads the heated state, runs equilibration, and saves the equilibrated state.
        """
        logger.info("Starting equilibration...")
        in_xml = str(self.rundir / "hu.xml")
        simulation.loadState(in_xml)
        enum = enumerate(simulation.system.getForces()) 
        idx, bb_restraint = [(idx, f) for idx, f in enum if f.getName() == 'BackboneRestraint'][0]
        fc = bb_restraint.getGlobalParameterDefaultValue(0)
        fcname = bb_restraint.getGlobalParameterName(0)
        for i in range(n_cycles):
            simulation.step(steps_per_cycle)
            new_fc = fc * (1 - (i + 1) / n_cycles)
            simulation.context.setParameter(fcname, new_fc)
        # Remove the restraints and reinitialize context - we need to get rib of bb_fc
        simulation.system.removeForce(idx)
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        simulation.context.reinitialize(preserveState=False)
        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocities(state.getVelocities())
        simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        simulation.saveState(str(self.rundir / "eq.xml"))
        logger.info("Equilibration completed.")

    @memprofit(level=logging.INFO)
    @timeit(level=logging.INFO, unit='auto')
    def md(self, simulation, time=1000, nsteps=None, **kwargs):
        """Run production MD simulation.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of production steps (default: 100000).
        **kwargs : dict, optional
            Additional keyword arguments.

        Notes
        -----
        Loads the equilibrated state, runs production, and saves the final simulation state.
        """
        logger.info("Production run...")
        in_xml = self.rundir / "eq.xml"
        simulation.loadState(str(in_xml))
        if not nsteps:
            dt = simulation.integrator.getStepSize()
            logger.info(f"Total MD time: %s", time)
            nsteps = int(time / dt)
        logger.info(f"Number of steps left: %s", nsteps)
        out_xml = self.rundir / "md.xml"
        simulation.saveState(str(out_xml))
        logger.info("Production completed.")

    @memprofit(level=logging.INFO)
    @timeit(level=logging.INFO, unit='auto')
    def extend(self, simulation, until_time=1000, nsteps=None, **kwargs):
        """Extend production MD simulation"""
        logger.info("Extending run...")
        in_xml = os.path.join(self.rundir, "md.xml")
        simulation.loadState(in_xml)
        state = simulation.context.getState()
        curr_time = state.getTime()
        if not nsteps:
            dt = simulation.integrator.getStepSize()
            logger.info(f"Current time: %s", curr_time)
            logger.info(f"Extend until: %s", until_time)
            nsteps = int((until_time - curr_time) / dt)
            if nsteps <= 0:
                logger.warning("Current simulation is longer than UNTIL_TIME, exiting!")
                sys.exit(0) 
        logger.info(f"Number of steps left: %s", nsteps)
        simulation.step(nsteps)
        self.save_state(simulation, "md")
        logger.info("Production completed.")


class MmReporter(object):
    """Most of this code is adapted from https://github.com/sef43/openmm-mdanalysis-reporter.
    MDAReporter outputs a series of frames from a Simulation to any file format supported by MDAnalysis.
    To use it, create a MDAReporter, then add it to the Simulation's list of reporters.
    """
    def __init__(
        self,
        file,
        reportInterval,
        enforcePeriodicBox=None,
        selection: str = None,
        writer_kwargs: dict = None
    ):
        """Create a MDAReporter.
        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        selection : str
            MDAnalysis selection string (https://docs.mdanalysis.org/stable/documentation_pages/selections.html)
            which will be passed to MDAnalysis.Universe.select_atoms. If None (the default), all atoms will we selected.
        writer_kwargs : dict
            Additional keyword arguments to pass to the MDAnalysis.Writer object.
            
        """
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self._filename = file
        self._topology = None
        self._nextModel = 0
        self._mdaUniverse = None
        self._mdaWriter = None
        self._selection = selection
        self._atomGroup = None
        self._writer_kwargs = writer_kwargs or {}

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        root, ext = get_ext(self._filename) 
        if ext in ["trr"]:
            positions, velocities, forces = True, True, True
        else:
            positions, velocities, forces = True, False, False
        return steps, positions, velocities, forces, False, self._enforcePeriodicBox

    def report(self, simulation, state):
        """Generate a report.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if self._nextModel == 0:
            self._topology = simulation.topology
            dt = simulation.integrator.getStepSize()*self._reportInterval
            self._mdaUniverse = mda.Universe(
                simulation.topology,
                simulation,
                topology_format='OPENMMTOPOLOGY',
                format='OPENMMSIMULATION',
                dt=dt
            )
            if self._selection is not None:
                self._atomGroup = self._mdaUniverse.select_atoms(self._selection)
            else:
                self._atomGroup = self._mdaUniverse.atoms
            self._mdaWriter = mda.Writer(
                self._filename,
                n_atoms=len(self._atomGroup),
                **self._writer_kwargs
            )
            self._nextModel += 1
        # update the positions and velocities if present, convert from OpenMM nm to MDAnalysis angstroms
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        self._mdaUniverse.atoms.positions = positions
        save_velocities = self.describeNextReport(simulation)[2]
        if save_velocities:
            velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom/unit.picosecond)
            self._mdaUniverse.atoms.velocities = velocities
        # update box vectors
        boxVectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        self._mdaUniverse.dimensions = triclinic_box(*boxVectors)
        self._mdaUniverse.dimensions[:3] = self._sanitize_box_angles(self._mdaUniverse.dimensions[:3])
        # write to the trajectory file
        self._mdaWriter.write(self._atomGroup)
        self._nextModel += 1

    def __del__(self):
        if self._mdaWriter:
            self._mdaWriter.close()

    @staticmethod
    def _sanitize_box_angles(angles):
        """ Ensure box angles correspond to first quadrant
        See `discussion on unitcell angles <https://github.com/MDAnalysis/mdanalysis/pull/2917/files#r620558575>`_
        """
        inverted = 180 - angles
        return np.min(np.array([angles, inverted]), axis=0)


def _get_platform_info():
    """Report OpenMM platform and hardware information."""
    info = {}
    # Get number of available platforms and their names
    num_platforms = mm.Platform.getNumPlatforms()
    info['available_platforms'] = [mm.Platform.getPlatform(i).getName() 
                                 for i in range(num_platforms)]
    # Try to get the fastest platform (usually CUDA or OpenCL)
    platform = None
    for platform_name in ['CUDA', 'OpenCL', 'CPU']:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            info['platform'] = platform_name
            break
        except Exception:
            continue 
    if platform is None:
        platform = mm.Platform.getPlatform(0)
        info['platform'] = platform.getName()
    # Get platform properties
    info['properties'] = {}
    try:
        if info['platform'] in ['CUDA', 'OpenCL']:
            info['properties']['device_index'] = platform.getPropertyDefaultValue('DeviceIndex')
            info['properties']['precision'] = platform.getPropertyDefaultValue('Precision')
            if info['platform'] == 'CUDA':
                info['properties']['cuda_version'] = mm.version.cuda
            info['properties']['gpu_name'] = platform.getPropertyValue(platform.createContext(), 'DeviceName')
        info['properties']['cpu_threads'] = platform.getPropertyDefaultValue('Threads')
    except Exception as e:
        logger.warning(f"Could not get some platform properties: {str(e)}")
    # Get OpenMM version
    info['openmm_version'] = mm.version.full_version
    # Log the information
    logger.info("OpenMM Platform Information:")
    logger.info(f"Available Platforms: {', '.join(info['available_platforms'])}")
    logger.info(f"Selected Platform: {info['platform']}")
    logger.info(f"OpenMM Version: {info['openmm_version']}")
    logger.info("Platform Properties:")
    for key, value in info['properties'].items():
        logger.info(f"  {key}: {value}")
    return info

