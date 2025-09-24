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
from openmm import app
from openmm.unit import angstrom, nanometer, molar, kilojoules_per_mole, kelvin, bar, nanoseconds, femtoseconds, picosecond
from reforge import cli, pdbtools, io
from reforge.pdbtools import AtomList
from reforge.utils import cd, clean_dir, logger, timeit, memprofit
from reforge.mdsystem.mdsystem import MDSystem, MDRun


class MmSystem(MDSystem):
    """Subclass for OpenMM"""

    def __init__(self, sysdir, sysname, **kwargs):
        """Initialize the MD system with required directories and file paths."""
        super().__init__(sysdir, sysname)
        
    def prepare_files(self, *args, **kwargs):
        """Extension for OpenMM system"""
        super().prepare_files(*args, **kwargs)

    def clean_pdb(self, pdb_file, **kwargs):
        """Clean the starting PDB file using PDBfixer by OpenMM.

        Parameters
        ----------
        pdb_file : str
            Path to the input PDB file.
        **kwargs : dict, optional
            Additional keyword arguments for the cleaning routine.
        """
        logger.info("Cleaning the PDB")
        pdbtools.clean_pdb(pdb_file, self.inpdb, **kwargs)

    @staticmethod
    def forcefield(force_field="amber14-all.xml", water_model="amber14/tip3p.xml", **kwargs):
        """Create and return an OpenMM ForceField object.

        Parameters
        ----------
        force_field : str, optional
            Force field file or identifier (default: 'amber14-all.xml').
        water_model : str, optional
            Water model file or identifier (default: 'amber14/tip3p.xml').
        **kwargs : dict, optional
            Additional keyword arguments for the ForceField constructor.

        Returns
        -------
        openmm.app.ForceField
            The constructed ForceField object.
        """
        forcefield = app.ForceField(force_field, water_model, **kwargs)
        return forcefield

    @staticmethod
    def modeller(inpdb, forcefield, **kwargs):
        """Generate a modeller object with added solvent.

        Parameters
        ----------
        inpdb : str
            Path to the input PDB file.
        forcefield : openmm.app.ForceField
            The force field object to be used.
        **kwargs : dict, optional
            Additional keyword arguments for solvent addition. Default values include:
                model : 'tip3p'
                boxShape : 'dodecahedron'
                padding : 1.0 * nanometer
                positiveIon : 'Na+'
                negativeIon : 'Cl-'
                ionicStrength : 0.1 * molar

        Returns
        -------
        openmm.app.Modeller
            The modeller object with solvent added.
        """
        kwargs.setdefault("model", "tip3p")
        kwargs.setdefault("boxShape", "dodecahedron")
        kwargs.setdefault("padding", 1.0 * nanometer)
        kwargs.setdefault("positiveIon", "Na+")
        kwargs.setdefault("negativeIon", "Cl-")
        kwargs.setdefault("ionicStrength", 0.1 * molar)
        pdb_file = app.PDBFile(str(inpdb))
        modeller_obj = app.Modeller(pdb_file.topology, pdb_file.positions)
        modeller_obj.addSolvent(forcefield, **kwargs)
        return modeller_obj

    def model(self, forcefield, modeller_obj, barostat=None, thermostat=None, **kwargs):
        """Create a simulation model using the specified force field and modeller.

        Parameters
        ----------
        forcefield : openmm.app.ForceField
            The force field object.
        modeller_obj : openmm.app.Modeller
            The modeller object with the prepared topology.
        barostat : openmm.Force, optional
            Barostat force to add (default: None).
        thermostat : openmm.Force, optional
            Thermostat force to add (default: None).
        **kwargs : dict, optional
            Additional keyword arguments for creating the system. Defaults include:
                nonbondedMethod : app.PME
                nonbondedCutoff : 1.0 * nanometer
                constraints : app.HBonds

        Returns
        -------
        openmm.System
            The simulation system created by the force field.
        """
        kwargs.setdefault("nonbondedMethod", app.PME)
        kwargs.setdefault("nonbondedCutoff", 1.0 * nanometer)
        kwargs.setdefault("constraints", app.HBonds)
        model_obj = forcefield.createSystem(modeller_obj.topology, **kwargs)
        if barostat:
            model_obj.addForce(barostat)
        if thermostat:
            model_obj.addForce(thermostat)
        with open(self.syspdb, "w", encoding="utf-8") as file:
            app.PDBFile.writeFile(modeller_obj.topology, modeller_obj.positions, file, keepIds=True)
        with open(self.sysxml, "w", encoding="utf-8") as file:
            file.write(mm.XmlSerializer.serialize(model_obj))
        return model_obj


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
        in_xml = os.path.join(self.rundir, "hu.xml")
        simulation.loadState(in_xml)
        enum = enumerate(simulation.system.getForces()) 
        idx, bb_restraint = [(idx, f) for idx, f in enum if f.getName() == 'BackboneRestraint'][0]
        fc = bb_restraint.getGlobalParameterDefaultValue(0)
        fcname = bb_restraint.getGlobalParameterName(0)
        for i in range(n_cycles):
            simulation.step(steps_per_cycle)
            new_fc = fc * (1 - (i + 1) / n_cycles)
            simulation.context.setParameter(fcname, new_fc)
        simulation.system.removeForce(idx)
        simulation.context.reinitialize(preserveState=True)
        self.save_state(simulation, "eq")
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
        in_xml = os.path.join(self.rundir, "eq.xml")
        simulation.loadState(in_xml)
        if not nsteps:
            dt = simulation.integrator.getStepSize()
            logger.info(f"Total MD time: %s", time)
            nsteps = int(time / dt)
        logger.info(f"Number of steps left: %s", nsteps)
        simulation.step(nsteps)
        self.save_state(simulation, "md")
        logger.info("Production completed.")

    @memprofit(level=logging.INFO)
    @timeit(level=logging.INFO, unit='auto')
    def extend(self, simulation, until_time=1000, nsteps=None, **kwargs):
        """Extend production MD simulation"""
        logger.info("Extending run...")
        xml_file = os.path.join(self.rundir, "md.xml")
        simulation.loadState(xml_file)
        st = simulation.context.getState()
        if not nsteps:
            dt = simulation.integrator.getStepSize()
            curr_time = st.getTime()
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
        positions = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        self._mdaUniverse.atoms.positions = positions
        save_velocities = self.describeNextReport(simulation)[2]
        if save_velocities:
            velocities = state.getVelocities(asNumpy=True).value_in_unit(angstrom/picosecond)
            self._mdaUniverse.atoms.velocities = velocities
        # update box vectors
        boxVectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(angstrom)
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


################################################################################
# Helper functions
################################################################################

def sort_uld(alist):
    """Sort characters in a list in a specific order.

    Parameters
    ----------
    alist : list of str
        List of characters to sort.

    Returns
    -------
    list of str
        Sorted list with uppercase letters first, then lowercase letters, and digits last.

    Notes
    -----
    This function is used to help organize GROMACS multichain files.
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist
