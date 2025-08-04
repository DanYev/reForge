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

import os
import sys
import shutil
import openmm as mm
from openmm import app
from openmm.unit import nanometer, molar
from reforge import cli, pdbtools, io
from reforge.pdbtools import AtomList
from reforge.utils import cd, clean_dir, logger
from reforge.mdsystem.mdsystem import MDSystem, MDRun


class MmSystem(MDSystem):
    """Subclass for OpenMM"""

    def __init__(self, sysdir, sysname, **kwargs):
        """Initialize the MD system with required directories and file paths."""
        super().__init__(sysdir, sysname)
        self.sysxml = self.root / "system.xml"
        self.sysgro = self.root / "system.gro"
        self.systop = self.root / "system.top"
        
    def prepare_files(self):
        """Extension for OpenMM system"""
        super().prepare_files()

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
        pdb_file = os.path.join(self.rundir, file_prefix + ".pdb")
        xml_file = os.path.join(self.rundir, file_prefix + ".xml")
        simulation_obj.saveState(xml_file)
        state = simulation_obj.context.getState(getPositions=True)
        positions = state.getPositions()
        with open(pdb_file, "w", encoding="utf-8") as file:
            app.PDBFile.writeFile(simulation_obj.topology, positions, file, keepIds=True)

    def em(self, simulation_obj, tolerance=100, max_iterations=1000):
        """Perform energy minimization for the simulation.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        tolerance : float, optional
            Tolerance for energy minimization (default: 100).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).

        Notes
        -----
        Minimizes the energy, saves the minimized state, and logs progress.
        """
        logger.info("Minimizing energy...")
        log_file = os.path.join(self.rundir, "em.log")
        reporter = app.StateDataReporter(
            log_file, 100, step=True, potentialEnergy=True, temperature=True
        )
        simulation_obj.reporters.append(reporter)
        simulation_obj.minimizeEnergy(tolerance, max_iterations)
        self.save_state(simulation_obj, "em")
        logger.info("Minimization complete.")

    def get_std_reporters(self, append, prefix='md', nout=1000, nlog=10000, nchk=10000, **kwargs):
        kwargs.setdefault("step", True)
        kwargs.setdefault("time", True)
        kwargs.setdefault("potentialEnergy", True)
        kwargs.setdefault("temperature", True)
        kwargs.setdefault("density", False)
        dcd_file = os.path.join(self.rundir, f"{prefix}.dcd")
        xtc_file = os.path.join(self.rundir, f"{prefix}.xtc")
        log_file = os.path.join(self.rundir, f"{prefix}.log")
        xml_file = os.path.join(self.rundir, f"{prefix}.xml")
        pdb_file = os.path.join(self.rundir, f"{prefix}.pdb")
        dcd_reporter = app.DCDReporter(dcd_file, nout, append=append, enforcePeriodicBox=True)
        xtc_reporter = app.XTCReporter(xtc_file, nout, append=append, enforcePeriodicBox=True)
        log_reporter = app.StateDataReporter(log_file, nlog, append=append, **kwargs)
        xml_reporter = app.CheckpointReporter(xml_file, nchk, writeState=True)
        # If PDBReporter is not used, remove the assignment to avoid unused variable warnings.
        reporters = [dcd_reporter, xtc_reporter, log_reporter, xml_reporter]
        return reporters

    def eq(self, simulation_obj, nsteps=10000, nout=10000, nlog=10000, nchk=10000, **kwargs):
        """Run equilibration simulation.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of steps for equilibration (default: 10000).
        nlog : int, optional
            Logging frequency (default: 10000).
        **kwargs : dict, optional
            Additional keyword arguments for the StateDataReporter.

        Notes
        -----
        Loads the minimized state, runs equilibration, and saves the equilibrated state.
        """
        logger.info("Starting equilibration...")
        
        em_xml = os.path.join(self.rundir, "em.xml")
        simulation_obj.loadState(em_xml)
        reporters = self.get_std_reporters(append=False, nout=nout, nlog=nlog, nchk=nchk, density=True, **kwargs)
        simulation_obj.reporters = []
        simulation_obj.reporters.extend(reporters)
        simulation_obj.step(nsteps)
        self.save_state(simulation_obj, "eq")
        simulation_obj.reporters.clear()
        logger.info("Equilibration complete.")

    def md(self, simulation_obj, nsteps=100000, nout=1000, nlog=10000, nchk=10000, **kwargs):
        """Run production MD simulation.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of production steps (default: 100000).
        nout : int, optional
            Frequency for writing trajectory frames (default: 1000).
        nlog : int, optional
            Logging frequency (default: 10000).
        nchk : int, optional
            Checkpoint frequency (default: 10000).
        **kwargs : dict, optional
            Additional keyword arguments for the StateDataReporter.

        Notes
        -----
        Loads the equilibrated state, runs production, and saves the final simulation state.
        """
        logger.info("Production run...")
        
        eq_xml = os.path.join(self.rundir, "eq.xml")
        pdb_file = os.path.join(self.rundir, "md.pdb")
        simulation_obj.loadState(eq_xml)
        reporters = self.get_std_reporters(append=False, nout=nout, nlog=nlog, nchk=nchk, **kwargs)
        simulation_obj.reporters = []
        simulation_obj.reporters.extend(reporters)
        simulation_obj.step(nsteps)
        self.save_state(simulation_obj, "md")
        logger.info("Production complete.")

    def extend(self, simulation_obj, nsteps=100000, nout=1000, nlog=10000, nchk=10000, **kwargs):
        """Extend production MD simulation"""
        logger.info("Extending run...")
        xml_file = os.path.join(self.rundir, "md.xml")
        simulation_obj.loadState(xml_file)
        reporters = self.get_std_reporters(append=True, nout=nout, nlog=nlog, nchk=nchk, **kwargs)
        simulation_obj.reporters = []
        simulation_obj.reporters.extend(reporters)
        simulation_obj.step(nsteps)
        self.save_state(simulation_obj, "md")
        logger.info("Production complete.")


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
