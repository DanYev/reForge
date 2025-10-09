import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import clean_dir, get_logger

logger = get_logger(__name__)

# Global settings
INPDB = '1btl.pdb'
# ENM parameters
ENM_CUTOFF = 1.1 * unit.nanometer  # cutoff distance for ENM springs
ENM_FORCE_CONSTANT = 500.0 * unit.kilojoules_per_mole / (unit.nanometer**2) 
# Production parameters
TEMPERATURE = 300 * unit.kelvin
GAMMA = 1 / unit.picosecond
# Either steps or time
TOTAL_TIME = 100 * unit.picoseconds
TSTEP = 20 * unit.femtoseconds
TOTAL_STEPS = 500000
# Reporting: save every NOUT steps
TRJ_NOUT = 100           # Trajectory   
LOG_NOUT = 10000         # Log file   
CHK_NOUT = 100000        # Checkpoint
OUT_SELECTION = "name CA"
TRJEXT = 'trr' # trr saves positions, velocities, forces

##########################################################
### Utils ###
##########################################################

def _save_system_to_xml(system, filename):
    with open(str(filename), "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved system to {filename}")


def _load_system_from_xml(filename):
    with open(str(filename), 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    logger.info(f"Loaded system from {filename}")
    return system


def setup(*args):
    """Main setup function for ENM system"""
    print(f"Setup called with args: {args}", file=sys.stderr)
    setup_enm(*args)

##########################################################
### ENM stuff ###
##########################################################

def extract_ca_positions(pdb_file):
    """Extract CA positions from PDB file and create a mock CA-only structure"""
    logger.info(f"Reading PDB file: {pdb_file}")
    u = mda.Universe(str(pdb_file))
    ca_atoms = u.select_atoms("name CA")
    positions = 0.1 * ca_atoms.positions # Convert Angstrom to nanometers
    return ca_atoms, positions


def create_ca_topology(ca_atoms):
    """Create OpenMM topology with only CA atoms"""
    topology = app.Topology()
    chain = topology.addChain()
    for i, atom in enumerate(ca_atoms):
        residue = topology.addResidue(f"GLY", chain)  # Use GLY as placeholder
        topology.addAtom("CA", app.Element.getBySymbol("C"), residue)
    return topology


def _get_bond_force_constant(atom_i, atom_j, distance, default_force_constant):
    """Calculate force constant for a specific bond between atoms i and j
    
    This function can be customized to return different force constants based on:
    - Atom types (e.g., different for backbone vs sidechain)
    - Residue types (e.g., stronger for certain amino acids)
    - Distance ranges (e.g., weaker for long-range contacts)
    - Secondary structure (e.g., stronger within helices/sheets)
    
    For now, returns the same force constant for all bonds (testing phase)
    """
    # That's just for mock testing
    force_constant = np.random.normal(1.0, 0.2)**2 * default_force_constant
    # Distance-dependent force constants
    if distance > 0.8:  # Long-range contacts
        return force_constant * 0.5
    # Sequential neighbors (would need sequence info)
    if abs(atom_i - atom_j) == 1:  # Adjacent in sequence
        return force_constant * 2.0  # Stronger covalent-like
    # # For testing: use the same force constant for all bonds
    return force_constant


def add_enm_forces(system, positions, cutoff=ENM_CUTOFF, force_constant=ENM_FORCE_CONSTANT, ca_atoms=None):
    """Add Elastic Network Model harmonic forces based on cutoff distance
    
    Args:
        system: OpenMM system to add forces to
        positions: CA atom positions (N x 3 array)
        cutoff: Distance cutoff for ENM springs
        force_constant: Default force constant for springs
        ca_atoms: MDAnalysis AtomGroup with CA atoms (for future customization)
    """
    logger.info(f"Adding ENM forces with cutoff {cutoff} and force constant {force_constant}")
    n_atoms = len(positions)
    cutoff_nm = cutoff.value_in_unit(unit.nanometer)
    # Create custom bond force for ENM springs with per-bond force constants
    enm_force = mm.CustomBondForce('k * (r - r0)^2')
    enm_force.addPerBondParameter('k')   # Force constant per bond
    enm_force.addPerBondParameter('r0')  # Equilibrium distance per bond
    bonds_added = 0
    
    # Add bonds between atoms within cutoff distance
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pos_i = positions[i]
            pos_j = positions[j]
            distance = np.sqrt(
                (pos_i[0] - pos_j[0])**2 + 
                (pos_i[1] - pos_j[1])**2 + 
                (pos_i[2] - pos_j[2])**2
            )
            # Add spring if within cutoff
            if distance <= cutoff_nm:
                distance_with_units = distance * unit.nanometer
                bond_force_constant = _get_bond_force_constant(i, j, distance, force_constant)
                enm_force.addBond(i, j, [bond_force_constant, distance_with_units])
                bonds_added += 1
    logger.info(f"Added {bonds_added} ENM springs")
    system.addForce(enm_force)
    return system


def setup_enm(sysdir, sysname):
    """Setup ENM system with CA atoms only"""
    mdsys = MmSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
    mdsys.prepare_files()
    # Extract CA positions and create topology
    ca_atoms, positions = extract_ca_positions(inpdb)
    topology = create_ca_topology(ca_atoms)
    # Create OpenMM system
    system = mm.System()
    # Add particles (CA atoms) - use carbon mass
    carbon_mass = 60.01 * unit.amu # Fake carbons
    for i in range(len(positions)):
        system.addParticle(carbon_mass)
    system = add_enm_forces(system, positions, ENM_CUTOFF, ENM_FORCE_CONSTANT, ca_atoms)
    # Save system to XML file
    _save_system_to_xml(system, mdsys.sysxml)
    logger.info(f"Saved ENM system to {mdsys.sysxml}")
    # Create positions with units for OpenMM PDB writing
    positions_with_units = positions * unit.nanometer
    # Save CA-only PDB file using OpenMM (matches topology we created)
    with open(mdsys.syspdb, 'w') as file:
        app.PDBFile.writeFile(topology, positions_with_units, file)
    logger.info(f"Saved CA-only structure to {mdsys.syspdb}")

###########################################################
### MD ###
###########################################################

def _get_reporters(mdrun, append=False, prefix="md"):
    """Get reporters for MD simulation using custom MmReporter for velocities"""
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    # Log reporter (file)
    log_reporter = app.StateDataReporter(
        str(mdrun.rundir / f"{prefix}.log"), 
        LOG_NOUT, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Error reporter (stderr)
    err_reporter = app.StateDataReporter(
        sys.stderr, LOG_NOUT, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Custom trajectory reporter with velocities using MmReporter
    logger.info(f'Setting up trajectory reporter with selection: {OUT_SELECTION}')
    traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.{TRJEXT}"), 
        reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    # State/checkpoint reporter
    state_reporter = app.CheckpointReporter(str(mdrun.rundir / f"{prefix}.xml"), CHK_NOUT, writeState=True)
    return log_reporter, err_reporter, traj_reporter, state_reporter


def md_nve(sysdir, sysname, runname):
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    system = _load_system_from_xml(mdsys.sysxml)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, 0.5*TSTEP) # NVT integrator for equilibration
    simulation = app.Simulation(pdb.topology, system, integrator) 
    # --- Initialize state, minimize, equilibrate ---
    logger.info("Minimizing energy...")
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)  
    logger.info("Equilibrating...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    simulation.step(10000)  # equilibrate 
    # --- Run NVE (need to change the integrator and reset simulation) ---
    logger.info("Running NVE production...")
    integrator = mm.VerletIntegrator(TSTEP)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setState(state)
    logger.info(f'Saving reference PDB with selection: {OUT_SELECTION}')
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb") # SAVE PDB FOR THE SELECTION
    reporters = _get_reporters(mdrun, append=False, prefix="md")
    simulation.reporters.extend(reporters)
    simulation.step(TOTAL_STEPS)  
    logger.info("Done!")



def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # INPUT
    top = mdrun.rundir / "md.pdb"
    # top = mdrun.syspdb  # use original topology if needed
    traj = mdrun.rundir / f"md.{TRJEXT}"
    ext_trajs = sorted([f for f in mdrun.rundir.glob(f"md_*.{TRJEXT}")])
    trajs = [traj] + ext_trajs
    logger.info(f'Input trajectory files: {trajs}')
    # CONVERT
    out_top = mdrun.rundir / "topology.pdb"
    out_traj = mdrun.rundir / f"samples.{TRJEXT}"
    logger.info(f'Converting trajectory with selection: {OUT_SELECTION}')
    convert_trajectories(top, trajs, out_top, out_traj, selection=OUT_SELECTION, step=1)
    logger.info("Done!")


def extract_trajectory_data(sysdir, sysname, runname, prefix="md"):
    """Extract positions and velocities from trajectory and save as numpy arrays"""
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    # Check if trajectory file exists
    traj_file = mdrun.rundir / f"{prefix}.{TRJEXT}"
    logger.info(f"Extracting trajectory data from {traj_file}")
    u = mda.Universe(str(mdsys.syspdb), str(traj_file))
    logger.info(f"Loaded trajectory with {len(u.trajectory)} frames")
    n_frames = len(u.trajectory)
    n_atoms = len(u.atoms)
    # Initialize arrays for positions and velocities
    positions_array = np.zeros((n_frames, n_atoms, 3))
    velocities_array = np.zeros((n_frames, n_atoms, 3))
    # Extract positions and velocities for each frame
    logger.info("Extracting positions and velocities...")
    for i, ts in enumerate(u.trajectory):
        # Positions are in Angstroms, convert to nanometers
        positions_array[i] = ts.positions / 10.0
        # Velocities (if available in the trajectory)
        if hasattr(ts, 'velocities') and ts.velocities is not None:
            # Velocities are in Angstrom/ps, convert to nm/ps
            velocities_array[i] = ts.velocities / 10.0
        else:
            logger.warning(f"No velocities found in frame {i}")
    # Save arrays
    pos_file = mdrun.rundir / f"{prefix}_positions.npy"
    vel_file = mdrun.rundir / f"{prefix}_velocities.npy"
    np.save(pos_file, positions_array)
    np.save(vel_file, velocities_array)
    logger.info(f"Saved positions to {pos_file}")
    logger.info(f"Saved velocities to {vel_file}")
    logger.info(f"Array shapes: positions {positions_array.shape}, velocities {velocities_array.shape}")
    # Also save metadata
    metadata = {
        'n_frames': n_frames,
        'n_atoms': n_atoms,
        'timestep_ps': TSTEP.value_in_unit(unit.picosecond),
        'total_time_ps': n_frames * TSTEP.value_in_unit(unit.picosecond),
        'trajectory_file': str(traj_file),
        'units': {
            'positions': 'nanometers',
            'velocities': 'nm/ps',
            'time': 'picoseconds'
        }
    }
    metadata_file = mdrun.rundir / f"{prefix}_metadata.npy"
    np.save(metadata_file, metadata)
    logger.info(f"Saved metadata to {metadata_file}")
    return positions_array, velocities_array, metadata


def main(sysdir, sysname, runname):
    # sysdir = "/scratch/dyangali/reforge/workflows/systems/"
    # sysname = "enm_system"
    # runname = "nve_run"
    setup(sysdir, sysname)
    md_nve(sysdir, sysname, runname)
    trjconv(sysdir, sysname, runname)
    # extract_trajectory_data(sysdir, sysname, runname, prefix="md")


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()