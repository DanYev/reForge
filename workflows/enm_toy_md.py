import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from reforge import mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import clean_dir, get_logger
import plots

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


def load_force_constants_matrix(sysdir, sysname):
    """Load the saved pairwise force constants matrix"""
    mdsys = MmSystem(sysdir, sysname)
    force_constants_file = mdsys.sysdir / "enm_force_constants.npy"
    if force_constants_file.exists():
        force_constants_matrix = np.load(force_constants_file)
        logger.info(f"Loaded force constants matrix from {force_constants_file}")
        logger.info(f"Matrix shape: {force_constants_matrix.shape}")
        logger.info(f"Force constant range: {np.min(force_constants_matrix[force_constants_matrix > 0]):.2f} - {np.max(force_constants_matrix):.2f} kJ/mol/nm²")
        return force_constants_matrix
    else:
        logger.warning(f"Force constants file not found: {force_constants_file}")
        return None


def plot_force_constants_matrix(sysdir, sysname, output_dir=None):
    """Plot the pairwise force constants matrix as a heatmap"""
    import matplotlib.pyplot as plt
    
    force_constants_matrix = load_force_constants_matrix(sysdir, sysname)
    if force_constants_matrix is None:
        return
    
    if output_dir is None:
        mdsys = MmSystem(sysdir, sysname)
        output_dir = mdsys.sysdir
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(force_constants_matrix, cmap='viridis', aspect='auto')
    
    ax.set_xlabel('CA Atom Index')
    ax.set_ylabel('CA Atom Index')
    ax.set_title('Pairwise ENM Force Constants Matrix')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Force Constant (kJ/mol/nm²)')
    
    # Save plot
    output_file = Path(output_dir) / "enm_force_constants_matrix.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved force constants matrix plot to {output_file}")
    return output_file


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
    """Calculate force constant for a specific bond between atoms i and j.
    For now this is for mock testing.
    """
    # For testing use the same force constant for all bonds
    force_constant = np.random.normal(1.0, 0.2)**2 * default_force_constant
    # Distance-dependent force constants
    if distance > 0.7:  # Long-range contacts
        return force_constant * 0.5
    # Sequential neighbors (would need sequence info)
    if abs(atom_i - atom_j) == 1:  # Adjacent in sequence
        return force_constant * 2.0  # Stronger covalent-like
    return force_constant


def add_enm_forces(system, positions, cutoff=ENM_CUTOFF, force_constant=ENM_FORCE_CONSTANT, ca_atoms=None):
    """Add Elastic Network Model harmonic forces based on cutoff distance
    
    Args:
        system: OpenMM system to add forces to
        positions: CA atom positions (N x 3 array)
        cutoff: Distance cutoff for ENM springs
        force_constant: Default force constant for springs
        ca_atoms: MDAnalysis AtomGroup with CA atoms (for future customization)
    
    Returns:
        system: OpenMM system with ENM forces added
        force_constants_matrix: N x N numpy array with pairwise force constants
    """
    logger.info(f"Adding ENM forces with cutoff {cutoff} and force constant {force_constant}")
    n_atoms = len(positions)
    cutoff_nm = cutoff.value_in_unit(unit.nanometer)
    
    # Initialize pairwise force constants matrix
    force_constants_matrix = np.zeros((n_atoms, n_atoms))
    
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
                
                # Store force constant in matrix (symmetric)
                force_constant_value = bond_force_constant.value_in_unit(unit.kilojoules_per_mole / (unit.nanometer**2))
                force_constants_matrix[i, j] = force_constant_value
                force_constants_matrix[j, i] = force_constant_value
                
                enm_force.addBond(i, j, [bond_force_constant, distance_with_units])
                bonds_added += 1
    
    logger.info(f"Added {bonds_added} ENM springs")
    system.addForce(enm_force)
    return system, force_constants_matrix


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
    system, force_constants_matrix = add_enm_forces(system, positions, ENM_CUTOFF, ENM_FORCE_CONSTANT, ca_atoms)
    
    # Save pairwise force constants matrix
    force_constants_file = mdsys.datdir / "enm_force_constants.npy"
    np.save(force_constants_file, force_constants_matrix)
    logger.info(f"Saved pairwise force constants matrix to {force_constants_file}")
    logger.info(f"Force constants matrix shape: {force_constants_matrix.shape}")
    logger.info(f"Number of non-zero connections: {np.count_nonzero(force_constants_matrix) // 2}")  # Divide by 2 since matrix is symmetric
    
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

################################################################################
### ENM analysis ###
################################################################################

def enm_analysis(sysdir, sysname):
    """Calculate ENM-based metrics."""
    system = MDSystem(sysdir, sysname)
    in_pdb = system.syspdb
    u = mda.Universe(in_pdb)
    ag = u.select_atoms("name CA")
    # vecs = np.array(ag.positions).astype(np.float64) # (n_atoms, 3)
    # hess = mdm.hessian(vecs, spring_constant=5, cutoff=11, dd=0) # distances in Angstroms, dd=0 no distance-dependence
    hess = np.load(system.datdir / "md_hess.npy")
    covmat = mdm.inverse_matrix(hess, device="gpu_dense", k_singular=6, n_modes=1000, dtype=np.float64)
    covmat = covmat * 1.25 # kb*T at 300K in kJ/mol
    outfile = system.datdir / "enm_cov.npy"
    np.save(outfile, covmat)
    pertmat = mdm.perturbation_matrix_iso(covmat)
    rmsf = np.sqrt(np.diag(covmat).reshape(-1, 3).sum(axis=1)) # (n_atoms,)
    dfi = mdm.dfi(pertmat)
    plots.simple_residue_plot(system, [rmsf], outtag="enm_rmsf")
    plots.simple_residue_plot(system, [dfi], outtag="enm_dfi")


def get_hessian_from_md(sysdir, sysname):
    system = MDSystem(sysdir, sysname)
    covmat = np.load(system.datdir / "covmat_av.npy")
    hess = mdm.inverse_matrix(covmat, device="gpu_dense", k_singular=6, n_modes=1000, dtype=np.float64)
    hess = hess * 1.25 # kb*T at 300K in kJ/mol
    outfile = system.datdir / "md_hess.npy"
    np.save(outfile, hess)


def main(sysdir, sysname, runname):
    # sysdir = "/scratch/dyangali/reforge/workflows/systems/"
    # sysname = "enm_system"
    # runname = "nve_run"
    setup(sysdir, sysname)
    md_nve(sysdir, sysname, runname)
    trjconv(sysdir, sysname, runname)
    extract_trajectory_data(sysdir, sysname, runname, prefix="md")


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()