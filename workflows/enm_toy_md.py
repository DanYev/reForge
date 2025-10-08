import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import clean_dir, get_logger

logger = get_logger(__name__)

# Global settings
INPDB = '1btl.pdb'
# ENM parameters
ENM_CUTOFF = 1.1 * unit.nanometer  # cutoff distance for ENM springs
ENM_FORCE_CONSTANT = 500.0 * unit.kilojoules_per_mole / (unit.nanometer**2)  # spring constant
# Production parameters
TEMPERATURE = 300 * unit.kelvin
GAMMA = 1 / unit.picosecond
# Either steps or time
TOTAL_TIME = 100 * unit.picoseconds
TSTEP = 20 * unit.femtoseconds
TOTAL_STEPS = 50000
# Reporting: save every NOUT steps
TRJ_NOUT = 100          # Trajectory   
LOG_NOUT = 1000         # Log file   
CHK_NOUT = 10000        # Checkpoint
OUT_SELECTION = "name CA"
TRJEXT = 'trr' # trr saves positions, velocities, forces


def setup(*args):
    """Main setup function for ENM system"""
    print(f"Setup called with args: {args}", file=sys.stderr)
    setup_enm(*args)


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


def add_enm_forces(system, positions, cutoff=ENM_CUTOFF, force_constant=ENM_FORCE_CONSTANT):
    """Add Elastic Network Model harmonic forces based on cutoff distance"""
    logger.info(f"Adding ENM forces with cutoff {cutoff} and force constant {force_constant}")
    n_atoms = len(positions)
    cutoff_nm = cutoff.value_in_unit(unit.nanometer)
    # Create custom bond force for ENM springs
    enm_force = mm.CustomBondForce('k * (r - r0)^2')
    enm_force.addGlobalParameter('k', force_constant)
    enm_force.addPerBondParameter('r0')
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
                enm_force.addBond(i, j, [distance_with_units])
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
    system = add_enm_forces(system, positions, ENM_CUTOFF, ENM_FORCE_CONSTANT)
    # Save system to XML file
    with open(mdsys.sysxml, "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved ENM system to {mdsys.sysxml}")
    # Create positions with units for OpenMM PDB writing
    positions_with_units = positions * unit.nanometer
    # Save CA-only PDB file using OpenMM (matches topology we created)
    with open(mdsys.syspdb, 'w') as file:
        app.PDBFile.writeFile(topology, positions_with_units, file)
    logger.info(f"Saved CA-only structure to {mdsys.syspdb}")


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
    """Run NVE simulation with ENM forces"""
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: {mdrun.rundir}")
    # Load system and topology
    with open(mdsys.sysxml, 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    pdb = app.PDBFile(str(mdsys.syspdb)) # Load CA-only structure
    integrator = mm.VerletIntegrator(TSTEP)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    # Minimize energy
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    # Set initial velocities
    logger.info("Setting initial velocities...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    # Setup reporters
    reporters = _get_reporters(mdrun, append=False, prefix="md")
    simulation.reporters.extend(reporters)
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb")
    # Run simulation
    logger.info(f"Running NVE simulation for {TOTAL_STEPS} steps...")
    simulation.step(TOTAL_STEPS)
    logger.info("NVE simulation completed!")


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
    tmp_traj = mdrun.rundir / f"conv.{TRJEXT}"
    out_traj = mdrun.rundir / f"samples.{TRJEXT}"
    logger.info(f'Converting trajectory with selection: {SELECTION}')
    _trjconv_selection(trajs, top, tmp_traj, out_top, selection=SELECTION, step=1)
    # FIT + OUTPUT
    _trjconv_fit(tmp_traj, out_top, out_traj, transform_vels=TRJEXT=='trr')
    os.remove(tmp_traj)


def _trjconv_selection(input_traj, input_top, output_traj, output_top, selection="name CA", step=1):
    u = mda.Universe(input_top, input_traj)
    selected_atoms = u.select_atoms(selection)
    n_atoms = selected_atoms.n_atoms
    selected_atoms.write(output_top)
    with mda.Writer(str(output_traj), n_atoms=n_atoms) as writer:
        for ts in u.trajectory[::step]:
            writer.write(selected_atoms)
    logger.info("Saved selection '%s' to %s and topology to %s", selection, output_traj, output_top)


def _trjconv_fit(input_traj, input_top, output_traj, transform_vels=False):
    u = mda.Universe(input_top, input_traj)
    ag = u.atoms
    ref_u = mda.Universe(input_top) 
    ref_ag = ref_u.atoms
    u.trajectory.add_transformations(fit_rot_trans(ag, ref_ag,))
    logger.info("Converting/Writing Trajecory")
    with mda.Writer(str(output_traj), ag.n_atoms) as W:
        for ts in u.trajectory:   
            if transform_vels:
                transformed_vels = _tranform_velocities(ts.velocities, ts.positions, ref_ag.positions)
                ag.velocities = transformed_vels
            W.write(ag)
            if ts.frame % 1000 == 0:
                frame = ts.frame
                time_ns = ts.time / 1000
                logger.info(f"Current frame: %s at %s ns", frame, time_ns)
    logger.info("Done!")


def _tranform_velocities(vels, poss, ref_poss):
    R = _kabsch_rotation(poss, ref_poss)
    vels_aligned = vels @ R
    return vels_aligned
    

def _kabsch_rotation(P, Q):
    """
    Return the 3x3 rotation matrix R that best aligns P onto Q (both Nx3),
    after removing centroids (i.e., pure rotation via Kabsch).
    """
    # subtract centroids
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    # covariance and SVD
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # right-handed correction
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


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


def calculate_hessian_gromacs(sysdir, sysname, runname=None):
    """Calculate Hessian matrix using GROMACS assuming topology files are already prepared
    
    Assumes the following files exist in the system directory:
    - conf.gro: Initial structure (from pdb2gmx)
    - topol.top: Topology file (from pdb2gmx) 
    - nm.mdp: MDP file for normal mode analysis
    """
    mdsys = GmxSystem(sysdir, sysname)
    work_dir = mdsys.root
    logger.info(f"Calculating Hessian in directory: {work_dir}")
    pdb_file = mdsys.sysdir / INPDB
    # mdsys.prepare_files()
    # mdsys.clean_pdb_gmx(pdb_file, clinput="8\n 7\n", ignh="no", renum="yes")
    # mdsys.gmx('pdb2gmx', clinput="8\n 1\n", f="inpdb.pdb", o="conf.gro", p="topol.top")
    input_gro = work_dir / "conf.gro"
    top_file = work_dir / "topol.top" 
    # Expected .mdp files (should already exist)
    nm_mdp_file = Path("nm.mdp").resolve()  # Assuming nm.mdp is in the current directory
    minim_mdp_file = Path("minim.mdp").resolve()  # Minimization MDP file
    
    # Output files
    minim_tpr_file = work_dir / "minim.tpr"
    minimized_gro = work_dir / "minimized.gro"
    tpr_file = work_dir / "nm.tpr"
    hessian_file = work_dir / "hessian.mtx"
    eigenval_file = work_dir / "eigenval.xvg"
    eigenvec_file = work_dir / "eigenvec.trr"
    
    # Check if required files exist
    required_files = [input_gro, top_file, nm_mdp_file, minim_mdp_file]
    for req_file in required_files:
        if not req_file.exists():
            logger.error(f"Required file not found: {req_file}")
            logger.error("Please run pdb2gmx first to generate topology files and ensure nm.mdp and minim.mdp are present.")
            return None
    
    # Step 1: Create large box around protein (simulates infinite dilution)
    boxed_gro = work_dir / "boxed.gro"
    logger.info("Creating large box around protein...")
    mdsys.gmx('editconf', f=str(input_gro), o=str(boxed_gro), c='', d='3.0', bt='dodecahedron')
    logger.info(f"Created boxed structure: {boxed_gro}")
    
    # Step 2: Energy minimization of protein-only system
    logger.info("Running energy minimization...")
    mdsys.gmx('grompp', f=str(minim_mdp_file), c=str(boxed_gro), p=str(top_file), o=str(minim_tpr_file))
    logger.info(f"Created minimization TPR file: {minim_tpr_file}")
    mdsys.gmx('mdrun', s=str(minim_tpr_file), c=str(minimized_gro))
    logger.info(f"Energy minimization completed. Minimized structure: {minimized_gro}")
    # Step 2: Create TPR file for normal mode analysis using minimized structure
    logger.info("Running grompp for normal mode analysis...")
    mdsys.gmx('grompp', f=str(nm_mdp_file), c=str(minimized_gro), p=str(top_file), o=str(tpr_file))
    logger.info(f"Created TPR file for normal modes: {tpr_file}")
    # Step 3: Calculate Hessian using mdrun with nm integrator
    logger.info("Calculating Hessian matrix...")
    mdsys.gmx('mdrun', deffnm='eigenvec', s=str(tpr_file), mtx=str(hessian_file), v='')
    logger.info(f"Hessian matrix saved to: {hessian_file}")
    # Step 4: Calculate normal modes (eigenvalues and eigenvectors)
    logger.info("Calculating normal modes...")
    mdsys.gmx('nmeig', f=str(hessian_file), s=str(tpr_file), ol=str(eigenval_file), v=str(eigenvec_file), xvg='none')
    logger.info(f"Eigenvalues saved to: {eigenval_file}")
    logger.info(f"Eigenvectors saved to: {eigenvec_file}")
    # Step 4: Convert eigenvalues to numpy array
    logger.info("Converting eigenvalues to numpy array...")
    eigenvalues = []
    with open(eigenval_file, 'r') as f:
        for line in f:
            if not line.startswith('#') and not line.startswith('@'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    eigenvalues.append(float(parts[1]))
    
    eigenvalues = np.array(eigenvalues)
    # Save eigenvalues as numpy array
    eigenval_npy = work_dir / "eigenvalues.npy"
    np.save(eigenval_npy, eigenvalues)
    logger.info(f"Eigenvalues saved as numpy array: {eigenval_npy}")
    # Summary
    n_modes = len(eigenvalues)
    logger.info(f"Calculated {n_modes} normal modes")
    if n_modes > 0:
        logger.info(f"Frequency range: {np.min(eigenvalues):.3f} to {np.max(eigenvalues):.3f} cm^-1")


def main(sysdir, sysname, runname):
    # sysdir = "/scratch/dyangali/reforge/workflows/systems/"
    # sysname = "enm_system"
    # runname = "nve_run"

    # setup(sysdir, sysname)
    # md_nve(sysdir, sysname, runname)
    # trjconv(sysdir, sysname, runname)
    # extract_trajectory_data(sysdir, sysname, runname, prefix="md")
    calculate_hessian_gromacs(sysdir, sysname, runname)


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()