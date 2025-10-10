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

##########################################################
### Setup ###
##########################################################

def setup(*args):
    """Main setup function for ENM system"""
    print(f"Setup called with args: {args}", file=sys.stderr)
    # setup_enm(*args)
    setup_aa(*args)


def setup_aa(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
    mdsys.prepare_files()
    mdsys.clean_pdb(inpdb, add_missing_atoms=True, add_hydrogens=True)
    pdb = app.PDBFile(str(mdsys.inpdb))
    forcefield = app.ForceField("amber19-all.xml")
    with open(mdsys.syspdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, file, keepIds=True)    
    logger.info("Saved vacuum system to %s", mdsys.syspdb)
    logger.info("Generating topology...")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,  # No cutoff for vacuum
        constraints=None,              # No constraints for Hessian
        removeCMMotion=False,          # No COM motion removal
        hydrogenMass=None              # Keep original hydrogen masses
    )
    _save_system_to_xml(system, mdsys.sysxml)
    logger.info("Setup complete.")


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
    simulation.step(100000)  # equilibrate 
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

def get_hessian_from_md(sysdir, sysname):
    system = MDSystem(sysdir, sysname)
    covmat = np.load(system.datdir / "covmat_av.npy")
    hess = mdm.inverse_matrix(covmat, device="gpu_dense", k_singular=6, n_modes=1000, dtype=np.float64)
    hess = hess * 1.25 # kb*T at 300K in kJ/mol
    outfile = system.datdir / "md_hess.npy"
    np.save(outfile, hess)


def get_hessian_from_enm(sysdir, sysname):
    """Calculate Hessian matrix directly from ENM force constants and PDB structure using vectorized operations"""
    system = MDSystem(sysdir, sysname)
    # Load force constants matrix
    force_constants_matrix = np.load(system.datdir / "enm_force_constants.npy")
    if force_constants_matrix is None:
        raise ValueError("Force constants matrix not found. Run setup first.")
    # Load CA positions from PDB
    mdsys = MmSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
    ca_atoms, positions = extract_ca_positions(inpdb)
    n_atoms = len(positions)
    logger.info(f"Building Hessian matrix for {n_atoms} CA atoms using vectorized operations")
    # Create pairwise distance vectors: r_ij = r_j - r_i
    r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    r_ij_norms = np.linalg.norm(r_ij, axis=2)
    r_ij_norms_safe = np.where(r_ij_norms == 0, 1.0, r_ij_norms)
    u_ij = r_ij / r_ij_norms_safe[:, :, np.newaxis]
    # Zero out diagonal elements (no self-interaction)
    diagonal_mask = np.eye(n_atoms, dtype=bool)
    u_ij[diagonal_mask] = 0
    # Create outer products for all pairs: u_ij ⊗ u_ij
    u_outer = u_ij[:, :, :, np.newaxis] * u_ij[:, :, np.newaxis, :]
    # Multiply by force constants: -k_ij * (u_ij ⊗ u_ij)
    h_blocks = -force_constants_matrix[:, :, np.newaxis, np.newaxis] * u_outer
    # Initialize full Hessian matrix (3N x 3N)
    hess = np.zeros((3 * n_atoms, 3 * n_atoms))
    # Fill off-diagonal blocks using advanced indexing
    i_indices = np.arange(n_atoms)
    j_indices = np.arange(n_atoms)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    # Create index arrays for block assignment
    i_starts = 3 * i_grid
    j_starts = 3 * j_grid
    # Vectorized block assignment
    for di in range(3):
        for dj in range(3):
            hess[i_starts + di, j_starts + dj] = h_blocks[:, :, di, dj]
    # Build diagonal blocks: H_ii = -sum(H_ij for j != i)
    # This can also be vectorized, but keeping simple loop for clarity
    for i in range(n_atoms):
        i_start, i_end = 3 * i, 3 * (i + 1)
        # Sum all off-diagonal blocks for atom i and negate
        diagonal_block = -np.sum(h_blocks[i, :, :, :], axis=0)  # Sum over j axis
        hess[i_start:i_end, i_start:i_end] = diagonal_block
    logger.info(f"Hessian matrix shape: {hess.shape}")
    logger.info(f"Hessian matrix range: {np.min(hess):.6f} to {np.max(hess):.6f}")
    logger.info(f"Number of non-zero connections: {np.count_nonzero(force_constants_matrix) // 2}")
    outfile = system.datdir / "enm_hess.npy"
    np.save(outfile, hess)
    logger.info(f"Saved ENM Hessian matrix to {outfile}")


def get_hessian_numerical(sysdir, sysname, delta=0.0001):
    """Calculate Hessian matrix numerically using finite differences of forces from OpenMM
    
    The Hessian H_ij = d²U/dx_i dx_j can be calculated as:
    H_ij ≈ -(F_i(x_j + δ) - F_i(x_j - δ)) / (2δ)
    where F_i is the force on coordinate i and δ is a small displacement.
    
    Args:
        sysdir: System directory
        sysname: System name
        delta: Finite difference step size in nanometers (default: 0.001 nm = 0.01 Å)
    
    Returns:
        hess: Numerical Hessian matrix (3N x 3N)
    """
    mdsys = MmSystem(sysdir, sysname)
    logger.info(f"Calculating numerical Hessian with finite difference step δ = {delta} nm")
    # Load system and structure
    pdb = app.PDBFile(str(mdsys.syspdb))
    system = _load_system_from_xml(mdsys.sysxml)
    # Create integrator and simulation (we only need context for energy/force calculations)
    integrator = mm.VerletIntegrator(0.001 * unit.picosecond)  # Step size doesn't matter for static calculations
    # Choose CPU platform (always uses double precision)
    platform = mm.Platform.getPlatformByName('CPU')
    properties = {}  # CPU platform is always double precision
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    context = simulation.context
    # Set initial positions and minimize
    logger.info("Setting positions and minimizing energy...")
    context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000, tolerance=1e-6)
    # Get minimized positions
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    n_atoms = len(positions)
    n_coords = 3 * n_atoms
    logger.info(f"Minimized structure with {n_atoms} atoms ({n_coords} coordinates)")
    # Initialize Hessian matrix
    hess = np.zeros((n_coords, n_coords))
    # Convert delta to OpenMM units
    delta_vec = np.zeros_like(positions)
    delta_openmm = delta * unit.nanometer
    logger.info("Calculating finite differences... This may take a while.")
    # Calculate Hessian using finite differences: H_ij = -dF_i/dx_j
    for j in range(n_coords):
        # Get atom and coordinate indices
        atom_j = j // 3
        coord_j = j % 3
        if j % 500 == 0:
            logger.info(f"Processing coordinate {j+1}/{n_coords}")
        # Forward displacement: x_j -> x_j + δ
        delta_vec[atom_j, coord_j] = delta
        pos_plus = positions + delta_vec
        context.setPositions(pos_plus * unit.nanometer)
        state_plus = context.getState(getForces=True)
        forces_plus = state_plus.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        # Backward displacement: x_j -> x_j - δ
        pos_minus = positions - delta_vec
        context.setPositions(pos_minus * unit.nanometer)
        state_minus = context.getState(getForces=True)
        forces_minus = state_minus.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        # Calculate finite difference: H_ij = -(F_i(+δ) - F_i(-δ)) / (2δ)
        force_diff = (forces_plus - forces_minus) / (2 * delta)
        # Fill Hessian column j
        for i in range(n_coords):
            atom_i = i // 3
            coord_i = i % 3
            hess[i, j] = -force_diff[atom_i, coord_i]  # Negative because F = -dU/dx
        # Reset displacement
        delta_vec[atom_j, coord_j] = 0
        # Reset to minimized positions for next iteration
        context.setPositions(positions * unit.nanometer)
    logger.info(f"Numerical Hessian calculation complete")
    logger.info(f"Hessian matrix shape: {hess.shape}")
    logger.info(f"Hessian matrix range: {np.min(hess):.6f} to {np.max(hess):.6f} kJ/mol/nm²")
    # Check symmetry
    symmetry_error = np.max(np.abs(hess - hess.T))
    logger.info(f"Hessian symmetry error: {symmetry_error:.6e} (should be small)")
    # Symmetrize the matrix (average with transpose to ensure perfect symmetry)
    hess = 0.5 * (hess + hess.T)
    # Save numerical Hessian
    outfile = mdsys.datdir / "num_hess.npy"
    np.save(outfile, hess)
    logger.info(f"Saved numerical Hessian matrix to {outfile}")


def get_per_residue_hessian(sysdir, sysname):
    """Calculate per-residue Hessian by averaging 3x3 blocks for each residue
    
    Args:
        sysdir: System directory
        sysname: System name  
        
    Returns:
        per_res_hess: Per-residue Hessian matrix (N_res x N_res)
    """
    mdsys = MmSystem(sysdir, sysname)
    hessfile = mdsys.datdir / "num_hess.npy"
    hess = np.load(hessfile)
    n_coords = hess.shape[0]
    n_atoms = n_coords // 3
    logger.info(f"Hessian shape: {hess.shape} ({n_atoms} atoms)")
    # Load all-atom structure to get residue information
    u = mda.Universe(str(mdsys.syspdb))
    atoms = u.atoms
    logger.info(f"Using all {len(atoms)} atoms for per-residue calculation")
    if len(atoms) != n_atoms:
        raise ValueError(f"Atom count mismatch: Hessian has {n_atoms} atoms, structure has {len(atoms)} atoms")
    # Get residue information
    residue_atom_indices = {}  # residue_id -> list of atom indices
    for i, atom in enumerate(atoms):
        resid = atom.resid
        if resid not in residue_atom_indices:
            residue_atom_indices[resid] = []
        residue_atom_indices[resid].append(i)
    # Create residue_info list
    residue_ids = sorted(residue_atom_indices.keys())
    residue_info = []
    for resid in residue_ids:
        atom_indices = residue_atom_indices[resid]
        resname = atoms[atom_indices[0]].resname  # Get resname from first atom
        residue_info.append((resname, resid, atom_indices))
    n_residues = len(residue_info)
    logger.info(f"Found {n_residues} residues")
    # Calculate per-residue Hessian (3N_res x 3N_res)
    per_res_hess = np.zeros((3 * n_residues, 3 * n_residues))
    logger.info("Calculating per-residue Hessian...")
    for i, (resname_i, resid_i, atom_indices_i) in enumerate(residue_info):
        for j, (resname_j, resid_j, atom_indices_j) in enumerate(residue_info):
            # Collect all 3x3 blocks between residues i and j
            blocks_3x3 = []
            for atom_i in atom_indices_i:
                for atom_j in atom_indices_j:
                    # Get 3x3 block from full Hessian
                    i_start, i_end = 3 * atom_i, 3 * (atom_i + 1)
                    j_start, j_end = 3 * atom_j, 3 * (atom_j + 1)
                    block_3x3 = hess[i_start:i_end, j_start:j_end]
                    blocks_3x3.append(block_3x3)
            # Average all 3x3 blocks between these two residues element-wise
            if blocks_3x3:
                avg_block = np.mean(blocks_3x3, axis=0)
                # Place averaged 3x3 block in per-residue Hessian
                res_i_start, res_i_end = 3 * i, 3 * (i + 1)
                res_j_start, res_j_end = 3 * j, 3 * (j + 1)
                per_res_hess[res_i_start:res_i_end, res_j_start:res_j_end] = avg_block
    logger.info(f"Per-residue Hessian shape: {per_res_hess.shape}")
    logger.info(f"Per-residue Hessian range: {np.min(per_res_hess):.6f} to {np.max(per_res_hess):.6f}")
    # Save per-residue Hessian
    outfile = mdsys.datdir / "per_residue_hess.npy"
    np.save(outfile, per_res_hess)
    logger.info(f"Saved per-residue Hessian to {outfile}")
    

def get_ca_hessian(sysdir, sysname):
    """Extract CA-only Hessian from full all-atom Hessian matrix
    
    Args:
        sysdir: System directory
        sysname: System name  
        hess: Full all-atom Hessian matrix from get_hessian_numerical()
        
    Returns:
        ca_hess: CA-only Hessian matrix (3N_CA x 3N_CA)
        ca_info: List of CA atom information (resname, resid, atom_index)
    """
    mdsys = MmSystem(sysdir, sysname)
    hessfile = mdsys.datdir / "num_hess.npy"
    hess = np.load(hessfile)
    n_coords = hess.shape[0]
    n_atoms = n_coords // 3
    logger.info(f"Full Hessian shape: {hess.shape} ({n_atoms} atoms)")
    # Load all-atom structure and extract CA atoms
    u = mda.Universe(str(mdsys.syspdb))
    all_atoms = u.atoms
    ca_atoms = u.select_atoms("name CA")
    logger.info(f"Total atoms: {len(all_atoms)}, CA atoms: {len(ca_atoms)}")
    if len(all_atoms) != n_atoms:
        raise ValueError(f"Atom count mismatch: Hessian has {n_atoms} atoms, structure has {len(all_atoms)} atoms")
    # Get CA atom indices in the full structure
    ca_indices = []
    ca_info = []
    for ca_atom in ca_atoms:
        # Find the index of this CA atom in the full atom list
        atom_index = ca_atom.index
        ca_indices.append(atom_index)
        ca_info.append((ca_atom.resname, ca_atom.resid, atom_index))
    n_ca = len(ca_indices)
    logger.info(f"Found {n_ca} CA atoms")
    # Extract CA-only Hessian matrix
    ca_hess = np.zeros((3 * n_ca, 3 * n_ca))
    logger.info("Extracting CA-only Hessian blocks...")
    for i, atom_i in enumerate(ca_indices):
        for j, atom_j in enumerate(ca_indices):
            # Get 3x3 block from full Hessian
            full_i_start, full_i_end = 3 * atom_i, 3 * (atom_i + 1)
            full_j_start, full_j_end = 3 * atom_j, 3 * (atom_j + 1)
            # Get 3x3 block for CA atoms i and j
            block_3x3 = hess[full_i_start:full_i_end, full_j_start:full_j_end]
            # Place in CA-only Hessian
            ca_i_start, ca_i_end = 3 * i, 3 * (i + 1)
            ca_j_start, ca_j_end = 3 * j, 3 * (j + 1)
            ca_hess[ca_i_start:ca_i_end, ca_j_start:ca_j_end] = block_3x3
    logger.info(f"CA Hessian shape: {ca_hess.shape}")
    logger.info(f"CA Hessian range: {np.min(ca_hess):.6f} to {np.max(ca_hess):.6f}")
    # Save CA-only Hessian
    outfile = mdsys.datdir / "ca_hess.npy"
    np.save(outfile, ca_hess)
    logger.info(f"Saved CA Hessian to {outfile}")
    

def enm_analysis(sysdir, sysname):
    """Calculate ENM-based metrics."""
    system = MDSystem(sysdir, sysname)
    in_pdb = system.syspdb
    u = mda.Universe(in_pdb)
    ag = u.select_atoms("name CA")
    # vecs = np.array(ag.positions).astype(np.float64) # (n_atoms, 3)
    # hess = mdm.hessian(vecs, spring_constant=5, cutoff=11, dd=0) # distances in Angstroms, dd=0 no distance-dependence
    hess = np.load(system.datdir / "ca_hess.npy")
    covmat = mdm.inverse_matrix(hess, device="gpu_dense", k_singular=6, n_modes=1000, dtype=np.float64)
    covmat = covmat * 1.25 # kb*T at 300K in kJ/mol
    outfile = system.datdir / "enm_cov.npy"
    np.save(outfile, covmat)
    pertmat = mdm.perturbation_matrix(covmat)
    rmsf = np.sqrt(np.diag(covmat).reshape(-1, 3).sum(axis=1)) * 10.0 # (n_atoms,)
    dfi = mdm.dfi(pertmat)
    plots.simple_residue_plot(system, [rmsf], outtag="enm_rmsf")
    plots.simple_residue_plot(system, [dfi], outtag="enm_dfi")

################################################################################
### Main ###
################################################################################

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