import os
import sys
import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter
from reforge.utils import clean_dir, get_logger

logger = get_logger(__name__)

# Global settings
INPDB = '1btl.pdb'
# ENM parameters
ENM_CUTOFF = 1.0 * unit.nanometer  # cutoff distance for ENM springs
ENM_FORCE_CONSTANT = 100.0 * unit.kilojoules_per_mole / (unit.nanometer**2)  # spring constant
# Production parameters
TEMPERATURE = 300 * unit.kelvin
GAMMA = 1 / unit.picosecond
# Either steps or time
TOTAL_TIME = 100 * unit.picoseconds
TSTEP = 2 * unit.femtoseconds
TOTAL_STEPS = 50000
# Reporting: save every NOUT steps
TRJ_NOUT = 1000
LOG_NOUT = 1000
CHK_NOUT = 10000
OUT_SELECTION = "name CA"
TRJEXT = 'trr'


def setup(*args):
    """Main setup function for ENM system"""
    print(f"Setup called with args: {args}", file=sys.stderr)
    setup_enm(*args)


def extract_ca_positions(pdb_file):
    """Extract CA positions from PDB file and create a mock CA-only structure"""
    logger.info(f"Reading PDB file: {pdb_file}")
    
    # Use MDAnalysis to extract CA atoms
    u = mda.Universe(str(pdb_file))
    ca_atoms = u.select_atoms("name CA")
    
    if len(ca_atoms) == 0:
        raise ValueError("No CA atoms found in the PDB file")
    
    logger.info(f"Found {len(ca_atoms)} CA atoms")
    
    # Create positions array
    positions = ca_atoms.positions / 10.0  # Convert Angstrom to nanometers
    
    return ca_atoms, positions


def create_ca_topology(ca_atoms):
    """Create OpenMM topology with only CA atoms"""
    topology = app.Topology()
    
    # Create a single chain
    chain = topology.addChain()
    
    # Add CA atoms as residues
    for i, atom in enumerate(ca_atoms):
        residue = topology.addResidue(f"GLY", chain)  # Use GLY as placeholder
        topology.addAtom("CA", app.Element.getBySymbol("C"), residue)
    
    return topology


def add_enm_forces(system, positions, cutoff, force_constant):
    """Add Elastic Network Model harmonic forces based on cutoff distance"""
    logger.info(f"Adding ENM forces with cutoff {cutoff} and force constant {force_constant}")
    
    n_atoms = len(positions)
    positions_nm = positions * unit.nanometer
    
    # Create custom bond force for ENM springs
    enm_force = mm.CustomBondForce('k * (r - r0)^2')
    enm_force.addGlobalParameter('k', force_constant)
    enm_force.addPerBondParameter('r0')
    
    bonds_added = 0
    
    # Add bonds between atoms within cutoff distance
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Calculate distance between atoms i and j
            pos_i = positions_nm[i]
            pos_j = positions_nm[j]
            
            distance = np.sqrt(
                (pos_i[0] - pos_j[0])**2 + 
                (pos_i[1] - pos_j[1])**2 + 
                (pos_i[2] - pos_j[2])**2
            )
            
            # Add spring if within cutoff
            if distance <= cutoff:
                enm_force.addBond(i, j, [distance])
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
    carbon_mass = 12.01 * unit.amu
    for i in range(len(positions)):
        system.addParticle(carbon_mass)
    
    # Add ENM forces
    system = add_enm_forces(system, positions, ENM_CUTOFF, ENM_FORCE_CONSTANT)
    
    # Save system to XML file
    with open(mdsys.sysxml, "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved ENM system to {mdsys.sysxml}")
    
    # Save CA-only PDB file
    ca_pdb_file = mdsys.sysdir / f"{sysname}_ca_only.pdb"
    ca_atoms.write(str(ca_pdb_file))
    logger.info(f"Saved CA-only structure to {ca_pdb_file}")
    
    # Also save as system PDB for consistency
    ca_atoms.write(str(mdsys.syspdb))
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
    
    # Load CA-only structure
    pdb = app.PDBFile(str(mdsys.syspdb))
    
    # Create NVE integrator (Verlet)
    integrator = mm.VerletIntegrator(TSTEP)
    
    # Create simulation
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    # Minimize energy
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Set initial velocities
    logger.info("Setting initial velocities...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    
    # Setup reporters
    reporters = _get_reporters(mdrun, append=False, prefix="md_nve")
    simulation.reporters.extend(reporters)
    
    # Run simulation
    logger.info(f"Running NVE simulation for {TOTAL_STEPS} steps...")
    simulation.step(TOTAL_STEPS)
    
    logger.info("NVE simulation completed!")


def md_nvt(sysdir, sysname, runname):
    """Run NVT simulation with ENM forces"""
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: {mdrun.rundir}")
    
    # Load system and topology
    with open(mdsys.sysxml, 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    
    # Load CA-only structure
    pdb = app.PDBFile(str(mdsys.syspdb))
    
    # Create NVT integrator (Langevin)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, TSTEP)
    
    # Create simulation
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    # Minimize energy
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Set initial velocities
    logger.info("Setting initial velocities...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    
    # Setup reporters
    reporters = _get_reporters(mdrun, append=False, prefix="md_nvt")
    simulation.reporters.extend(reporters)
    
    # Run simulation
    logger.info(f"Running NVT simulation for {TOTAL_STEPS} steps...")
    simulation.step(TOTAL_STEPS)
    
    logger.info("NVT simulation completed!")


def analyze_enm_modes(sysdir, sysname):
    """Analyze ENM normal modes (optional analysis function)"""
    logger.info("ENM normal mode analysis could be implemented here")
    # This would require additional libraries like ProDy or custom implementation
    pass


if __name__ == "__main__":
    # Example usage
    sysdir = "/scratch/dyangali/reforge/workflows/systems/"
    sysname = "enm_system"

    setup(sysdir, sysname)
    md_nve(sysdir, sysname, "nve_run")
    md_nvt(sysdir, sysname, "nvt_run")
