import os
import sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import fit_rot_trans
import openmm as mm
from openmm import app, unit
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories
from reforge.utils import clean_dir, get_logger

logger = get_logger(__name__)

# Global settings
INPDB = '1btl.pdb'
MARTINI=False  # True for CG systems, False for AA systems
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibraion
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
# Either steps or time
TOTAL_TIME = 100 * unit.picoseconds
TSTEP = 2 * unit.femtoseconds
TOTAL_STEPS = 100000 
# Reporting: save every NOUT steps
TRJ_NOUT = 1000     # normally you want 10000 or 100000 here
LOG_NOUT = 1000     # 100000 or more
CHK_NOUT = 100000 
OUT_SELECTION = "protein" 
TRJEXT = 'trr' # 'xtc' if don't need velocities or 'trr' if do
# Analysis and trjconv
SELECTION = "name CA" 


def setup(*args):
    print(f"Setup called with args: {args}", file=sys.stderr)
    if not MARTINI:
        setup_aa(*args)
    else:
        setup_martini(*args)


def setup_aa(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
    mdsys.prepare_files()
    mdsys.clean_pdb(inpdb, add_missing_atoms=True, add_hydrogens=True)
    pdb = app.PDBFile(str(mdsys.inpdb))
    model = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    logger.info("Adding solvent and ions")
    model.addSolvent(forcefield, 
        model='tip3p', 
        boxShape='dodecahedron', #  ‘cube’, ‘dodecahedron’, and ‘octahedron’
        padding=1.0 * unit.nanometer,
        ionicStrength=0.1 * unit.molar,
        positiveIon='Na+',
        negativeIon='Cl-')
    with open(mdsys.syspdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(model.topology, model.positions, file, keepIds=True)    
    logger.info("Saved solvated system to %s", mdsys.syspdb)


def _add_bb_restraints(system, pdb, bb_aname='CA'):
    restraint = mm.CustomExternalForce('bb_fc*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.setName('BackboneRestraint')
    restraint.addGlobalParameter('bb_fc', 1000.0*unit.kilojoules_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    system.addForce(restraint)
    for atom in pdb.topology.atoms():
        if atom.name == bb_aname:
            restraint.addParticle(atom.index, pdb.positions[atom.index])


def setup_martini(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    # here we need to run the CG setup from gmx_md first and then convert to OpenMM
    # 1.5. GMX -> OpenMM
    top_file = str(mdsys.systop)
    conf = app.GromacsGroFile(str(mdsys.sysgro))
    box_vectors = conf.getPeriodicBoxVectors()
    top = martini_openmm.MartiniTopFile(top_file, periodicBoxVectors=box_vectors, epsilon_r=15.0)
    system = top.create_system(nonbonded_cutoff=1.1*nanometer)
    pdb = app.PDBFile(str(mdsys.syspdb))
    _add_bb_restraints(system, pdb, bb_aname='BB')
    with open(mdsys.sysxml, "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))


def md_nve(sysdir, sysname, runname):
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    ff  = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    # --- Build a system WITHOUT any motion remover; no barostat/thermostat added ---
    logger.info("Generating topology...")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=False,     # important for strict NVE
        ewaldErrorTolerance=1e-5
    )
    # --- NVT integrator (for short equilibration) ---
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, 0.5*TSTEP)
    simulation = app.Simulation(pdb.topology, system, integrator) #  platform, properties)
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


def md_npt(sysdir, sysname, runname): 
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    ff  = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,)
    _add_bb_restraints(system, pdb, bb_aname='CA')
    integrator = mm.LangevinMiddleIntegrator(0, GAMMA, 0.5*TSTEP)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    reporters = _get_reporters(mdrun, prefix="eq")
    simulation.reporters.extend(reporters)
    # EM + HU
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    logger.info("Heating up...")
    n_cycles = 100
    steps_per_cycle = 100
    for i in range(n_cycles):
        simulation.integrator.setTemperature(TEMPERATURE*i/n_cycles)
        simulation.step(steps_per_cycle)
    simulation.saveState(str(mdrun.rundir / "hu.xml"))
    # Adding barostat + EQ
    logger.info("Equilibrating...")
    barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE)
    simulation.system.addForce(barostat)
    simulation.integrator.setTemperature(TEMPERATURE)
    simulation.context.reinitialize(preserveState=True)
    mdrun.eq(simulation, n_cycles=100, steps_per_cycle=100)
    # MD
    logger.info("Production...")
    simulation.loadState(str(mdrun.rundir / "eq.xml"))
    simulation.integrator.setStepSize(TSTEP)
    logger.info(f'Saving reference PDB with selection: {OUT_SELECTION}')
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb")
    simulation.reporters = []  # clear existing reporters
    reporters = _get_reporters(mdrun, append=False, prefix='md')
    simulation.reporters.extend(reporters)
    nsteps = int(TOTAL_TIME / TSTEP)
    simulation.step(nsteps)
    # simulation.step(TOTAL_STEPS)
    simulation.saveState(str(mdrun.rundir / "md.xml"))
    logger.info("Done!")


def _get_reporters(mdrun, append=False, prefix="md"):
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    log_reporter = app.StateDataReporter(
            str(mdrun.rundir / f"{prefix}.log"), 
            LOG_NOUT, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, speed=True, append=append)
    err_reporter =  app.StateDataReporter(
            sys.stderr, LOG_NOUT, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, speed=True, append=append)
    logger.info(f'Setting up trajectory reporter with selection: {OUT_SELECTION}')
    traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.{TRJEXT}"), 
            reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    state_reporter = app.CheckpointReporter(str(mdrun.rundir / f"{prefix}.xml"), CHK_NOUT, writeState=True)
    return log_reporter, err_reporter, traj_reporter, state_reporter


def _get_run_prefix(mdrun):
    existing_md = list(mdrun.rundir.glob(f"md*.{TRJEXT}")) 
    nums = [int(f.stem.split('md_')[-1]) for f in existing_md if 'md_' in f.stem]
    if not nums:
        curr_prefix = "md"
        return curr_prefix, f"{curr_prefix}_1"
    max_num = max(nums)
    curr_prefix = f"md_{max_num}"
    return curr_prefix, f"{curr_prefix}_{max_num+1}"


def extend(sysdir, sysname, runname):    
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    pdb = app.PDBFile(str(mdrun.syspdb))
    ff  = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,)
    barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, TSTEP)
    simulation = app.Simulation(pdb.topology, system, integrator)
    curr_prefix, next_prefix = _get_run_prefix(mdrun)
    logger.info(f"Current prefix for trajectory: {curr_prefix}, Next prefix: {next_prefix}")
    reporters = _get_reporters(mdrun, append=False, prefix=next_prefix)
    simulation.reporters.extend(reporters)
    mdrun.extend(simulation, curr_prefix=curr_prefix, next_prefix=next_prefix, until_time=1.2*TOTAL_TIME)


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
    logger.info(f'Converting trajectory with selection: {SELECTION}')
    convert_trajectories(top, trajs, out_top, out_traj, selection=SELECTION, step=1)
    logger.info("Done!")


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()