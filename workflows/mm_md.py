import sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import fit_rot_trans
import openmm as mm
from openmm import app, unit
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter
from reforge.utils import clean_dir, get_logger

logger = get_logger()

# Global settings
INPDB = '1btl.pdb'
MARTINI=False  # True for CG systems, False for AA systems
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibraion
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
# Either steps or time
TOTAL_TIME = 1000 * unit.nanoseconds
TSTEP = 2 * unit.femtoseconds
TOTAL_STEPS = 100000 
# Reporting
TRJ_NOUT = 1000 # save every NOUT steps
CHK_NOUT = 100000 
LOG_NOUT = 1000 
OUT_SELECTION = "protein" 
TRJEXT = 'trr' # 'xtc' or 'trr'
# Analysis
SELECTION = "protein" 


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
    steps_per_cycle = 1000
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
    mdrun.eq(simulation, n_cycles=100, steps_per_cycle=1000)
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
    reporters = _get_reporters(mdrun, append=False, prefix='md')
    simulation.reporters.extend(reporters)
    simulation.step(TOTAL_STEPS)  
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
    if TRJEXT == 'trr':
        traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.trr"), 
            reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    if TRJEXT == 'xtc':
        # traj_reporter = app.XTCReporter(str(mdrun.rundir / f"{prefix}.xtc"), 
        #     reportInterval=TRJ_NOUT)
        traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.xtc"), 
            reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    state_reporter = app.CheckpointReporter(str(mdrun.rundir / f"{prefix}.xml"), CHK_NOUT, writeState=True)
    return log_reporter, err_reporter, traj_reporter, state_reporter


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
    reporters = _get_reporters(mdrun, append=False, prefix='ext')
    simulation.reporters.extend(reporters)
    mdrun.extend(simulation, until_time=TOTAL_TIME)
    

def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # INPUT
    top = mdrun.rundir / "md.pdb"
    # top = mdrun.syspdb  # use original topology to avoid missing atoms
    traj = mdrun.rundir / f"md.{TRJEXT}"
    ext = mdrun.rundir / f"ext.{TRJEXT}"
    trajs = [traj]  # combine both md and ext
    if ext.exists():
        trajs.append(ext)
    # CONVERT
    conv_top = mdrun.rundir / "topology.pdb"
    conv_traj = mdrun.rundir / f"md_selection.{TRJEXT}"
    logger.info(f'Converting trajectory with selection: {SELECTION}')
    _trjconv_selection(trajs, top, conv_traj, conv_top, selection=SELECTION, step=1)
    # FIT + OUTPUT
    out_traj = mdrun.rundir / f"samples.{TRJEXT}"
    _trjconv_fit(conv_traj, conv_top, out_traj, transform_vels=TRJEXT=='trr')


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


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()