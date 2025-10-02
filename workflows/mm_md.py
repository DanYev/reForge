import inspect
import logging
import os
import shutil
import sys
from pathlib import Path
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import fit_rot_trans
from MDAnalysis.coordinates.memory import MemoryReader
import openmm as mm
from openmm import app
import openmm.unit as unit
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter
from reforge.utils import clean_dir, logger
from bioemu.sample import main as sample


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global settings
INPDB = '1btl.pdb'
MARTINI=False  # True for CG systems, False for AA systems
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibraion
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
TOTAL_TIME = 10 * unit.picoseconds
TSTEP = 2 * unit.femtoseconds
NOUT = 1000 # save every NOUT steps
OUT_SELECTION = "name CA" 
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
    restraint.addGlobalParameter('bb_fc', 1000.0*kilojoules_per_mole/nanometer)
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
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    ff  = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=True, 
        ewaldErrorTolerance=1e-5
    )
    integrator = mm.LangevinMiddleIntegrator(0, GAMMA, 0.5*TSTEP)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    # EM + HU
    _get_platform_info()
    mdrun.em(simulation, tolerance=1, max_iterations=1000)
    mdrun.hu(simulation, TEMPERATURE, n_cycles=100, steps_per_cycle=1000)
    # Adding barostat + EQ
    barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE)
    simulation.system.addForce(barostat)
    simulation.integrator.setTemperature(TEMPERATURE)
    simulation.context.reinitialize(preserveState=True)
    mdrun.eq(simulation, n_cycles=100, steps_per_cycle=1000)
    # MD
    simulation.integrator.setStepSize(TSTEP)
    mdrun.md(simulation, time=TOTAL_TIME, nout=NOUT, velocities_needed=True)


def md_nve(sysdir, sysname, runname):
    # --- Inputs ---
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    ff  = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    # --- Build a system WITHOUT any motion remover; no barostat/thermostat added ---
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

    integrator.setConstraintTolerance(1e-6)
    simulation = app.Simulation(pdb.topology, system, integrator) #  platform, properties)

    simulation.reporters.extend(log_reporters)
    # --- Initialize state, minimize, equilibrate ---
    logger.info("Minimizing energy...")
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)  
    logger.info("Equilibrating...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)  # set initial kinetic energy
    simulation.step(10000)  # equilibrate for 10 ps
    # --- Run NVE (need to change the integrator and reset simulation) ---
    logger.info("Running NVE production...")
    integrator = mm.VerletIntegrator(TSTEP)
    integrator.setConstraintTolerance(1e-6)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setState(state)
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb") # SAVE PDB FOR THE SELECTION
    traj_reporter = MmReporter(str(mdrun.rundir / "md.trr"), reportInterval=NOUT, selection=OUT_SELECTION)
    simulation.reporters.append(traj_reporter)
    simulation.reporters.extend(log_reporters)
    simulation.step(100000)  
    logger.info("Done!")


def _get_reporters(log_nout=10000):
    log_reporters = [
        app.StateDataReporter(
            str(mdrun.rundir / "md.log"), log_nout, step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True
        ),
        app.StateDataReporter(
            sys.stderr, log_nout, step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True
        ),
    ]
    return log_reporters


def extend(sysdir, sysname, runname, ntomp):    
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    pdb = app.PDBFile(str(mdrun.syspdb))
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, TSTEP)
    with open(str(mdrun.sysxml)) as f:
        system = mm.XmlSerializer.deserialize(f.read())
    simulation = app.Simulation(pdb.topology, system, integrator)
    barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE)
    simulation.system.addForce(barostat)
    enum = enumerate(simulation.system.getForces()) 
    idx, bb_restraint = [(idx, f) for idx, f in enum if f.getName() == 'BackboneRestraint'][0]
    simulation.system.removeForce(idx)
    simulation.context.reinitialize(preserveState=True)
    mdrun.extend(simulation, until_time=TOTAL_TIME)
    

def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    traj = str(mdrun.rundir / "md.trr")
    top = str(mdrun.rundir / "md.pdb")
    # top = mdrun.syspdb  # use original topology to avoid missing atoms
    conv_top = str(mdrun.rundir / "topology.pdb")
    if SELECTION != OUT_SELECTION:
        conv_traj = str(mdrun.rundir / f"md_selection.trr")
        _trjconv_selection(traj, top, conv_traj, conv_top, selection=SELECTION, step=1)
    else:
        conv_traj = traj
        shutil.copy(top, conv_top)
    out_traj = str(mdrun.rundir / f"samples.trr")
    _trjconv_fit(conv_traj, conv_top, out_traj, transform_vels=True)


def _trjconv_selection(input_traj, input_top, output_traj, output_top, selection="name CA", step=1):
    u = mda.Universe(input_top, input_traj)
    selected_atoms = u.select_atoms(selection)
    n_atoms = selected_atoms.n_atoms
    selected_atoms.write(output_top)
    with mda.Writer(output_traj, n_atoms=n_atoms) as writer:
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
    with mda.Writer(output_traj, ag.n_atoms) as W:
        for ts in u.trajectory:   
            if transform_vels:
                transformed_vels = _tranform_velocities(ts.velocities, ts.positions, ref_ag.positions)
                ag.velocities = transformed_vels
            W.write(ag)
            if ts.frame % 1000 == 0:
                frame = ts.frame
                time_ns = ts.time // 1000
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mm_md.py <function_name> [args...]", file=sys.stderr)
        print("Available functions: setup, setup_aa, setup_martini, md_npt, md_nve, extend, trjconv", file=sys.stderr)
        sys.exit(1)
    
    function_name = sys.argv[1]
    args = sys.argv[2:]
    
    # Available functions mapping
    functions = {
        'setup': setup,
        'setup_aa': setup_aa,
        'setup_martini': setup_martini,
        'md_npt': md_npt,
        'md_nve': md_nve,
        'extend': extend,
        'trjconv': trjconv,
        'platform_info': _get_platform_info
    }
    
    if function_name not in functions:
        print(f"Error: Unknown function '{function_name}'", file=sys.stderr)
        print(f"Available functions: {', '.join(functions.keys())}", file=sys.stderr)
        sys.exit(1)
    
    func = functions[function_name]
    
    try:
        print(f"Calling {function_name} with args: {args}", file=sys.stderr)
        if args:
            func(*args)
        else:
            func()
        print(f"Successfully completed {function_name}", file=sys.stderr)
    except Exception as e:
        print(f"Error executing {function_name}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)