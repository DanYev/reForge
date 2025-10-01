import inspect
import logging
import os
import sys
from pathlib import Path
import warnings
import MDAnalysis as mda
from MDAnalysis.transformations import fit_rot_trans
from MDAnalysis.coordinates.memory import MemoryReader
import openmm as mm
from openmm import app
from openmm.unit import nanometer, molar, kilojoules_per_mole, kelvin, bar, nanoseconds, femtoseconds, picosecond
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.gmxmd import GmxSystem
from reforge.mdsystem.mmmd import MmSystem, MmRun
from reforge.utils import clean_dir, logger
from bioemu.sample import main as sample

from config import (
    MARTINI, INPDB, TEMPERATURE, PRESSURE, 
    TOTAL_TIME, TSTEP, GAMMA, NOUT
)


def setup(*args):
    if not MARTINI:
        setup_aa(*args)
    else:
        setup_martini(*args)


def setup_aa(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
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
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    mdsys.sort_input_pdb(mdsys.sysdir / INPDB) # sorts chain and atoms in the input file and returns makes mdsys.inpdb file

    # 1.2 Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    mdsys.clean_pdb_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)
    mdsys.split_chains()
    # mdsys.clean_chains_mm(add_missing_atoms=False, add_hydrogens=False, pH=7.0)  # if didn"t work for the whole PDB

    # 1.3. COARSE-GRAINING. Done separately for each chain. If don"t want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=500, el=0.5, eu=1.0, p="none", append=False)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=0.8, p="none", append=False) # Martini + Go-network FF
    # mdsys.martinize_rna(elastic="no", ef=50, el=0.5, eu=1.3, p="none", append=True) # Martini RNA FF 
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    
    # 1.4. Coarse graining is *hopefully* done. Need to add solvent and ions
    mdsys.make_box(d="1.2", bt="dodecahedron")
    solvent = os.path.join(mdsys.root, "water.gro")
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")
    
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


def md_npt(sysdir, sysname, runname, ntomp): 
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    mdrun.prepare_files()
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    with open(str(mdsys.sysxml)) as f:
        system = mm.XmlSerializer.deserialize(f.read())
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
    mdrun.prepare_files()
    inpdb = mdsys.root / 'system.pdb'
    pdb = app.PDBFile(str(inpdb))
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
    # --- Reporters (energies to monitor drift) ---
    log_reporters = [
        app.StateDataReporter(
            str(mdrun.rundir / "md.log"), 1000, step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True
        ),
        app.StateDataReporter(
            sys.stderr, 1000, step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True
        ),
    ]
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