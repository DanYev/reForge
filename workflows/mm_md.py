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
    mdsys.prepare_files()
    mdsys.clean_pdb(inpdb, add_missing_atoms=True, add_hydrogens=True)
    pdb = app.PDBFile(str(mdsys.inpdb))
    model = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    model.addSolvent(forcefield, 
        model='tip3p', 
        boxShape='dodecahedron',
        padding=1*nanometer,
        ionicStrength=0.1*molar,
        positiveIon='Na+',
        negativeIon='Cl-')
    system = forcefield.createSystem(model.topology, 
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*nanometer, 
        constraints=app.HBonds)
    _add_bb_restraints(system, pdb, bb_aname='CA')
    with open(mdsys.syspdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(model.topology, model.positions, file, keepIds=True)    
    with open(mdsys.sysxml, "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    

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


def md(sysdir, sysname, runname, ntomp): 
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
    traj = str(mdrun.rundir / "md.xtc")
    top = str(system.syspdb)
    step = 1
    if not MARTINI:
        selection = "name CA"
        out_selection = "name CA" 
        refpdb = system.root / "inpdb.pdb" 
    else:
        selection = "name BB*"
        out_selection = "name BB* or name SC* or name CA"
        refpdb = system.root / "solute.pdb" 
    # Read
    logger.info("Reading the trajectory")
    u = mda.Universe(top, traj)
    ag = u.atoms.select_atoms(out_selection)
    ref_u = mda.Universe(refpdb) # 
    ref_ag = ref_u.atoms.select_atoms(selection)
    # Create in-memory trajectory with only subset atoms
    logger.info("Copying the selection trajectory")
    coords = u.trajectory.timeseries(ag)[::step]
    u = mda.Merge(ag)  # new Universe with only those atoms
    ag = u.atoms
    u.load_new(coords, format=MemoryReader)
    # Align
    u.trajectory.add_transformations(fit_rot_trans(ag.select_atoms(selection), ref_ag,))
    # Write
    out_ag = ref_u.atoms.select_atoms(out_selection)
    out_ag.atoms.write(str(mdrun.rundir / "top.pdb"))
    ag = u.atoms.select_atoms(out_selection)
    logger.info("Converting/Writing Trajecory")
    with mda.Writer(str(mdrun.rundir / "mdc.xtc"), ag.n_atoms) as W:
        for ts in u.trajectory:   
            W.write(ag)
            if ts.frame % 1000 == 0:
                frame = ts.frame
                time_ns = ts.time // 1000
                logger.info(f"Current frame: %s at %s ns", frame, time_ns)
    logger.info("Done!")

        
def _get_platform_info():
    """Report OpenMM platform and hardware information.
    
    Returns
    -------
    dict
        Dictionary containing platform and hardware information
    """
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


def sample_emu(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    sequence = _pdb_to_seq(mdrun.sysdir / INPDB)
    sample(sequence=sequence, num_samples=1000, batch_size_100=20, output_dir=mdrun.rundir)


def _pdb_to_seq(pdb):
    u = mda.Universe(pdb)
    protein = u.select_atoms("protein")
    seq = "".join(res.resname for res in protein.residues)  # three-letter codes
    seq_oneletter = "".join(mda.lib.util.convert_aa_code(res.resname) for res in protein.residues)
    return seq_oneletter


