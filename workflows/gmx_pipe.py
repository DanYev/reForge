"""
Gmx Pipe Tutorial
=================

This module serves as a tutorial for setting up and running coarse-grained
molecular dynamics (MD) simulations using the reForge package and GROMACS. It
provides a pipeline that performs several key tasks:
    - System Setup: Copying force field and parameter files, preparing directories,
    cleaning and sorting the input PDB file, and splitting chains.
    - Coarse-Graining: Applying coarse-graining methods for proteins (using both
    Go-model and elastic network approaches) and nucleotides.
    - Solvation and Ion Addition: Solvating the system and adding ions to neutralize it.
    - MD Simulation: Running energy minimization, equilibration, and production MD runs.
    - Post-Processing: Converting trajectories, and performing
    analyses such as RMSF, RMSD, covariance analysis, clustering, and time-dependent
    correlation calculations.

Author: DY
Date: 2025-XX-XX
"""

import logging
import os
from pathlib import Path
import sys
import shutil
import numpy as np
import MDAnalysis as mda
from reforge import cli, io, mdm, pdbtools
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import *


def setup(*args):
    setup_cg_protein_rna(*args)
    # setup_cg_protein_membrane(*args)


def setup_cg_protein_rna(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    mdsys.prepare_files() # be careful it can overwrite later files
    mdsys.sort_input_pdb(f"{sysname}.pdb") # sorts chain and atoms in the input file and returns makes mdsys.inpdb file

    # 1.2.1 Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    mdsys.clean_pdb_mm(add_missing_atoms=False, add_hydrogens=True, pH=7.0)
    mdsys.split_chains()
    mdsys.clean_chains_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)  # if didn't work for the whole PDB
    
    # # 1.2.2 Same but if we want Go-Model for the proteins
    # mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput='8\n 7\n', ignh='no', renum='yes') # 8 for CHARMM, sometimes you need to refer to AMBER FF
    # mdsys.split_chains()
    # mdsys.clean_chains_gmx(clinput='8\n 7\n', ignh='yes', renum='yes')
    # mdsys.get_go_maps(append=True)

    # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=700, el=0.3, eu=0.8, p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p='backbone', pf=500, append=False) # Martini + Go-network FF
    mdsys.martinize_rna(ef=200, el=0.3, eu=1.2, p='backbone', pf=500, append=True) # Martini RNA FF 
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.make_box(d='1.2', bt='dodecahedron')

    # 1.4. Coarse graining is *hopefully* done. Need to add solvent and ions
    solvent = os.path.join(mdsys.wdir, 'water.gro')
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent) # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.15, pname='NA', nname='CL')

    # 1.5. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2", ])
   
      
def setup_cg_protein_membrane(sysdir, sysname):
    ### FOR CG PROTEIN+LIPID BILAYERS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories.
    mdsys.prepare_files() # be careful it can overwrite later files
    mdsys.sort_input_pdb(f"{sysname}.pdb") # sorts chain and atoms in the input file and returns makes mdsys.inpdb file
    label_segments(in_pdb=mdsys.inpdb, out_pdb=mdsys.inpdb)
    
    # # We may need to run mdsys.inpdb through OpenMM's or GROMACS' pdb tools but not always needed.
    # mdsys.clean_pdb_mm(add_missing_atoms=False, add_hydrogens=True, pH=7.0)
    # mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput='8\n 7\n', ignh='no', renum='yes') # 8 for CHARMM, 6 for AMBER FF
    
    # 1.2. Splitting chains before the coarse-graining and cleaning if needed.
    mdsys.split_chains()
    # mdsys.clean_chains_gmx(clinput='8\n 7\n', ignh='yes', renum='yes')
    # mdsys.clean_chains_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)  # if didn't work for the whole PDB
    # mdsys.get_go_maps(append=True)

    # # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=700, el=0.0, eu=0.9, p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=9.414, go_low=0.3, go_up=1.1, p='backbone', pf=500, append=True) # Martini + Go-network FF
    mdsys.make_cg_topology(add_resolved_ions=False, prefix='chain') # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG topology. Returns mdsys.solupdb ("solute.pdb") file
    label_segments(in_pdb=mdsys.solupdb, out_pdb=mdsys.solupdb)

    # We can now insert the protein in a membrane. It may require a few attempts to get the geometry right.
    # Option 'dm' shifts the membrane along z-axis
    mdsys.insert_membrane(
        f=mdsys.solupdb, o=mdsys.sysgro, p=mdsys.systop, 
        x=15, y=15, z=30, dm=10, 
        u='POPC:1', l='POPC:1', sol='W',
    )

    # 1.4 Insert membrane generates a .gro file but we want to have a .pdb so we will convert it first and then add ions to the box
    mdsys.gmx('editconf', f=mdsys.sysgro, o=mdsys.syspdb)
    mdsys.add_bulk_ions(conc=0.15, pname='NA', nname='CL')

    # 1.5. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 0.System 1.Solute 2.Backbone 3.Solvent 4. Not water. 5-. Chains. Custom groups can be added using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])


def md(sysdir, sysname, runname, ntomp): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    # Choose appropriate mdp files
    em_mdp = os.path.join(mdrun.mdpdir, 'em_cg.mdp')
    eq_mdp = os.path.join(mdrun.mdpdir, 'eq_cg.mdp')
    md_mdp = os.path.join(mdrun.mdpdir, 'md_cg.mdp')
    mdrun.empp(f=em_mdp) # Preprocessing 
    mdrun.mdrun(deffnm='em', ntomp=ntomp) # Actual run
    mdrun.eqpp(f=eq_mdp, c='em.gro', r='em.gro', maxwarn=10) 
    mdrun.mdrun(deffnm='eq', ntomp=ntomp)
    mdrun.mdpp(f=md_mdp, c='eq.gro', r='eq.gro')
    mdrun.mdrun(deffnm='md', ntomp=ntomp, bonded='gpu')


def extend(sysdir, sysname, runname, ntomp):    
    mdsys = GmxSystem(sysdir, sysname)
    mdrun = mdsys.initmd(runname)
    mdrun.mdrun(deffnm='md', cpi='md.cpt', bonded='gpu', ntomp=ntomp, nsteps=-2) 


def make_ndx(sysdir, sysname, **kwargs):
    mdsys = GmxSystem(sysdir, sysname)
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB2", ])


def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault('b', 0) # in ps
    kwargs.setdefault('dt', 200) # in ps
    kwargs.setdefault('e', 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    k = 1 # NDX groups: 0.System 1.Solute 2.Backbone 3.Solvent 4.Not water 5-.Chains
    mdrun.convert_tpr(clinput=f'{k}\n {k}\n', s='md.tpr', n=mdrun.sysndx, o='conv.tpr')
    mdrun.trjconv(clinput=f'{k}\n {k}\n {k}\n', s='md.tpr', f='md.xtc', n=mdrun.sysndx, o='conv.xtc',  
       pbc='nojump', **kwargs)
    mdrun.trjconv(clinput='0\n0\n0\n', s='conv.tpr', f='conv.xtc', o='mdc.xtc', fit='rot+trans')
    mdrun.trjconv(clinput='0\n0\n0\n', s='conv.tpr', f='conv.xtc', o='mdc.pdb', fit='rot+trans', e=0)
    clean_dir(mdrun.rundir)
    

def rms_analysis(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault('b',  50000) # in ps
    kwargs.setdefault('dt', 1000) # in ps
    kwargs.setdefault('e', 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    # 2 for backbonea
    mdrun.rmsf(clinput='2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='yes', res='yes', **kwargs)
    mdrun.rmsd(clinput='2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='rot+trans', **kwargs)

    
def cluster(sysdir, sysname, runname, **kwargs):
    mdrun = GmxRun(sysdir, sysname, runname)
    clean_dir(mdrun.cludir, 'trajout_Cluster*')
    k = 1 # k = 1 - solute, k = 2 - backbone.
    mdrun.cluster(clinput=f'{k}\n 2\n', n=mdrun.sysndx,
        b=00000, e=1000000, dt=200, 
        cutoff=0.15, method='linkage', av='yes', 
        cl='clusters.xtc', clndx='cluster.ndx', )
    mdrun.extract_cluster()
    clean_dir(mdrun.cludir)


def cov_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname) 
    clean_dir(mdrun.covdir, '*npy')
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms.select_atoms("name BB or name BB2")
    # Begin at 'b' picoseconds, end at 'e', split into 'n' parts, sample each 'sample_rate' frame
    mdrun.get_covmats(u, ag, b=100000, e=1000000, n=18, sample_rate=1, outtag='covmat') 
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag='dfi')
    mdrun.get_dci(outtag='dci', asym=False)
    mdrun.get_dci(outtag='asym', asym=True)
    # Calc DCI between segments
    atoms = io.pdb2atomlist(mdrun.solupdb)
    backbone_anames = ["BB", "BB2"]
    bb = atoms.mask(backbone_anames, mode='name')
    bb.renum() # Renumber atids form 0, needed to mask numpy arrays
    groups = bb.segments.atids # mask for the arrays
    labels = [segids[0] for segids in bb.segments.segids]
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, outtag='dci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, transpose=True, outtag='tdci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=True, outtag='asym')
    clean_dir(mdrun.covdir, 'covmat*')


def tdlrt_analysis(sysdir, sysname, runname):
    # WARNING! The computation can potentially be VERY VERY memory extensive.
    # Estimate the size of the output before running this function.
    # Dumping the whole trajectory can in theory output terabytes of data and will crash in practice.
    # In such cases, request more memory, bigger GPU, or do it segment by segment.
    logger.setLevel(logging.DEBUG)
    mdrun = GmxRun(sysdir, sysname, runname) 
    b = 500000
    e = 3000000
    sample_rate = 1
    ntmax = 100 # how many frames to save
    # Reading trajectory
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms.select_atoms("name BB")
    positions = io.read_positions(u, ag, b=b, e=e, sample_rate=sample_rate)
    # velocities = io.read_velocities(u, ag, b=b, e=e, sample_rate=sample_rate,)
    # CALC CCF # CCF params FRAME_DT=20 ps
    corr = mdm.ccf(positions, positions, ntmax=ntmax, n=10, mode='gpu', center=True, dtype=np.float32)
    corr_file = mdrun.lrtdir / 'corr_pp.npy'
    np.save(corr_file, corr)


def get_averages(sysdir, sysname):
    mdsys = GmxSystem(sysdir, sysname)   
    mdsys.get_mean_sem(pattern='pertmat*.npy')
    mdsys.get_mean_sem(pattern='covmat*.npy')
    mdsys.get_mean_sem(pattern='dfi*.npy')
    mdsys.get_mean_sem(pattern='dci*.npy')
    mdsys.get_mean_sem(pattern='asym*.npy')
    mdsys.get_mean_sem(pattern='rmsf*.npy')
    mdsys.get_mean_sem(pattern='ggdci*.npy') # group-group DCI    
    for chain in mdsys.chains:
        logger.info(f'Processing chain {chain}')
        mdsys.get_mean_sem(pattern=f'gdci_{chain}*.npy')


def get_td_averages(sysdir, sysname, loop=True, fname='corr_pv.npy'):
    """
    Need to loop for big arrays
    """
    mdsys = GmxSystem(sysdir, sysname)  
    print('Getting averages', file=sys.stderr)  
    files = io.pull_files(mdsys.mddir, fname)
    if loop:
        print(f'Processing {files[0]}', file=sys.stderr) 
        average = np.load(files[0])
        for f in files[1:]:
            print(f'Processing {f}', file=sys.stderr)  
            arr = np.load(f)
            average += arr
        average /= len(files)
    else:
        arrays = [np.load(f) for f in files]
        average = np.average(arrays, axis=0)
    np.save(os.path.join(mdsys.datdir, fname), average) 
    print('Done!', file=sys.stderr)  
    return average


def sysjob(sysdir, sysname):    
    pass


def runjob(sysdir, sysname, runname):    
    trjconv(sysdir, sysname, runname)
    rms_analysis(sysdir, sysname, runname)
    # get_averages(sysdir, sysname)

        
if __name__ == '__main__':
    command = sys.argv[1]
    args = sys.argv[2:]
    commands = {
        "setup": setup,
        "md": md,
        "extend": extend,
        "make_ndx": make_ndx,
        "trjconv": trjconv,
        "rms_analysis": rms_analysis,
        "cluster": cluster,
        "cov_analysis": cov_analysis,
        "tdlrt_analysis": tdlrt_analysis,
        "get_averages": get_averages,
        "get_td_averages": get_td_averages,
        "sysjob": sysjob,
        "runjob": runjob, 
    }
    if command in commands:   # Then, assuming `command` is the command name (a string)
        commands[command](*args) # `args` is a list/tuple of arguments
    else:
        raise ValueError(f"Unknown command: {command}")
   
