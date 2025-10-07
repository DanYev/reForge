import os
from pathlib import Path
import sys
import shutil
import numpy as np
import cupy as cp
import MDAnalysis as mda
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import clean_dir, get_logger

logger = get_logger(__name__)


def setup(*args):
    setup_cg_protein_membrane(*args)


def setup_cg_protein_membrane(sysdir, sysname):
    ### FOR CG PROTEIN+LIPID BILAYERS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files, prepare directories and clean input PDB
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    mdsys.sort_input_pdb(mdsys.sysdir / ".." / "egfr_v3.pdb") # sorts chains in the input file and returns mdsys.inpdb file
    label_segments(in_pdb=mdsys.inpdb, out_pdb=mdsys.inpdb) # label the segments in the input PDB file
    mdsys.clean_pdb_mm(pdb_file=mdsys.inpdb, add_missing_atoms=True, add_hydrogens=False, pH=7.0)
    # mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput='8\n 7\n', ignh='yes', renum='yes') # 8 for CHARMM, 6 for AMBER FF
    
    # 1.2. Splitting chains before the coarse-graining and cleaning if needed.
    mdsys.split_chains()
    # mdsys.clean_chains_gmx(clinput='8\n 7\n', ignh='yes', renum='yes')
    # mdsys.clean_chains_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)  # if didn't work for the whole PDB
    # mdsys.get_go_maps(append=True)

    # # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=700, el=0.0, eu=0.9, p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=9.414, go_low=0.3, go_up=1.1, from_ff='amber', p='backbone', pf=500, append=False) # Martini + Go-network FF
    mdsys.make_cg_topology(add_resolved_ions=False, prefix='chain') # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG topology. Returns mdsys.solupdb ("solute.pdb") file
    label_segments(in_pdb=mdsys.solupdb, out_pdb=mdsys.solupdb) # label the segments in the CG PDB file 

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
    mdsys.make_system_ndx(backbone_atoms=["BB"])


def label_segments(in_pdb, out_pdb):
    def get_domain_label(resid):
        if 1 <= resid <= 165:
            return "EC1"  # Extracellular Domain I
        elif 166 <= resid <= 310:
            return "EC2"  # Extracellular Domain II
        elif 311 <= resid <= 480:
            return "EC3"  # Extracellular Domain III
        elif 481 <= resid <= 621:
            return "EC4"  # Extracellular Domain IV
        elif 622 <= resid <= 644:
            return "TM"  # Transmembrane domain
        elif 645 <= resid <= 682:
            return "JM"  # Juxtamembrane segment
        elif 683 <= resid <= 711:
            return "UNK"  # Undefined/Not clearly assigned
        elif 712 <= resid <= 978:
            return "KDN"  # Kinase domain
        elif 979 <= resid <= 995:
            return "CT"  # C-terminal tail
        else:
            return "UNK "
    logger.info("Relabelling Segment IDs")
    atoms = io.pdb2atomlist(in_pdb)
    for atom in atoms:
        label = get_domain_label(atom.resid)
        if label != 'KDN':
            atom.segid = label + atom.chid
        else:
            atom.segid = label 
    atoms.write_pdb(out_pdb)


def md(sysdir, sysname, runname, ntomp): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    # Choose appropriate mdp files
    em_mdp = os.path.join(mdrun.mdpdir, 'em_cgmem.mdp')
    eq_mdp = os.path.join(mdrun.mdpdir, 'eq_cgmem.mdp')
    md_mdp = os.path.join(mdrun.mdpdir, 'md_cgmem.mdp')
    mdrun.empp(f=em_mdp) # Preprocessing 
    mdrun.mdrun(deffnm='em', ntomp=ntomp) # Actual run
    mdrun.eqpp(f=eq_mdp, c='em.gro', r='em.gro', maxwarn=10) 
    mdrun.mdrun(deffnm='eq', ntomp=ntomp)
    mdrun.mdpp(f=md_mdp, c='eq.gro', r='eq.gro')
    mdrun.mdrun(deffnm='md', ntomp=ntomp)


def extend(sysdir, sysname, runname, ntomp):    
    mdsys = GmxSystem(sysdir, sysname)
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.mdrun(deffnm='md', cpi='md.cpt', ntomp=ntomp, nsteps=-2) 


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
    kwargs.setdefault('b',  500000) # in ps
    kwargs.setdefault('dt', 200) # in ps
    kwargs.setdefault('e', 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    # 2 for backbone
    mdrun.rmsf(clinput='2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='yes', res='yes', **kwargs)
    mdrun.rmsd(clinput='2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='rot+trans', **kwargs)

    
def cluster(sysdir, sysname, runname, **kwargs):
    mdrun = GmxRun(sysdir, sysname, runname)
    clean_dir(mdrun.cludir, 'trajout_Cluster*')
    ent_1 = 'seg_KDN'
    ent_2 = 'seg_KDN'
    mdrun.cluster(clinput=f'{ent_1}\n {ent_2}\n', n=mdrun.sysndx,
        b=200000, e=10000000, dt=1000,
        cutoff=0.23, method='linkage', nlevels='40',
        cl='clusters.pdb', clndx='cluster.ndx', av='yes')
    mdrun.extract_cluster()
    clean_dir(mdrun.cludir)


def cov_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname) 
    clean_dir(mdrun.covdir, '*npy')
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms.select_atoms("name BB or name BB1 or name BB3")
    # Begin at 'b' picoseconds, end at 'e', split into 'n' parts, sample each 'sample_rate' frame
    mdrun.get_covmats(u, ag, b=1000000, e=5000000, n=40, sample_rate=1, outtag='covmat') 
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag='dfi')
    mdrun.get_dci(outtag='dci', asym=False)
    mdrun.get_dci(outtag='asym', asym=True)
    # Calc DCI between segments
    atoms = io.pdb2atomlist(mdrun.solupdb)
    backbone_anames = ["BB"]
    bb = atoms.mask(backbone_anames, mode='name')
    bb.renum() # Renumber atids form 0, needed to mask numpy arrays
    groups = bb.segments.atids # mask for the arrays
    labels = [segids[0] for segids in bb.segments.segids]
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, outtag='dci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, transpose=True, outtag='tdci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=True, outtag='asym')
    # clean_dir(mdrun.covdir, 'covmat*')


def nm_analysis(sysdir, sysname):
    # DOES NOT WORK PROPERLY YET! BUT STILL KINDA USEFUL.
    logger.setLevel(logging.DEBUG)
    from animate import make_nm_animation
    mdsys = GmxSystem(sysdir, sysname)
    # getting eigensystem
    covmat_file = mdsys.datdir / 'covmat_av.npy'
    covmat = np.load(covmat_file)
    covmat_gpu = cp.array(covmat)
    evals_gpu, evecs_gpu = cp.linalg.eigh(covmat_gpu)
    evals = evals_gpu.get()
    evecs = evecs_gpu.get()
    # structure
    u = mda.Universe(mdsys.solupdb, in_memory=True)
    ag = u.atoms.select_atoms("name BB")
    coords = np.array(ag.positions) * 10
    # animation
    for idx in range(20, 32):
        make_nm_animation(coords, evals[-idx], evecs[-idx], idx, outfile=mdsys.pngdir / f'nma_{idx}.mp4')
    

def get_averages(sysdir, sysname):
    # Calculate mean and std_err_of_mean for each metric from all runs
    mdsys = GmxSystem(sysdir, sysname)   
    mdsys.get_mean_sem(pattern='pertmat*.npy')
    mdsys.get_mean_sem(pattern='covmat*.npy')
    mdsys.get_mean_sem(pattern='dfi*.npy')
    mdsys.get_mean_sem(pattern='dci*.npy')
    mdsys.get_mean_sem(pattern='asym*.npy')
    mdsys.get_mean_sem(pattern='rmsf*.npy')
    mdsys.get_mean_sem(pattern='ggdci*.npy') # group-group DCI  
    mdsys.get_mean_sem(pattern='ggasym*.npy') # group-group DCI-ASYM 
    for segment in mdsys.segments: # by segment
        logger.info('Processing segment %s', {segment})
        mdsys.get_mean_sem(pattern=f'gdci_{segment}*.npy')
        mdsys.get_mean_sem(pattern=f'gtdci_{segment}*.npy')
        mdsys.get_mean_sem(pattern=f'gasym_{segment}*.npy')
    logger.info("Done!")

        
if __name__ == '__main__':
    from reforge.cli import run_command
    run_command()