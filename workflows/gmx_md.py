import inspect
import os
import sys
from pathlib import Path
import MDAnalysis as mda
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import clean_dir, logger

from config import MARTINI, INPDB


def setup(*args):
    if not MARTINI:
        setup_aa(*args)
    else:
        setup_martini(*args)


def setup_martini(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    mdsys.sort_input_pdb(mdsys.sysdir / INPDB) # sorts chain and atoms in the input file and returns makes mdsys.inpdb file

    # 1.2.1 Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    mdsys.clean_pdb_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)
    mdsys.split_chains()
    # mdsys.clean_chains_mm(add_missing_atoms=False, add_hydrogens=False, pH=7.0)  # if didn"t work for the whole PDB
    
    # # 1.2.2 Same but if we want Go-Model for the proteins
    # mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput="8\n 7\n", ignh="no", renum="yes") # 8 for CHARMM, sometimes you need to refer to AMBER FF
    # mdsys.split_chains()
    # mdsys.clean_chains_gmx(clinput="8\n 7\n", ignh="yes", renum="yes")
    # mdsys.get_go_maps(append=True)

    # 1.3. COARSE-GRAINING. Done separately for each chain. If don"t want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=1000, el=0.3, eu=0.9, p="backbone", pf=500, append=True)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=0.9, p="backbone", pf=1000, append=False) # Martini + Go-network FF
    # mdsys.martinize_rna(elastic="no", ef=50, el=0.5, eu=1.3, p="backbone", pf=500, append=True) # Martini RNA FF 
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    
    # # # 1.4. Coarse graining is *hopefully* done. Need to add solvent and ions
    mdsys.make_box(d="1.2", bt="dodecahedron")
    solvent = os.path.join(mdsys.root, "water.gro")
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")

    # # 1.5. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])
    
    
def md(sysdir, sysname, runname, ntomp): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.hupp(f=mdrun.mdpdir / "hu_cg.mdp", c="em.gro", r="em.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="hu", ntomp=ntomp)
    mdrun.eqpp(f=mdrun.mdpdir / "eq_cg.mdp", c="hu.gro", r="hu.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=mdrun.mdpdir / "md_cg.mdp", maxwarn="1")
    mdrun.mdrun(deffnm="md", ntomp=ntomp) # bonded="gpu")
    
    
def extend(sysdir, sysname, runname, ntomp):    
    mdrun = GmxRun(sysdir, sysname, runname)
    dt = 0.020 # picoseconds
    t_ext = 1000 # nanoseconds
    nsteps = int(t_ext * 1e3 / dt)
    mdrun.mdrun(deffnm="md", cpi="md.cpt", ntomp=ntomp, nsteps=nsteps, bonded="gpu") 
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault("b", 0) # in ps
    kwargs.setdefault("dt", 200) # in ps
    kwargs.setdefault("e", 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    # mdrun.trjconv(clinput=f"0\n 0\n", s="eq.tpr", f="eq.gro", o="viz.pdb", n=mdrun.sysndx, pbc="atom", ur="compact", e=0)
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="conv.tpr")
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="cluster", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s="conv.tpr", f="conv.xtc", o="top.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s="conv.tpr", f="conv.xtc", o="mdc.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)



    