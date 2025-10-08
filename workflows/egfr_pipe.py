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
    mdsys.sort_input_pdb(mdsys.sysdir / "egfr_v3.pdb") # sorts chains in the input file and returns mdsys.inpdb file
    label_segments(in_pdb=mdsys.inpdb, out_pdb=mdsys.inpdb) # label the segments in the input PDB file
    # mdsys.clean_pdb_mm(mdsys.inpdb, add_missing_atoms=True, add_hydrogens=False, pH=7.0)
    mdsys.clean_pdb_gmx(mdsys.inpdb, clinput='8\n 7\n', ignh='no', renum='yes') # 8 for CHARMM, 6 for AMBER FF

    # 1.2. Splitting chains before the coarse-graining and cleaning if needed.
    mdsys.split_chains()
    # mdsys.get_go_maps(append=True)

    # # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    mdsys.martinize_proteins_en(ef=700, el=0.0, eu=0.9, from_ff='charmm', p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    # mdsys.martinize_proteins_go(go_eps=9.414, go_low=0.3, go_up=1.1, from_ff='charmm', p='backbone', pf=500, append=False) # Martini + Go-network FF
    mdsys.make_cg_topology(add_resolved_ions=False, prefix='chain') # CG topology. Returns mdsys.systop ("system.top") file
    mdsys.make_cg_structure() # CG topology. Returns mdsys.solupdb ("solute.pdb") file
    label_segments(in_pdb=mdsys.solupdb, out_pdb=mdsys.solupdb) # label the segments in the CG PDB file 

    # We can now insert the protein in a membrane. It may require a few attempts to get the geometry right.
    # Option 'dm' shifts the membrane along z-axis
    mdsys.insert_membrane(
        f=mdsys.solupdb, o=mdsys.sysgro, p=mdsys.systop, 
        x=14, y=14, z=22, dm=10, 
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
    logger.info("Relabelling Segment IDs")s
    atoms = io.pdb2atomlist(in_pdb)
    for atom in atoms:
        label = get_domain_label(atom.resid)
        if label != 'KDN':
            atom.segid = label + atom.chid
        else:
            atom.segid = label 
    atoms.write_pdb(out_pdb)

################################################################
### FOR THE MD AND ANALYSIS REFER TO gmx_md.py and common.py ###
################################################################
        
if __name__ == '__main__':
    from reforge.cli import run_command
    run_command()