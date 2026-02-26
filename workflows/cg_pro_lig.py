import logging
import os
import shutil
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from reforge import logger
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun, get_ntomp
from reforge.utils import clean_dir
from reforge.forge.topology import Topology

# logger = logging.getLogger("reforge")

# Global settings
INPDB = 'KDA.pdb'
DT = 0.020  # Time step in picoseconds
total_time = 1000  # Total simulation time in nanoseconds
NSTEPS = int(total_time * 1e3 / DT)  # Number of MD steps for production run

def workflow(sysdir, sysname, runname):
    md_npt(sysdir, sysname, runname, nsteps=NSTEPS)
    trjconv(sysdir, sysname, runname)


def setup(*args):
    setup_martini(*args)


def setup_martini(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)
    input_pdb = Path(sysdir) / INPDB
    # 1.1. Need to copy force field and md-parameter files and prepare PDBs and directories
    # mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    # mdsys.clean_pdb_mm(input_pdb, add_missing_atoms=False, add_hydrogens=False, pH=7.0) # Generates Amber ff names in PDB
    # mdsys.clean_pdb_gmx(input_pdb, clinput="8\n 7\n", ignh="no", renum="yes") # 8 for CHARMM, sometimes you need to refer to AMBER FF
    # mdsys.split_chains()
    
    # 1.2. COARSE-GRAINING. Done separately for each chain. 
    # If don"t want to split some of them, it needs to be done manually.
    add_command = "-water-bias -water-bias-eps H:3.6" 
    mdsys.martinize_proteins_en(ef=1000, el=0.3, eu=0.9, text=add_command, append=False)  # Martini + Elastic network FF 
    # mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=1.0, text="-water-bias", append=False) # MAKES chain_A.itp to merge ligands into later
    exit()
    # shutil.copy(mdsys.topdir / "chain_A.itp", mdsys.topdir / "tmp.itp") 
    shutil.copy(mdsys.topdir / "tmp.itp", mdsys.topdir / "chain_A.itp") 

    # LIGANDS [list of lists of (ATOM1, ATOM2, DISTANCE, FORCE_CONSTANT) tuples for each ligand]
    shutil.copy("/home/dyangali/LigPar/systems/ANP/mapping/ANP_updated_tmp.itp", 
      mdsys.root / "ligands"/ "ANP"/ "ANP.itp")
    shutil.copy("/home/dyangali/LigPar/systems/ANP/mapping/ANP.map", 
      mdsys.root / "ligands"/ "ANP"/ "ANP.map")
    mdsys.martinize_ligands(input_pdb=input_pdb, ligands=["ANP", "MG"], merge_with="chain_A")
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    add_protein_ligand_bonds(mdsys, ligand_bead_names=["P04", "N05", "D01", "MG"])
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    
    # 1.3. Coarse graining is *hopefully* done. Need to add solvent and ions
    mdsys.make_box(d="1.2", bt="dodecahedron")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")

    # 1.4. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])


def add_protein_ligand_bonds(mdsys, ligand_bead_names) -> None:
    """Find closest protein beads to specified ligand beads using solute.pdb.
    
    Parameters
    ----------
    mdsys : GmxSystem
        The molecular dynamics system object
    ligand_bead_names : list, optional
        List of ligand bead names (e.g., ["204", "N06", "D01", "MG"]).
        If None, uses a default list.
    """
    # Load solute structure
    u = mda.Universe(str(mdsys.solupdb))
    
    # Get protein atoms (exclude CA virtual sites)
    protein_atoms = u.select_atoms("name BB* or name SC*") # Martini backbone and sidechain beads
    if len(protein_atoms) == 0:
        logger.warning("No protein atoms found")
        return
    
    restraints = []
    
    # For each ligand bead name, find matching atoms and their closest protein partner
    for bead_name in ligand_bead_names:
        ligand_atoms = u.select_atoms(f"name {bead_name}")
        
        if len(ligand_atoms) == 0:
            logger.warning(f"No ligand atoms found with name: {bead_name}")
            continue
        
        for lig_atom in ligand_atoms:
            lig_pos = lig_atom.position
            
            # Find closest protein atom
            distances = np.array([np.linalg.norm(lig_pos - p.position) for p in protein_atoms])
            closest_idx = np.argmin(distances)
            closest_protein = protein_atoms[closest_idx]
            
            distance_angstrom = distances[closest_idx]
            distance_nm = distance_angstrom / 10.0
            
            # Use serial numbers from PDB (1-indexed)
            ligand_id = lig_atom.index + 1
            protein_id = closest_protein.index + 1
            
            restraints.append(((ligand_id, protein_id), (1, distance_nm, 1000), "BONDED DISTANCE RESTRAINT"))
            logger.info(f"Bond: protein atom {protein_id} ({closest_protein.name}) <-> "
                       f"ligand atom {ligand_id} ({lig_atom.name}), distance: {distance_angstrom:.2f} Å")
    
    if not restraints:
        logger.warning("No restraints generated")
        return
    
    # Update topology with generated restraints
    itp_file = mdsys.topdir / "chain_A.itp"
    target_topo = Topology.from_itp(itp_file)
    for restraint in restraints:
        target_topo.bonds.append(restraint)
    target_topo.write_to_itp(itp_file)
    logger.info("Saved topology with %d bonded restraints to %s", len(restraints), itp_file)

    
    
    
def md_npt(sysdir, sysname, runname, nsteps=None): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    ntomp = get_ntomp()
    mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.eqpp(f=mdrun.mdpdir / "eq_cg.mdp", c="em.gro", r="em.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=mdrun.mdpdir / "md_cg.mdp", maxwarn="1")    
    if nsteps is None:
        nsteps = NSTEPS
    mdrun.mdrun(deffnm="md", ntomp=ntomp, nsteps=nsteps, ) # bonded="gpu")
    
    
def extend(sysdir, sysname, runname, nsteps=None):    
    mdrun = GmxRun(sysdir, sysname, runname)
    ntomp = get_ntomp()
    if nsteps is None:
        t_ext = 10000 # nanoseconds
        nsteps = int(t_ext * 1e3 / DT)
    mdrun.mdrun(deffnm="md", cpi="md.cpt", ntomp=ntomp, nsteps=nsteps, ) 
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault("b", 0) # in ps
    kwargs.setdefault("dt", 200) # in ps
    kwargs.setdefault("e", 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    # mdrun.trjconv(clinput=f"0\n 0\n", s="eq.tpr", f="eq.gro", o="viz.pdb", n=mdrun.sysndx, pbc="atom", ur="compact", e=0)
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="topology.tpr")
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="cluster", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="topology.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="samples.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)




if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()

    