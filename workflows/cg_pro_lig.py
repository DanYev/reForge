import logging
import os
import shutil
import numpy as np
import MDAnalysis as mda
import mdtraj
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Importing reforge modules...")
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun, get_ntomp
from reforge.utils import clean_dir
from reforge.forge.topology import Topology


# Global settings
DT = 0.020  # Time step in picoseconds
total_time = 1000  # Total simulation time in nanoseconds
NSTEPS = int(total_time * 1e3 / DT)  # Number of MD steps for production run


def setup(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)
    input_pdb = Path("structures") / f"{sysname}.pdb"
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    mdsys.clean_pdb_mm(input_pdb, add_missing_atoms=True, add_hydrogens=True, pH=7.0) # Generates Amber ff names in PDB

    # Martinizing
    molname = "protein_0"
    idr_regions = f"A-282:475 B-282:475"
    idr_regions = _get_idr_regions(mdsys.inpdb, min_length=5, idr_start=282, idr_end=475)
    idr_regions = " ".join([f"A-{r}" for r in idr_regions.split()]) + " " + " ".join([f"B-{r}" for r in idr_regions.split()])
    add_command = f"-water-bias -water-bias-eps idr:0.5 -id-regions {idr_regions}" # martinize2 -h for help
    # add_command = f"-id-regions {idr_regions}" # martinize2 -h for help
    if not idr_regions:
        add_command = ""
    shutil.copy(mdsys.inpdb, mdsys.prodir / f"{molname}.pdb")
    # mdsys.martinize_proteins_en(append=False) # SWITCH APPEND TO TRUE IF ALREADY DONE
    mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=1.2, ff="martini3001",
        p="backbone", pf="500",  text=add_command, append=False) 
    shutil.copy(mdsys.topdir / f"{molname}.itp", mdsys.topdir / "tmp.itp") 
    # shutil.copy(mdsys.topdir / "tmp.itp", mdsys.topdir / f"{molname}.itp") 

    # LIGANDS 
    anp_dir = mdsys.root / "ligands" / "ANP"
    for x in ["A", "B"]:
        Path(mdsys.root / "ligands"/ f"AN{x}").mkdir(parents=True, exist_ok=True)
        shutil.copy(anp_dir / "ANP.itp", mdsys.root / "ligands"/ f"AN{x}"/ f"AN{x}.itp")
        shutil.copy(anp_dir / "ANP.map", mdsys.root / "ligands"/ f"AN{x}"/ f"AN{x}.map")
    mdsys.martinize_ligands(input_pdb=input_pdb, ligands=["ANA", "ANB", "MG"], merge_with=molname)
    # mdsys.martinize_ligands(input_pdb=input_pdb, ligands=["ANP", "MG"], merge_with=molname)
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    _add_protein_ligand_bonds(mdsys, molname, ligand_bead_names=["N04", "N07", "D01", "MG"])
    
    # PROTEIN+WATER SYSTEMS:
    mdsys.make_box(d="2.0", bt="dodecahedron")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")

    # # FOR MEMBRANE SYSTEMS:
    # mdsys.insert_membrane(
    #     f=mdsys.solupdb, o=mdsys.sysgro, p=mdsys.systop, 
    #     x=20, y=20, z=26, dm=9, 
    #     u='POPC:1', l='POPC:1', sol='W',
    # )
    mdsys.gmx('editconf', f=mdsys.sysgro, o=mdsys.syspdb)
    mdsys.add_bulk_ions(conc=0.10, pname='NA', nname='CL')

    # 1.4. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])


def _get_idr_regions(input_pdb, min_length=3, idr_start=0, idr_end=1000):
    struct = mdtraj.load_pdb(input_pdb)
    dssp = mdtraj.compute_dssp(struct, simplified=False)
    coil_token = ' '
    idr_regions = []
    curr_region = False
    for i, ss in enumerate(dssp[0][idr_start:idr_end]):
        if ss == coil_token and not curr_region:  # Coil regions are considered IDRs
            curr_region = True
            idr_region_start = idr_start + i + 1
            continue
        if curr_region and ss != coil_token:
            curr_region = False
            idr_region_end = idr_start + i
            if idr_region_end - idr_region_start + 1 >= min_length:  # Only consider regions of length >= min_length
                idr_regions.append(f"{idr_region_start}:{idr_region_end}")
    idr_regions_str = " ".join(map(str, idr_regions))
    logger.info(f"Identified IDR regions in {input_pdb}: {idr_regions_str}")
    return idr_regions_str


def _add_protein_ligand_bonds(mdsys, molname, ligand_bead_names) -> None:
    """Find closest protein beads to specified ligand beads using solute.pdb.
    
    Parameters
    ----------
    mdsys : GmxSystem
        The molecular dynamics system object
    ligand_bead_names : list, optional
        List of ligand bead names (e.g., ["D01", "MG"]).
        If None, uses a default list.
    """
    u = mda.Universe(str(mdsys.solupdb))
    protein_atoms = u.select_atoms("name BB* or name SC*") # Martini backbone and sidechain beads
    restraints = [] 
    # For each ligand bead name, find matching atoms and their closest protein partner
    for bead_name in ligand_bead_names:
        ligand_beads = u.select_atoms(f"name {bead_name}")
        for ligand_bead in ligand_beads:
            lig_pos = ligand_bead.position
            # Find closest protein atom
            distances = 0.1 * np.array([np.linalg.norm(lig_pos - p.position) for p in protein_atoms])
            closest_idx = np.argmin(distances)
            closest_protein = protein_atoms[closest_idx]
            distance = distances[closest_idx]
            # Use serial numbers from PDB (1-indexed)
            ligand_id = ligand_bead.index + 1
            protein_id = closest_protein.index + 1
            restraints.append(((ligand_id, protein_id), (1, distance, 1000), "BONDED DISTANCE RESTRAINT"))
            logger.info(f"Bond: protein atom {protein_id} ({closest_protein.name}) <-> "
                        f"ligand atom {ligand_id} ({ligand_bead.name}), distance: {distance:.2f} nm")
    # Update topology with generated restraints
    itp_file = mdsys.topdir / f"{molname}.itp"
    target_topo = Topology.from_itp(itp_file)
    target_topo.bonds.extend(restraints)
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
    mdrun.mdrun(deffnm="md", ntomp=ntomp, nsteps=NSTEPS, ) # bonded="gpu")
    
    
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
    kwargs.setdefault("e", 1e7) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="topology.tpr")
    topology = "topology.tpr" # mdrun.solupdb
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="atom", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s=topology, f="conv.xtc", o="conv.xtc", pbc="nojump")
    mdrun.trjconv(clinput="0\n 0\n", s=topology, f="conv.xtc", o="topology.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s=topology, f="conv.xtc", o="samples.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)




if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()

    