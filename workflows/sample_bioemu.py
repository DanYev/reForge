import logging
from pathlib import Path
import MDAnalysis as mda
from reforge.mdsystem.mdsystem import MDSystem, MDRun

logger = logging.getLogger("reforge")

################################################################################
### Bioemu ###
################################################################################

def sample_emu(sysdir, sysname, runname):
    from bioemu.sample import main as sample
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    input_pdb = mdrun.root / "inpdb.pdb"
    sequence = _pdb_to_seq(input_pdb, select="protein and chainID A")
    logger.info(f"Extracted sequence: {sequence}")
    sample(model_name="bioemu-v1.2", sequence=sequence, 
        num_samples=100, batch_size_100=10, output_dir=mdrun.rundir)


def _pdb_to_seq(pdb, select="protein"):
    u = mda.Universe(pdb)
    logger.info(f'Selecting {select} atoms for sequence extraction')
    atoms = u.select_atoms(select)
    seq = "".join(res.resname for res in atoms.residues)  # three-letter codes
    seq_oneletter = "".join(mda.lib.util.convert_aa_code(res.resname) for res in atoms.residues)
    return seq_oneletter


def initiate_systems_from_emu(*args):
    logger.info("Preparing directories from EMU samples")
    emu_dir = Path("systems") / "emu"
    newsys_dir = Path("systems") / "1btl_nve"
    samples = emu_dir / "samples.xtc"
    top = emu_dir / "topology.pdb"
    u = mda.Universe(top, samples)
    step = 10  # every 10 frames
    for i, ts in enumerate(u.trajectory[1::step]):
        idx = i 
        outdir = newsys_dir / f"sample_{idx:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        outpdb = outdir / "sample.pdb"
        with mda.Writer(outpdb, u.atoms.n_atoms) as W:
            W.write(u.atoms)
        logger.info(f"Saved initial structure {i} to {outpdb}")


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()