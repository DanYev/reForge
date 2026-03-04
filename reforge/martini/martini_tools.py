"""
Module for Martini simulation tools.

This module provides tools for preparing Martini simulations, such as topology file generation,
linking itp files, processing PDB files with GROMACS, and running various martinize2 routines.
Note that this module is intended for internal use.
"""

import logging
import os
import shutil
import warnings
from pathlib import Path
from MDAnalysis import Universe
from MDAnalysis.analysis.dssp import translate, DSSP
from reforge import cli
from reforge.martini import martinize_rna
from reforge.martini.martinize_rna import martinize_rna
from reforge.utils import cd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="Reader has no dt information, set to 1.0 ps")


def dssp(in_file):
    """Compute the DSSP secondary structure for the given PDB file.

    Parameters
    ----------
    in_file : str or Path
        Path to the PDB file.

    Returns
    -------
    str
        Secondary structure string with '-' replaced by 'C'.
    """
    logger.info("Doing DSSP")
    u = Universe(in_file)
    run = DSSP(u).run()
    mean_secondary_structure = translate(run.results.dssp_ndarray.mean(axis=0))
    ss = "".join(mean_secondary_structure).replace("-", "C")
    return ss


def append_to(in_file, out_file):
    """Append the contents of in_file (excluding the first line) to out_file.

    Parameters
    ----------
    in_file : str or Path
        Path to the source file.
    out_file : str or Path
        Path to the destination file.
    """
    # Convert to Path objects
    in_path = Path(in_file)
    out_path = Path(out_file)
    logger.debug(f"Appending {in_path} to {out_path} (excluding first line)")
    
    # Create parent directory for output file if it doesn't exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(in_path, "r", encoding="utf-8") as src:
        lines = src.readlines()
    
    lines_appended = len(lines) - 1 if len(lines) > 1 else 0
    
    with open(out_path, "a", encoding="utf-8") as dest:
        dest.writelines(lines[1:])
    
    logger.debug(f"Appended {lines_appended} lines from {in_path} to {out_path}")


def fix_go_map(wdir, in_map, out_map="go.map"):
    """Fix the Go-map file by removing the last column from lines that start with 'R '.

    Parameters
    ----------
    wdir : str or Path
        Working directory.
    in_map : str or Path
        Input map filename.
    out_map : str or Path, optional
        Output map filename. Default is "go.map".
    """
    wdir_path = Path(wdir)
    in_map_path = wdir_path / in_map
    out_map_path = wdir_path / out_map
    logger.info(f"Fixing Go-map file: {in_map_path} -> {out_map_path}")
    if not in_map_path.exists():
        logger.error(f"Input map file does not exist: {in_map_path}")
        raise FileNotFoundError(f"Input map file does not exist: {in_map_path}")
    lines_processed = 0
    lines_modified = 0
    with open(in_map_path, "r", encoding="utf-8") as in_file:
        with open(out_map_path, "w", encoding="utf-8") as out_file:
            for line in in_file:
                lines_processed += 1
                if line.startswith("R "):
                    new_line = " ".join(line.split()[:-1])
                    out_file.write(new_line + "\n")
                    lines_modified += 1
                else:
                    out_file.write(line)
    logger.info(f"Go-map fix completed: {lines_processed} lines processed, {lines_modified} lines modified")
    logger.info(f"Output written to: {out_map_path}")


def run_martinize_go(wdir, topdir, aapdb, cgpdb, name="protein_0", 
    go_eps=9.414, go_low=0.3, go_up=1.1, go_res_dist=3, from_ff="amber", **kwargs):
    """Run virtual site-based GoMartini via martinize2.

    Parameters
    ----------
    wdir : str
        Working directory.
    topdir : str
        Topology directory.
    aapdb : str
        Input all-atom PDB file.
    cgpdb : str
        Coarse-grained PDB file.
    name : str, optional
        Protein name. Default is "protein_0 ".
    go_eps : float, optional
        Strength of the Go-model bias. Default is 9.414.
    go_low : float, optional
        Lower distance cutoff (nm). Default is 0.3.
    go_up : float, optional
        Upper distance cutoff (nm). Default is 1.1.
    go_res_dist : int, optional
        Minimum residue distance below which contacts are removed. Default is 3.
    **kwargs :
        Additional keyword arguments.
    """
    kwargs.setdefault("f", aapdb)
    kwargs.setdefault("x", cgpdb)
    kwargs.setdefault("go", "")
    kwargs.setdefault("o", "protein.top")
    kwargs.setdefault("cys", 0.3)
    kwargs.setdefault("p", "backbone")
    kwargs.setdefault("pf", "500")
    kwargs.setdefault("resid", "input")
    kwargs.setdefault("ff", "martini3001")
    kwargs.setdefault("from", from_ff)
    kwargs.setdefault("maxwarn", "1000")
    text = kwargs.pop("text", "")
    # Convert paths to Path objects
    wdir_path = Path(wdir)
    topdir_path = Path(topdir)
    logger.info(f"Starting martinize_go for protein '{name}'")
    logger.info(f"Topology directory: {topdir_path}")
    logger.info(f"Input AA PDB: {aapdb}")
    logger.info(f"Output CG PDB: {cgpdb}")
    go_write_path = wdir_path / "maps" / f"{name}.map"
    go_write_path.parent.mkdir(parents=True, exist_ok=True)
    relative_map_path = go_write_path.relative_to(wdir_path)
    line = ("-name {} -go-eps {} -go-low {} -go-up {} -go-res-dis {} "
            "-go-write-file {} -dssp").format(
                name, go_eps, go_low, go_up, go_res_dist, relative_map_path)
    kwline = " ".join([f"-{k} {v}" for k, v in kwargs.items()])
    line = f"{line} {kwline} {text}"
    with cd(wdir):
        logger.info(f"Running martinize2 %s", line)
        cli.run("martinize2", line)
        _handle_vsites(topdir, name, vsites_name="go")
        # Move protein itp file to topology directory
        protein_itp_src = f"{name}.itp"
        protein_itp_dst = topdir / f"{name}.itp"
        shutil.move(protein_itp_src, protein_itp_dst)
        logger.debug(f"Moved {protein_itp_src} to {protein_itp_dst}")
    logger.info(f"martinize_go completed successfully for protein '{name}'")


def run_martinize_en(wdir, topdir, aapdb, cgpdb, name="protein_0", from_ff="amber", **kwargs):
    """Run protein elastic network generation via martinize2.

    Parameters
    ----------
    wdir : str
        Working directory.
    aapdb : str
        Input all-atom PDB file.
    cgpdb : str
        Coarse-grained PDB file.
    ef : float, optional
        Force constant. Default is 700.
    el : float, optional
        Lower cutoff. Default is 0.0.
    eu : float, optional
        Upper cutoff. Default is 0.9.
    **kwargs :
        Additional keyword arguments.
    """
    kwargs.setdefault("f", aapdb)
    kwargs.setdefault("x", cgpdb)
    kwargs.setdefault("o", "protein.top")
    kwargs.setdefault("merge", "all")
    kwargs.setdefault("cys", 0.3)
    kwargs.setdefault("p", "backbone")
    kwargs.setdefault("pf", "500")
    kwargs.setdefault("resid", "input")
    kwargs.setdefault("ff", "martini3001")
    kwargs.setdefault("from", from_ff)
    kwargs.setdefault("maxwarn", "1000")
    ef = kwargs.pop("ef", 700)
    el = kwargs.pop("el", 0.3)
    eu = kwargs.pop("eu", 0.9)
    text = kwargs.pop("text", "")
    # Convert paths to Path objects
    wdir = Path(wdir)
    topdir = Path(topdir)
    logger.info(f"Starting martinize_en for protein %s", name)
    # ss = dssp(aapdb)
    line = ("-elastic -ef {} -el {} -eu {} -dssp").format(ef, el, eu)
    kwline = " ".join([f"-{k} {v}" for k, v in kwargs.items()])
    line = f"{line} {kwline} {text}"
    with cd(wdir):
        logger.info(f"Running martinize2 %s", line)
        cli.run("martinize2", line)
        try:
            _handle_vsites(topdir, name, vsites_name="virtual_sites")
        except Exception as e:
            logger.warning(f"Virtual site files not found for protein '{name}': {e}")
            logger.warning("Proceeding without handling virtual sites.")
    logger.info(f"martinize_en completed successfully for protein '{name}'")


def _handle_vsites(topdir, name, vsites_name="go"):
    vs_atomtypes_src = Path(f"{vsites_name}_atomtypes.itp")
    vs_atomtypes_dst = topdir / f"{vsites_name}_atomtypes.itp"
    vs_nbparams_src = Path(f"{vsites_name}_nbparams.itp")
    if not vs_nbparams_src.exists():
        vs_nbparams_src = Path(f"{vsites_name}_nonbond_params.itp")
    vs_nbparams_dst = topdir / f"{vsites_name}_nbparams.itp"
    logger.debug(f"Moving files to topology directory: {topdir}")
    append_to(vs_atomtypes_src, vs_atomtypes_dst)
    append_to(vs_nbparams_src, vs_nbparams_dst)


def run_martinize_nucleotide(wdir, aapdb, cgpdb, **kwargs):
    """Run nucleotide coarse-graining using martinize_nucleotides.

    Parameters
    ----------
    wdir : str
        Working directory.
    aapdb : str
        Input all-atom PDB file.
    cgpdb : str
        Coarse-grained PDB file.
    **kwargs :
        Additional parameters.
    """
    kwargs.setdefault("f", aapdb)
    kwargs.setdefault("x", cgpdb)
    kwargs.setdefault("sys", "RNA")
    kwargs.setdefault("type", "ss")
    kwargs.setdefault("o", "topol.top")
    kwargs.setdefault("p", "bb")
    kwargs.setdefault("pf", 1000)
    with cd(wdir):
        script = "reforge.martini.martinize_nucleotides"
        cli.run("python3 -m", script, **kwargs)


def run_martinize_rna(wdir, **kwargs):
    """Run RNA coarse-graining using martinize_rna.

    Parameters
    ----------
    wdir : str
        Working directory.
    **kwargs :
        Additional parameters.
    """
    with cd(wdir):
        martinize_rna(**kwargs)


def insert_membrane(**kwargs):
    """Insert a membrane using the insane tool.
    """
    logger.info("Inserting membrane using INSANE")
    script = "reforge.martini.insane3"
    cli.run("python3 -m", script, **kwargs)
