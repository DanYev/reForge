"""
Module for Martini simulation tools.

This module provides tools for preparing Martini simulations, such as topology file generation,
linking itp files, processing PDB files with GROMACS, and running various martinize2 routines.
Note that this module is intended for internal use.
"""

import os
import shutil
import warnings
from MDAnalysis import Universe
from MDAnalysis.analysis.dssp import translate, DSSP
from reforge import cli
from reforge.utils import cd, logger

warnings.filterwarnings("ignore", message="Reader has no dt information, set to 1.0 ps")

def dssp(in_file):
    """Compute the DSSP secondary structure for the given PDB file.

    Parameters
    ----------
    in_file : str
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
    in_file : str
        Path to the source file.
    out_file : str
        Path to the destination file.
    """
    with open(in_file, "r", encoding="utf-8") as src:
        lines = src.readlines()
    with open(out_file, "a", encoding="utf-8") as dest:
        dest.writelines(lines[1:])


def fix_go_map(wdir, in_map, out_map="go.map"):
    """Fix the Go-map file by removing the last column from lines that start with 'R '.

    Parameters
    ----------
    wdir : str
        Working directory.
    in_map : str
        Input map filename.
    out_map : str, optional
        Output map filename. Default is "go.map".
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    with open(in_map, "r", encoding="utf-8") as in_file:
        with open(out_map, "w", encoding="utf-8") as out_file:
            for line in in_file:
                if line.startswith("R "):
                    new_line = " ".join(line.split()[:-1])
                    out_file.write(new_line + "\n")
    os.chdir(bdir)


@cli.from_wdir
def martinize_go(wdir, topdir, aapdb, cgpdb, name="protein", go_eps=9.414,
                 go_low=0.3, go_up=1.1, go_res_dist=3,
                 go_write_file="map/contacts.map", **kwargs):
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
        Protein name. Default is "protein".
    go_eps : float, optional
        Strength of the Go-model bias. Default is 9.414.
    go_low : float, optional
        Lower distance cutoff (nm). Default is 0.3.
    go_up : float, optional
        Upper distance cutoff (nm). Default is 1.1.
    go_res_dist : int, optional
        Minimum residue distance below which contacts are removed. Default is 3.
    go_write_file : str, optional
        Output file for Go-map. Default is "map/contacts.map".
    **kwargs :
        Additional keyword arguments.
    """
    kwargs.setdefault("f", aapdb)
    kwargs.setdefault("x", cgpdb)
    kwargs.setdefault("go", "")
    kwargs.setdefault("o", "protein.top")
    kwargs.setdefault("cys", 0.3)
    kwargs.setdefault("p", "all")
    kwargs.setdefault("pf", 1000)
    kwargs.setdefault("sep", " ")
    kwargs.setdefault("resid", "input")
    kwargs.setdefault("ff", "martini3001")
    kwargs.setdefault("maxwarn", "1000")
    kwargs.setdefault("from", "amber")
    # ss = dssp(aapdb)
    with cd(wdir):
        line = ("-name {} -go-eps {} -go-low {} -go-up {} -go-res-dis {} "
                "-go-write-file {} -dssp").format(
                    name, go_eps, go_low, go_up, go_res_dist, go_write_file)
        cli.run("martinize2", line, **kwargs)
        append_to("go_atomtypes.itp", os.path.join(topdir, "go_atomtypes.itp"))
        append_to("go_nbparams.itp", os.path.join(topdir, "go_nbparams.itp"))
        shutil.move(f"{name}.itp", os.path.join(topdir, f"{name}.itp"))


@cli.from_wdir
def martinize_en(wdir, aapdb, cgpdb, ef=700, el=0.0, eu=0.9, **kwargs):
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
    kwargs.setdefault("cys", 0.3)
    kwargs.setdefault("p", "all")
    kwargs.setdefault("pf", 1000)
    kwargs.setdefault("sep", "")
    kwargs.setdefault("resid", "input")
    kwargs.setdefault("ff", "martini3001")
    kwargs.setdefault("maxwarn", "1000")
    kwargs.setdefault("elastic", "")
    ss = dssp(aapdb)
    line = ("-ef {} -el {} -eu {} -ss {}").format(ef, el, eu, ss)
    with cd(wdir):
        cli.run("martinize2", line, **kwargs)


def martinize_nucleotide(wdir, aapdb, cgpdb, **kwargs):
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


def martinize_rna(wdir, **kwargs):
    """Run RNA coarse-graining using martinize_rna.

    Parameters
    ----------
    wdir : str
        Working directory.
    **kwargs :
        Additional parameters.
    """
    with cd(wdir):
        script = "reforge.martini.martinize_rna"
        cli.run("python3 -m", script, **kwargs)


def insert_membrane(**kwargs):
    """Insert a membrane using the insane tool.
    """
    script = "reforge.martini.insane3"
    cli.run("python3 -m", script, **kwargs)
