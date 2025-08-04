"""File: io.py

Description:
    This module provides I/O utilities for the reForge workflow, including functions 
    for reading and saving various data formats (e.g., CSV, NPY, XVG) as well as parsing 
    PDB files into domain-specific objects. Additionally, it offers helper functions for 
    filtering file names and recursively retrieving file paths from directories.

Usage Example:
    >>> from io import read_positions, pdb2system, npy2csv
    >>> import numpy as np
    >>> # Read positions from an MDAnalysis Universe object 'u'
    >>> positions = read_positions(u, ag, time_range=(0, 500), sample_rate=1)
    >>> # Save a NumPy array to CSV format
    >>> data = np.random.rand(100, 10)
    >>> npy2csv(data, 'output.csv')
    >>> # Parse a PDB file into a System object
    >>> system = pdb2system('structure.pdb')

Requirements:
    - Python 3.x
    - NumPy
    - Pandas
    - pathlib
    - reForge utilities (timeit, memprofit, logger)
    - reForge pdbtools (AtomList, System, PDBParser)

Author: Your Name
Date: YYYY-MM-DD
"""

from pathlib import Path
import numpy as np
import pandas as pd
from reforge.utils import timeit, memprofit, logger
from reforge.pdbtools import AtomList, System, PDBParser

################################################################################
## Reading Trajectories with MDAnalysis
################################################################################

@timeit
@memprofit
def read_positions(u, ag, b=0, e=10000000, sample_rate=1, dtype=np.float32):
    """Extract and return positions from an MDAnalysis trajectory.

    This function reads the positions for a specified atom group from the 
    trajectory stored in an MDAnalysis Universe. It extracts frames starting 
    from index `b` up to index `e`, sampling every `sample_rate` frames, and 
    returns the coordinates in a flattened, contiguous 2D array.

    Parameters
    ----------
    u : MDAnalysis.Universe
        The MDAnalysis Universe containing the trajectory.
    ag : MDAnalysis.AtomGroup
        The atom group from which to extract positions.
    b : int, optional
        The starting frame index (default is 0).
    e : int, optional
        The ending frame index (default is 10000000).
    sample_rate : int, optional
        The sampling rate for frames (default is 1, meaning every frame is used).
    dtype : data-type, optional
        The data type for the returned array (default is np.float32).

    Returns
    -------
    np.ndarray
        A contiguous 2D array with shape (n_coords, n_frames) containing flattened 
        position coordinates.
    """

    logger.info("Reading positions...")
    arr = np.array(
        [ag.positions.flatten() for ts in u.trajectory[::sample_rate] if b < ts.time < e],
        dtype=dtype,
    )
    arr = np.ascontiguousarray(arr.T)
    logger.info("Done!")
    return arr


@timeit
@memprofit
def read_velocities(u, ag, b=0, e=10000000, sample_rate=1, dtype=np.float32):
    """Saimilar to the previous. Read and return velocities from an MDAnalysis trajectory."""
    logger.info("Reading velocities...")
    arr = np.array(
        [ag.velocities.flatten() for ts in u.trajectory[::sample_rate] if b < ts.time < e],
        dtype=dtype,
    )
    arr = np.ascontiguousarray(arr.T)
    logger.info("Done!")
    return arr


def parse_covar_dat(file, dtype=np.float32):
    """Parse a GROMACS covar.dat file into a covariance matrix.

    Parameters
    ----------
    file : str
        Path to the covar.dat file.
    dtype : data-type, optional
        The data type for the covariance matrix (default is np.float32).

    Returns
    -------
    np.ndarray
        A reshaped 2D covariance matrix of shape (3*resnum, 3*resnum), where resnum 
        is inferred from the file.
    """
    df = pd.read_csv(file, sep="\\s+", header=None)
    covariance_matrix = np.asarray(df, dtype=dtype)
    resnum = int(np.sqrt(len(covariance_matrix) / 3))
    covariance_matrix = np.reshape(covariance_matrix, (3 * resnum, 3 * resnum))
    return covariance_matrix


################################################################################
## File Filtering and Retrieval Functions
################################################################################

def fname_filter(f, sw="", cont="", ew=""):
    """Check if a file name matches the specified start, contained, and end patterns.

    Parameters
    ----------
    f : str
        The file name to check.
    sw : str, optional
        Required starting substring (default is an empty string).
    cont : str, optional
        Required substring to be contained in the name (default is an empty string).
    ew : str, optional
        Required ending substring (default is an empty string).

    Returns
    -------
    bool
        True if the file name satisfies all specified conditions; otherwise, False.
    """
    return f.startswith(sw) and cont in f and f.endswith(ew)


def filter_files(fpaths, sw="", cont="", ew=""):
    """Filter a list of file paths based on name patterns.

    Parameters
    ----------
    fpaths : list[Path]
        A list of pathlib.Path objects representing file paths.
    sw : str, optional
        Required starting substring (default is an empty string).
    cont : str, optional
        Required substring to be contained (default is an empty string).
    ew : str, optional
        Required ending substring (default is an empty string).

    Returns
    -------
    list[Path]
        A list of Path objects that match the specified filters.
    """
    return [f for f in fpaths if fname_filter(f.name, sw=sw, cont=cont, ew=ew)]


def pull_files(directory, pattern):
    r"""Recursively search for files in a directory matching a given pattern.

    Parameters
    ----------
    directory : str or Path
        The root directory to search.
    pattern : str
        The glob pattern to match files (e.g., \*.txt).

    Returns
    -------
    list[str]
        A list of absolute file paths (as strings) that match the pattern.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist or is not a directory.
    """
    base_path = Path(directory)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist or is not a directory."
        )
    result = [str(p) for p in base_path.rglob(pattern)]
    return sorted(result)


def pull_all_files(directory):
    """Recursively retrieve all files in the specified directory and its subdirectories.

    Parameters
    ----------
    directory : str or Path
        The directory to search.

    Returns
    -------
    list[str]
        A list of absolute file paths for all files found.
    """
    return pull_files(directory, pattern="*")


################################################################################
## Data Conversion and I/O Functions
################################################################################

def xvg2npy(xvg_path, npy_path, usecols=(0, 1)):
    """Convert a GROMACS XVG file to a NumPy binary file (.npy).

    Parameters
    ----------
    xvg_path : str
        Path to the input XVG file.
    npy_path : str
        Path where the output .npy file will be saved.
    usecols : list of int, optional
        Column indices to read from the XVG file (default is [0, 1]).

    Returns
    -------
    None
    """
    try:
        df = pd.read_csv(xvg_path, sep="\\s+", header=None, usecols=usecols)
    except Exception as exc:
        raise ValueError("Error reading XVG file") from exc
    data = np.squeeze(df.to_numpy().T)
    np.save(npy_path, data)


def pdb2system(pdb_path) -> System:
    """Parse a PDB file and return a System object.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.

    Returns
    -------
    System
        A System object representing the parsed PDB structure.
    """
    parser = PDBParser(pdb_path)
    return parser.parse()


def pdb2atomlist(pdb_path) -> AtomList:
    """Parse a PDB file and return an AtomList object.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.

    Returns
    -------
    AtomList
        An AtomList object containing the atoms from the PDB file.
    """
    atoms = AtomList()
    atoms.read_pdb(pdb_path)
    return atoms


def read_data(fpath):
    """Read data from a file (.csv, .npy, .dat, or .xvg) and return it as a NumPy array.

    Parameters
    ----------
    fpath : str
        Path to the data file.

    Returns
    -------
    np.ndarray
        The data loaded from the file.

    Raises
    ------
    ValueError
        If the file cannot be read properly or the data does not meet expected criteria.
    """
    ftype = Path(fpath).suffix[1:]
    if ftype == "npy":
        try:
            data = np.load(fpath)
        except Exception as exc:
            raise ValueError("Error loading npy file") from exc
    elif ftype in {"csv", "dat"}:
        try:
            df = pd.read_csv(fpath, sep="\\s+", header=None)
            data = np.squeeze(df.values)
            if data.shape[0] != 1104:
                raise ValueError("Data shape mismatch for csv/dat file")
        except Exception as exc:
            raise ValueError("Error reading csv/dat file") from exc
    elif ftype == "xvg":
        try:
            df = pd.read_csv(fpath, sep="\\s+", header=None, usecols=[1])
            data = np.squeeze(df.values)
            if data.shape[0] > 10000:
                raise ValueError("Data shape too large for xvg file")
        except Exception as exc:
            raise ValueError("Error reading xvg file") from exc
    else:
        raise ValueError("Unsupported file type")
    return data


def read_xvg(fpath, usecols=(0, 1)):
    """Read a GROMACS XVG file and return its contents as a Pandas DataFrame.

    Parameters
    ----------
    fpath : str
        Path to the XVG file.
    usecols : list of int, optional
        Column indices to read from the file (default is [0, 1]).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected columns from the XVG file.

    Raises
    ------
    ValueError
        If the file cannot be read.
    """
    try:
        df = pd.read_csv(fpath, sep="\\s+", header=None, usecols=usecols)
    except Exception as exc:
        raise ValueError("Error reading xvg file") from exc
    return df


def npy2csv(data, fpath):
    """Save a NumPy array to a file in either .csv or .npy format.

    Parameters
    ----------
    data : np.ndarray
        The data to be saved.
    fpath : str
        Path to the output file. The file extension determines the format 
        (.csv or .npy).

    Returns
    -------
    None
    """
    ftype = Path(fpath).suffix[1:]
    if ftype == "csv":
        df = pd.DataFrame(data)
        df.to_csv(fpath, index=False, header=None, float_format="%.3E", sep=",")


def save_1d_data(data, ids=None, fpath="dfi.xvg", sep=" "):
    """Save one-dimensional data in GROMACS XVG format.

    Parameters
    ----------
    data : list or np.ndarray
        The y-column data to be saved.
    ids : list or np.ndarray, optional
        The x-column data (e.g., indices). If not provided, defaults to a range 
        starting from 1.
    fpath : str, optional
        Path to the output file (default is 'dfi.xvg').
    sep : str, optional
        Field separator in the output file (default is a single space).

    Returns
    -------
    None
    """
    if ids is None:
        ids = np.arange(1, len(data) + 1).astype(int)
    df = pd.DataFrame({"ids": ids, "data": data})
    df.to_csv(fpath, index=False, header=None, float_format="%.3E", sep=sep)


def save_2d_data(data, fpath="dfi.xvg", sep=" "):
    """Save two-dimensional data in GROMACS XVG format.

    Parameters
    ----------
    data : list or np.ndarray
        The 2D data to be saved.
    ids : list, optional
        Optional identifiers (unused in this function; provided for interface 
        consistency).
    fpath : str, optional
        Path to the output file (default is 'dfi.xvg').
    sep : str, optional
        Field separator in the output file (default is a single space).

    Returns
    -------
    None
    """
    df = pd.DataFrame(data)
    df.to_csv(fpath, index=False, header=None, float_format="%.3E", sep=sep)
