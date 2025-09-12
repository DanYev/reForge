"""Math for MD

Description:
    This module provides a unified interface for molecular dynamics and
    structural analysis routines within the reForge package. It wraps a variety
    of operations including FFT-based cross-correlation, covariance matrix
    computation, perturbation matrix calculations (for DFI/DCI metrics), and
    elastic network model (ENM) Hessian evaluations. Both CPU and GPU implementations
    are supported, with fallbacks to CPU methods if CUDA is not available.

Usage Example:
    >>> import numpy as np
    >>> from mdm import fft_ccf, calc_and_save_covmats, inverse_matrix
    >>>
    >>> # Compute FFT-based cross-correlation function in serial mode
    >>> ccf = fft_ccf(signal1, signal2, mode='serial')
    >>>
    >>> # Calculate and save covariance matrices from trajectory positions
    >>> calc_and_save_covmats(positions, outdir='./covmats', n=5)
    >>>
    >>> # Compute the inverse of a matrix using the unified inversion wrapper
    >>> inv_mat = inverse_matrix(matrix, device='cpu_sparse')

Requirements:
    - Python 3.x
    - MDAnalysis
    - NumPy
    - CuPy (if GPU routines are used)
    - Pandas
    - reForge utilities (logger, etc.)
    - reForge rfgmath modules (rcmath, rpymath)

Author: DY
Date: YYYY-MM-DD
"""

import os
import numpy as np
import cupy as cp
from reforge.utils import logger  
from reforge.rfgmath import rcmath, rpymath


def fft_ccf(*args, mode="serial", **kwargs):
    """
    Unified wrapper for FFT-based correlation functions.

    This function dispatches to one of the internal FFT correlation routines
    based on the specified mode:

        - 'serial' for sfft_ccf,
        - 'parallel' for pfft_ccf, or
        - 'gpu' for gfft_ccf.

    Parameters
    ----------
    *args :
        Positional arguments for the chosen correlation function.
    mode : str, optional
        Mode to use ('serial', 'parallel', or 'gpu'). Default is 'serial'.
    **kwargs :
        Additional keyword arguments for the internal routines.

    Returns
    -------
    np.ndarray
        The computed correlation function.

    Raises
    ------
    ValueError
        If an unsupported mode is specified.
    """
    if mode == "serial":
        return rpymath.sfft_ccf(*args, **kwargs)
    if mode == "parallel":
        return rpymath.pfft_ccf(*args, **kwargs)
    if mode == "gpu":
        result = rpymath.gfft_ccf(*args, **kwargs)
        # Use .get() only if available (e.g. for CuPy arrays)
        return result.get() if hasattr(result, "get") else result
    raise ValueError("Mode must be 'serial', 'parallel' or 'gpu'.")


def ccf(*args, **kwargs):
    """Similar to the previous. Unified wrapper for calculating cross-correlations."""
    corr = rpymath.ccf(*args, **kwargs)
    return corr


def covariance_matrix(positions, dtype=np.float64):
    """Compute the covariance matrix from trajectory positions.

    Parameters
    ----------
    positions : np.ndarray
        Array of position coordinates.
    dtype : data-type, optional
        Desired data type (default is np.float64).

    Returns
    -------
    np.ndarray
        The computed covariance matrix.
    """
    return rpymath.covariance_matrix(positions, dtype=dtype)


def calc_and_save_covmats(positions, outdir, n=1, outtag="covmat", dtype=np.float32):
    """Calculate and save covariance matrices by splitting a trajectory into segments.

    Parameters
    ----------
    positions : np.ndarray
        Array of positions with shape (n_coords, n_frames).
    outdir : str
        Directory where the covariance matrices will be saved.
    n : int, optional
        Number of segments to split the trajectory into (default is 1).
    outtag : str, optional
        Base tag for output file names (default is 'covmat').
    dtype : data-type, optional
        Desired data type for covariance computation (default is np.float32).

    Returns
    -------
    None
    """
    trajs = np.array_split(positions, n, axis=-1)
    for idx, traj in enumerate(trajs, start=1):
        logger.info("Processing covariance matrix %d", idx)
        covmat = covariance_matrix(traj, dtype=dtype)
        outfile = os.path.join(outdir, f"{outtag}_{idx}.npy")
        np.save(outfile, covmat)
        logger.info("Saved covariance matrix to %s", outfile)


def calc_and_save_rmsf(positions, outdir, n=1, outtag="rmsf", dtype=np.float64):
    """Calculate and save RMSF by splitting a trajectory into segments.

    Parameters
    ----------
    positions : np.ndarray
        Array of positions with shape (n_coords, n_frames).
    outdir : str
        Directory where the RMSF data will be saved.
    n : int, optional
        Number of segments to split the trajectory into (default is 1).
    outtag : str, optional
        Base tag for output file names (default is 'rmsf').
    dtype : data-type, optional
        Desired data type for covariance computation (default is np.float32).

    Returns
    -------
    None
    """
    trajs = np.array_split(positions, n, axis=-1)
    for idx, traj in enumerate(trajs, start=1):
        logger.info("Processing segment %d", idx)
        rmsf_comps = np.std(traj, axis=-1, dtype=dtype)
        rmsf_comps_reshaped = np.reshape(rmsf_comps, (len(rmsf_comps)//3, 3))
        rmsf_sq = np.sum(rmsf_comps_reshaped**2, axis=-1)
        rmsf = np.sqrt(rmsf_sq)
        outfile = os.path.join(outdir, f"{outtag}_{idx}.npy")
        np.save(outfile, rmsf)
        logger.info("Saved RMSF to %s", outfile)
    logger.info("Done!")

##############################################################
## DFI / DCI Calculations
##############################################################

def perturbation_matrix(covmat, dtype=np.float64):
    """Compute the perturbation matrix from a covariance matrix.

    Parameters
    ----------
    covmat : np.ndarray
        The covariance matrix.
    dtype : data-type, optional
        Desired data type (default is np.float64).

    Returns
    -------
    np.ndarray
        The computed perturbation matrix.
    """
    covmat = covmat.astype(np.float64)
    pertmat = rcmath.perturbation_matrix(covmat)
    return pertmat


def perturbation_matrix_iso(covmat, dtype=np.float64):
    """Compute the perturbation matrix from a covariance matrix"""
    covmat = covmat.astype(np.float64)
    pertmat = rcmath.perturbation_matrix_iso(covmat)
    return pertmat    


def td_perturbation_matrix(covmat, dtype=np.float64):
    """Compute the block-wise (td) perturbation matrix from a covariance matrix.

    Parameters
    ----------
    covmat : np.ndarray
        The covariance matrix.
    dtype : data-type, optional
        Desired data type (default is np.float64).

    Returns
    -------
    np.ndarray
        The computed block-wise perturbation matrix.
    """
    covmat = covmat.astype(np.float64)
    pertmat = rcmath.td_perturbation_matrix(covmat)
    return pertmat


def dfi(pert_mat):
    """Calculate the Dynamic Flexibility Index (DFI) from a perturbation matrix.

    Parameters
    ----------
    pert_mat : np.ndarray
        The perturbation matrix.

    Returns
    -------
    np.ndarray
        The DFI values.
    """
    dfi_val = np.average(pert_mat, axis=-1)
    return dfi_val


def dci(pert_mat, asym=False):
    """Calculate the Dynamic Coupling Index (DCI) from a perturbation matrix.

    Parameters
    ----------
    pert_mat : np.ndarray
        The perturbation matrix.
    asym : bool, optional
        If True, return an asymmetric version (default is False).

    Returns
    -------
    np.ndarray
        The DCI matrix.
    """
    dci_val = pert_mat / np.average(pert_mat, axis=-1, keepdims=True)
    if asym:
        dci_val = dci_val - dci_val.T
    return dci_val


def group_molecule_dci(pert_mat, groups=None, asym=False, transpose=False):
    """Calculate the DCI between groups of atoms and the remainder of the molecule.

    Parameters
    ----------
    pert_mat : np.ndarray
        The perturbation matrix.
    groups : list of lists, optional
        A list of groups, each containing indices of atoms.
        Defaults to a list containing an empty list.
    asym : bool, optional
        If True, use the asymmetric DCI (default is False).

    Returns
    -------
    list of np.ndarray
        A list of DCI values for each group.
    """
    if groups is None:
        groups = [[]]
    dcis = []
    dci_tot = pert_mat / np.sum(pert_mat, axis=-1, keepdims=True)
    if asym:
        dci_tot = dci_tot - dci_tot.T
    if transpose:
        dci_tot = dci_tot.T
    for ids in groups:
        top = np.sum(dci_tot[:, ids], axis=-1) * pert_mat.shape[0]
        bot = len(ids)
        dci_val = top / bot
        dcis.append(dci_val)
    return dcis


def group_group_dci(pert_mat, groups=None, asym=False):
    """Calculate the inter-group DCI matrix.

    Parameters
    ----------
    pert_mat : np.ndarray
        The perturbation matrix.
    groups : list of lists, optional
        A list of groups, each containing indices of atoms.
        Defaults to a list containing an empty list.
    asym : bool, optional
        If True, compute the asymmetric DCI (default is False).

    Returns
    -------
    list of lists
        A 2D list containing the DCI values between each pair of groups.
    """
    if groups is None:
        groups = [[]]
    dcis = []
    dci_tot = pert_mat / np.sum(pert_mat, axis=-1, keepdims=True)
    if asym:
        dci_tot = dci_tot - dci_tot.T
    for ids1 in groups:
        temp = []
        for ids2 in groups:
            idx1, idx2 = np.meshgrid(ids1, ids2, indexing="ij")
            top = np.sum(dci_tot[idx1, idx2]) * pert_mat.shape[0]
            bot = len(ids1) * len(ids2)
            dci_val = top / bot
            temp.append(dci_val)
        dcis.append(temp)
    return dcis


##############################################################
## Elastic Network Model (ENM)
##############################################################

def hessian(vecs, cutoff, spring_constant=1e3, dd=0):
    """Compute the Hessian matrix using an elastic network model.

    Parameters
    ----------
    vecs : np.ndarray
        Coordinate matrix of shape (n, 3) where each row corresponds to a residue.
    cutoff : float
        Distance cutoff threshold.
    spring_constant : float
        Base spring constant.
    dd : int
        Exponent modifier for the inverse distance.

    Returns
    -------
    np.ndarray
        The computed Hessian matrix.
    """
    # pylint: disable=c-extension-no-member
    return rcmath.hessian(vecs, cutoff, spring_constant, dd)


def inverse_matrix(matrix, device="cpu_sparse", k_singular=6, n_modes=100, dtype=None, **kwargs):
    """Unified wrapper for computing the inverse of a matrix via eigen-decomposition.

    Depending on the 'device' parameter, the function selects an appropriate routine.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix.
    device : str, optional
        Inversion method ('cpu_sparse', 'cpu_dense', 'gpu_sparse', or 'gpu_dense').
        Default is 'cpu_sparse'.
    k_singular : int, optional
        Number of smallest eigenvalues to set to zero (default is 6).
    n_modes : int, optional
        Number of eigenmodes to compute/consider.
    dtype : data-type, optional
        Desired data type for computations (default: matrix.dtype).
    **kwargs :
        Additional keyword arguments for the eigensolver.

    Returns
    -------
    np.ndarray or cp.ndarray
        The computed inverse matrix.
    """
    if dtype is None:
        dtype = matrix.dtype

    if device.lower().startswith("gpu"):
        try:
            if not cp.cuda.is_available():
                raise RuntimeError("CUDA not available.")
            if device.lower() == "gpu_sparse":
                if device.lower() == "gpu_sparse":
                    return rpymath.inverse_sparse_matrix_gpu(
                        matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs
                    )
            if device.lower() == "gpu_dense":
                return rpymath.inverse_matrix_gpu(
                    matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs
                )
            logger.info("Unknown GPU method; falling back to CPU sparse inversion.")
            return rpymath.inverse_sparse_matrix_cpu(
                matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.info("GPU inversion failed with error '%s'. Falling back to CPU sparse inversion.", e)
            return rpymath.inverse_sparse_matrix_cpu(
                matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs
            )
    if device.lower() == "cpu_dense":
        return rpymath.inverse_matrix_cpu(
            matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs
        )
    return rpymath.inverse_sparse_matrix_cpu(
        matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs
    )


##############################################################
## Miscellaneous Functions
##############################################################

def percentile(x):
    """Compute the percentile ranking for each element in an array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        An array containing the percentile (from 0 to 1) of each element in x.
    """
    sorted_idx = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_idx == n)[0][0] / len(x)
    return px


if __name__ == "__main__":
    pass
