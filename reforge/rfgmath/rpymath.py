"""Python math functions

Description:
    This module contains internal routines for performing various mathematical
    and signal processing operations required in our workflow. It includes FFT‐based
    correlation computations (serial, parallel, and GPU versions), covariance matrix
    calculation, dynamic coupling and flexibility index evaluations, sparse matrix
    inversion on both CPU and GPU, and additional helper functions such as percentile
    computation and FFT‐based convolution.
    
    Note: This module is intended for internal use only.
    
Usage Example:
    >>> from rpymath import _sfft_ccf, fft_ccf
    >>> import numpy as np
    >>> # Generate random signals
    >>> x = np.random.rand(10, 256)
    >>> y = np.random.rand(10, 256)
    >>> # Compute serial FFT‐based correlation
    >>> corr = _sfft_ccf(x, y, ntmax=64, center=True, loop=True)
    >>> # Or use the unified FFT correlation wrapper
    >>> corr = fft_ccf(x, y, mode='serial', ntmax=64, center=True)
    
Requirements:
    - Python 3.x
    - NumPy
    - SciPy
    - CuPy (for GPU-based functions)
    - joblib (for parallel processing)
    - MDAnalysis (if required elsewhere)
    
Author: DY
Date: YYYY-MM-DD
"""

import numpy as np
import cupy as cp
import scipy.sparse.linalg
import cupyx.scipy.sparse.linalg
from joblib import Parallel, delayed
from numpy.fft import fft, ifft
from reforge.utils import timeit, memprofit, logger

##############################################################
## For time dependent analysis ##
##############################################################

@memprofit
@timeit
def sfft_ccf(x, y, ntmax=None, center=False, loop=True, dtype=None):
    """Compute the correlation function between two signals using a serial FFT-based method.
    
    Parameters:
        x (np.ndarray): First input signal of shape (n_coords, n_samples).
        y (np.ndarray): Second input signal of shape (n_coords, n_samples).
        ntmax (int, optional): Maximum number of time samples to retain.
        center (bool, optional): If True, subtract the mean from each signal.
        loop (bool, optional): If True, compute the correlation in a loop.
        dtype (data-type, optional): Desired data type (default: x.dtype).
    
    Returns:
        np.ndarray: Correlation function array of shape (n_coords, n_coords, ntmax).
    """
    logger.info("Computing CCFs serially.")
    if dtype is None:
        dtype = x.dtype
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if ntmax is None or ntmax > (nt + 1) // 2:
        ntmax = (nt + 1) // 2
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT with zero-padding
    x_f = fft(x, n=2 * nt, axis=-1)
    y_f = fft(y, n=2 * nt, axis=-1)
    counts = np.arange(nt, nt - ntmax, -1).astype(dtype) ** -1

    # Define a local helper that only takes i and j by capturing outer variables.
    def compute_correlation(i, j):
        corr = ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real
        return corr * counts

    if loop:
        corr = np.zeros((nx, ny, ntmax), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                corr[i, j] = compute_correlation(i, j)
    else:
        corr = np.einsum("it,jt->ijt", x_f, np.conj(y_f))
        corr = ifft(corr, axis=-1).real / nt
    return corr


@memprofit
@timeit
def pfft_ccf(x, y, ntmax=None, center=False, dtype=None):
    """Compute the correlation function using a parallel FFT-based method.
    
    Parameters:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        ntmax (int, optional): Maximum number of time samples to retain.
        center (bool, optional): If True, subtract the mean.
        dtype (data-type, optional): Desired data type.
    
    Returns:
        np.ndarray: Correlation function array with shape (n_coords, n_coords, ntmax).
    """
    logger.info("Computing CCFs in parallel.")
    if dtype is None:
        dtype = x.dtype
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if ntmax is None or ntmax > (nt + 1) // 2:
        ntmax = (nt + 1) // 2
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x_f = fft(x, n=2 * nt, axis=-1)
    y_f = fft(y, n=2 * nt, axis=-1)
    counts = np.arange(nt, nt - ntmax, -1).astype(dtype) ** -1

    def compute_correlation(i, j):
        corr = ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real
        return corr * counts

    def parallel_fft_correlation():
        results = Parallel(n_jobs=-1)(
            delayed(compute_correlation)(i, j)
            for i in range(nx)
            for j in range(ny)
        )
        return np.array(results).reshape(nx, ny, ntmax)

    corr = parallel_fft_correlation()
    return corr


@memprofit
@timeit
def gfft_ccf(x, y, ntmax=None, center=True, dtype=None):
    """Compute the correlation function on the GPU using FFT.
    
    Parameters:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        ntmax (int, optional): Maximum number of time samples to retain.
        center (bool, optional): If True, subtract the mean.
        dtype (data-type, optional): Desired CuPy data type (default: inferred from x).
    
    Returns:
        cp.ndarray: The computed correlation function as a CuPy array.
    """
    logger.info("Computing CCFs on GPU.")
    if dtype is None:
        dtype = x.dtype
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if ntmax is None or ntmax > (nt + 1) // 2:
        ntmax = (nt + 1) // 2
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    x_f = cp.fft.fft(x, n=2 * nt, axis=-1)
    y_f = cp.fft.fft(y, n=2 * nt, axis=-1)
    counts = cp.arange(nt, nt - ntmax, -1, dtype=dtype) ** -1
    counts = counts[None, :]
    corr = cp.zeros((nx, ny, ntmax), dtype=dtype)
    logger.debug(f"{nx}, {ny}, {nt}, {ntmax}, {counts.shape}")
    for i in range(nx):
        corr[i] = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] * counts
    return corr


@memprofit
@timeit
def gfft_ccf_auto(x, y, ntmax=None, center=True, buffer_c=0.94, dtype=None):
    """Same as "gfft_ccf" but regulates GPU to CPU I/O based on the available GPU memory"""
    logger.info("Computing CCFs on GPU.")
    if dtype is None:
        dtype = x.dtype
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if ntmax is None or ntmax > (nt + 1) // 2:
        ntmax = (nt + 1) // 2
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    x_f = cp.fft.fft(x, n=2*nt, axis=-1)
    y_f = cp.fft.fft(y, n=2*nt, axis=-1)
    # Need to calculate how do distribute the memory
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo() 
    req_bytes = (x.shape[0] * y.shape[0] * ntmax + 8 * y.shape[0] * nt ) * np.dtype(dtype).itemsize
    mem_free = free_bytes >> 20
    mem_total = total_bytes >> 20
    mem_req = req_bytes >> 20
    logger.debug("Free: %s Mb, Total: %s Mb, required~ %s Mb", mem_free, mem_total, mem_req)
    if mem_req != 0:
        nxmax = int(buffer_c * mem_total / mem_req * nx)
    else:
        nxmax = nx
    if nxmax > nx:
        nxmax = nx
    logger.debug("nxmax %s", nxmax)
    n_sweeps = nx // nxmax
    n_remain = nx % nxmax
    logger.debug("Sweeps needed %s, remainder: %s", n_sweeps, n_remain)
    corr = np.empty(nx * ny  * ntmax, dtype=dtype)
    arr_gpu = cp.empty(nxmax * ny * ntmax, dtype=dtype) # Flat is way faster
    counts = cp.arange(nt, nt - ntmax, -1, dtype=dtype) ** -1
    counts = counts[None, :]
    # Calculating CCF for full sweeps:
    for sw in range(n_sweeps):
        logger.debug("Sweep number %s", sw)
        for i in range(nxmax):
            temp_res = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] * counts # shape (ny, ntmax)
            offset = i * (ny * ntmax) # Compute the offset into the flat array for this block
            arr_gpu[offset: offset + (ny * ntmax)] = temp_res.reshape(-1) # result.reshape(-1) gives a 1D view
        mem_pool = cp.get_default_memory_pool()
        used_bytes = mem_pool.used_bytes() >> 20
        logger.debug("Used: %s Mb", used_bytes)
        corr[ntmax * ny * nxmax * sw : ntmax * ny * nxmax * (sw + 1)] = arr_gpu.get() # Transfer the flat array 
    for i in range(n_remain): # Handling remaining elements:
        result = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] * counts
        offset = i * (ny * ntmax)
        arr_gpu[offset: offset + (ny * ntmax)] = result.reshape(-1)
    if n_remain > 0: # Transfer only the needed portion for the remaining data:
        corr[nxmax * ny * ntmax * n_sweeps: ] = arr_gpu.get()[: n_remain * ny * ntmax]
    return corr.reshape(nx, ny, ntmax)


def fft_ccf(*args, mode="serial", **kwargs):
    """Unified wrapper for FFT-based correlation functions.
    
    Parameters:
        args: Positional arguments for the chosen correlation function.
        mode (str): Mode to use ('serial', 'parallel', or 'gpu').
        kwargs: Additional keyword arguments.
    
    Returns:
        np.ndarray: The computed correlation function.
    
    Raises:
        ValueError: If an unsupported mode is specified.
    """
    if mode == "serial":
        return sfft_ccf(*args, **kwargs)
    if mode == "parallel":
        return pfft_ccf(*args, **kwargs)
    if mode == "gpu":
        result = gfft_ccf_auto(*args, **kwargs)
        return result
    raise ValueError("Currently 'mode' should be 'serial', 'parallel' or 'gpu'.")


@memprofit
@timeit
def ccf(xs, ys, ntmax=None, n=1, mode="parallel", center=True, dtype=None):
    """Compute the average cross-correlation function of two signals by segmenting them.
    
    Parameters:
        xs (np.ndarray): First input signal of shape (n_coords, n_samples).
        ys (np.ndarray): Second input signal of shape (n_coords, n_samples).
        ntmax (int, optional): Maximum number of time samples per segment.
        n (int, optional): Number of segments.
        mode (str, optional): Mode ('parallel', 'serial', or 'gpu').
        center (bool, optional): If True, mean-center the signals.
        dtype (data-type, optional): Desired data type.
    
    Returns:
        np.ndarray: The averaged cross-correlation function.
    """
    logger.info("Calculating cross-correlation.")
    if dtype is None:
        dtype = xs.dtype
    xs_segments = np.array_split(xs, n, axis=-1)
    ys_segments = np.array_split(ys, n, axis=-1)
    nx = xs_segments[0].shape[0]
    ny = ys_segments[0].shape[0]
    nt = xs_segments[-1].shape[1]
    logger.info("Splitting trajectory into %d parts", n)
    if ntmax is None or ntmax > (nt + 1) // 2:
        ntmax = (nt + 1) // 2
    corr = np.zeros((nx, ny, ntmax), dtype=dtype)
    counter = 1
    for seg_x, seg_y in zip(xs_segments, ys_segments):
        logger.info("Processing part %s", counter)
        corr_seg = fft_ccf(seg_x, seg_y, ntmax=ntmax, mode=mode, center=center, dtype=dtype)
        logger.debug("Segment correlation shape: %s", corr_seg.shape)
        corr += corr_seg
        counter += 1
    corr /= n
    logger.debug("RMS of correlation: %.6f", np.sqrt(np.average(corr**2)))
    logger.info("Finished calculating cross-correlation.")
    return corr


@memprofit
@timeit
def gfft_conv(x, y, loop=False, dtype=None):
    """Compute element-wise convolution between two signals on the GPU using FFT.
    
    Parameters:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        loop (bool, optional): If True, use a loop-based computation.
        dtype (data-type, optional): Desired CuPy data type.
    
    Returns:
        np.ndarray: Convolution result as a NumPy array.
    """
    logger.info("Doing convolution on GPU.")
    if dtype is None:
        dtype = x.dtype
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = x.shape[1]
    x_gpu = cp.asarray(x, dtype=dtype)
    y_gpu = cp.asarray(y, dtype=dtype)
    x_f = cp.fft.fft(x_gpu, n=2 * nt, axis=-1)
    y_f = cp.fft.fft(y_gpu, n=2 * nt, axis=-1)
    counts = cp.arange(nt, 0, -1, dtype=dtype) ** -1
    if loop:
        conv = np.zeros((nx, ny, nt), dtype=dtype)
        counts = counts[None, :]
        for i in range(nx):
            conv[i] = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :nt] * counts
            conv[i] = conv[i].get()
    else:
        counts = counts[None, None, :]
        conv = cp.fft.ifft(x_f * cp.conj(y_f), axis=-1).real[:, :, :nt] * counts
        conv = conv.get()
    return conv


@memprofit
@timeit
def sfft_cpsd(x, y, ntmax=None, center=True, loop=True, dtype=np.float64):
    """Compute the Cross-Power Spectral Density (CPSD) between two signals using FFT.
    
    Parameters:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        ntmax (int, optional): Number of frequency bins to retain.
        center (bool, optional): If True, mean-center the signals.
        loop (bool, optional): If True, use loop-based computation.
        dtype (data-type, optional): Desired data type.
    
    Returns:
        np.ndarray: The computed CPSD.
    """
    def compute_cpsd(i, j):
        cpsd_ij = x_f[i] * np.conj(y_f[j])
        cpsd_ij = np.abs(cpsd_ij) / nt
        return np.average(cpsd_ij)

    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if ntmax is None:
        ntmax = nt
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x_f = fft(x, axis=-1)
    y_f = fft(y, axis=-1)
    if loop:
        cpsd = np.zeros((nx, ny), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                cpsd[i, j] = compute_cpsd(i, j)
    else:
        cpsd = np.einsum("it,jt->ijt", x_f, np.conj(y_f))[:, :, :ntmax]
    cpsd = np.abs(cpsd)
    return cpsd


@memprofit
@timeit
def covariance_matrix(positions, dtype=np.float32):
    """Calculate the position-position covariance matrix from a set of positions.
    
    The function centers the input positions by subtracting their mean and then
    computes the covariance matrix using np.cov.
    
    Parameters:
        positions (np.ndarray): Array of positions.
        dtype (data-type, optional): Data type (default: np.float32).
    
    Returns:
        np.ndarray: The computed covariance matrix.
    """
    mean = positions.mean(axis=-1, keepdims=True)
    centered_positions = positions - mean
    covmat = np.cov(centered_positions, rowvar=True, dtype=dtype)
    return np.array(covmat)


##############################################################
## DCI and DFI Calculations ##
##############################################################

def dci(perturbation_matrix, asym=False):
    """Calculate the Dynamic Coupling Index (DCI) matrix from a perturbation matrix.
    
    Parameters:
        perturbation_matrix (np.ndarray): Input perturbation matrix.
        asym (bool, optional): If True, return asymmetric DCI.
    
    Returns:
        np.ndarray: The computed DCI matrix.
    """
    dci_val = (perturbation_matrix * perturbation_matrix.shape[0] /
               np.sum(perturbation_matrix, axis=-1, keepdims=True))
    if asym:
        dci_val = dci_val - dci_val.T
    return dci_val


def group_molecule_dci(perturbation_matrix, groups=None, asym=False):
    """Compute the DCI for specified groups of atoms relative to the entire molecule.
    
    Parameters:
        perturbation_matrix (np.ndarray): The perturbation matrix.
        groups (list of list, optional): List of groups of atom indices. Defaults to [[]].
        asym (bool, optional): If True, adjust for asymmetry.
    
    Returns:
        list: List of DCI values for each group.
    """
    if groups is None:
        groups = [[]]
    dcis = []
    dci_tot = (perturbation_matrix / 
               np.sum(perturbation_matrix, axis=-1, keepdims=True))
    if asym:
        dci_tot = dci_tot - dci_tot.T
    for ids in groups:
        top = np.sum(dci_tot[:, ids], axis=-1) * perturbation_matrix.shape[0]
        bot = len(ids)
        dci_val = top / bot
        dcis.append(dci_val)
    return dcis


def group_group_dci(perturbation_matrix, groups=None, asym=False):
    """Calculate the DCI matrix between different groups of atoms.
    
    Parameters:
        perturbation_matrix (np.ndarray): The perturbation matrix.
        groups (list of list, optional): List of groups (each a list of indices). Defaults to [[]].
        asym (bool, optional): If True, compute asymmetric DCI.
    
    Returns:
        list: A nested list representing the DCI matrix between groups.
    """
    if groups is None:
        groups = [[]]
    dcis = []
    dci_tot = (perturbation_matrix / 
               np.sum(perturbation_matrix, axis=-1, keepdims=True))
    if asym:
        dci_tot = dci_tot - dci_tot.T
    for ids1 in groups:
        temp = []
        for ids2 in groups:
            idx1, idx2 = np.meshgrid(ids1, ids2, indexing="ij")
            top = np.sum(dci_tot[idx1, idx2]) * perturbation_matrix.shape[0]
            bot = len(ids1) * len(ids2)
            dci_val = top / bot
            temp.append(dci_val)
        dcis.append(temp)
    return dcis


##############################################################
## Elastic Network Model (ENM) Functions ##
##############################################################

@timeit
@memprofit
def inverse_sparse_matrix_cpu(matrix, k_singular=6, n_modes=20, dtype=None, **kwargs):
    """Compute the inverse of a sparse matrix on the CPU using eigen-decomposition.
    
    Parameters:
        matrix (np.ndarray): Input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to compute.
        dtype: Desired data type (default: matrix.dtype).
        kwargs: Additional arguments for eigensolver.
    
    Returns:
        np.ndarray: The computed inverse matrix.
    """
    kwargs.setdefault("k", n_modes)
    kwargs.setdefault("which", "SA")
    kwargs.setdefault("tol", 0)
    kwargs.setdefault("maxiter", None)
    if dtype is None:
        dtype = matrix.dtype
    matrix = np.asarray(matrix, dtype=dtype)
    evals, evecs = scipy.sparse.linalg.eigsh(matrix, **kwargs)
    inv_evals = evals**-1
    inv_evals[:k_singular] = 0.0
    logger.info("%s", evals[:20])
    inv_matrix = np.matmul(evecs, np.matmul(np.diag(inv_evals), evecs.T))
    return inv_matrix


@timeit
@memprofit
def inverse_matrix_cpu(matrix, k_singular=6, n_modes=100, dtype=None, **kwargs):
    """Compute the inverse of a matrix on the CPU using dense eigen-decomposition.
    
    Parameters:
        matrix (np.ndarray): Input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to consider.
        dtype: Desired data type (default: matrix.dtype).
        kwargs: Additional arguments for the solver.
    
    Returns:
        np.ndarray: The inverse matrix computed on the CPU.
    """
    if dtype is None:
        dtype = matrix.dtype
    matrix = np.asarray(matrix, dtype=dtype)
    evals, evecs = np.linalg.eigh(matrix, **kwargs)
    evals = evals[:n_modes]
    evecs = evecs[:, :n_modes]
    inv_evals = evals**-1
    inv_evals[:k_singular] = 0.0
    logger.info("%s", evals[:20])
    inv_matrix = np.matmul(evecs, np.matmul(np.diag(inv_evals), evecs.T))
    return inv_matrix


@timeit
@memprofit
def inverse_sparse_matrix_gpu(matrix, k_singular=6, n_modes=20, dtype=None, **kwargs):
    """Compute the inverse of a sparse matrix on the GPU using eigen-decomposition.
    
    Parameters:
        matrix (np.ndarray): Input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to compute.
        dtype: Desired CuPy data type (default: matrix.dtype).
        kwargs: Additional arguments for the GPU eigensolver.
    
    Returns:
        cp.ndarray: The inverse matrix computed on the GPU.
    """
    kwargs.setdefault("k", n_modes)
    kwargs.setdefault("which", "SA")
    kwargs.setdefault("tol", 0)
    kwargs.setdefault("maxiter", None)
    if dtype is None:
        dtype = matrix.dtype
    matrix_gpu = cp.asarray(matrix, dtype)
    evals_gpu, evecs_gpu = cupyx.scipy.sparse.linalg.eigsh(matrix_gpu, **kwargs)
    inv_evals_gpu = evals_gpu**-1
    inv_evals_gpu[:k_singular] = 0.0
    logger.info("%s", evals_gpu[:20])
    inv_matrix_gpu = cp.matmul(
        evecs_gpu, cp.matmul(cp.diag(inv_evals_gpu), evecs_gpu.T)
    )
    return inv_matrix_gpu


@timeit
@memprofit
def inverse_matrix_gpu(matrix, k_singular=6, n_modes=100, dtype=None, **kwargs):
    """Compute the inverse of a matrix on the GPU using dense eigen-decomposition.
    
    Parameters:
        matrix (np.ndarray): Input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to consider.
        dtype: Desired CuPy data type (default: matrix.dtype).
        kwargs: Additional arguments for the solver.
    
    Returns:
        cp.ndarray: The inverse matrix computed on the GPU.
    """
    if dtype is None:
        dtype = matrix.dtype
    matrix_gpu = cp.asarray(matrix, dtype)
    evals_gpu, evecs_gpu = cp.linalg.eigh(matrix_gpu, **kwargs)
    evals_gpu = evals_gpu[:n_modes]
    evecs_gpu = evecs_gpu[:, :n_modes]
    inv_evals_gpu = evals_gpu**-1
    inv_evals_gpu[:k_singular] = 0.0
    logger.info("%s", evals_gpu[:20])
    inv_matrix_gpu = cp.matmul(
        evecs_gpu, cp.matmul(cp.diag(inv_evals_gpu), evecs_gpu.T)
    )
    return inv_matrix_gpu


##############################################################
## Miscellaneous Functions ##
##############################################################

def percentile(x):
    """Compute the empirical percentile rank for each element in an array.
    
    Parameters:
        x (np.ndarray): Input array.
    
    Returns:
        np.ndarray: Array of percentile ranks.
    """
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px


if __name__ == "__main__":
    pass

