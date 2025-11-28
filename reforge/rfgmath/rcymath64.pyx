"""Cython Math

Description:
    This module contains internal routines for performing optimized mathematical 
    operations. It includes functions for calculating position-position Hessian matrices 
    and perturbation matrices derived from coordinate and covariance data. 
    The computations are accelerated using Cython.

    Note: This module is intended for internal use only within the reForge workflow.

Usage Example:
    >>> import numpy as np
    >>> from rcmath import _calculate_hessian, _hessian, _perturbation_matrix, _td_perturbation_matrix
    >>> # Generate random coordinate data for residues
    >>> n = 10
    >>> vecs = np.random.rand(n, 3)
    >>> hessian2 = _hessian(vecs, cutoff=1.2, spring_constant=1000, dd=0)
    >>> # Compute perturbation matrix from a covariance matrix
    >>> cov_matrix = np.random.rand(3 * n, 3 * n)
    >>> pert_matrix = _perturbation_matrix(cov_matrix)
    >>> # Compute block-wise perturbation matrix with normalization
    >>> td_pert_matrix = _td_perturbation_matrix(cov_matrix, normalize=True)

Requirements:
    - Python 3.x
    - NumPy
    - Cython
    - reForge utilities (timeit, memprofit)

Author: Your Name
Date: YYYY-MM-DD
"""


import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, threadid, parallel
from libc.math cimport sqrt, pow
from reforge.utils import timeit, memprofit


cdef extern from "omp.h":
    int omp_get_max_threads()
    int omp_get_num_threads()


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def calculate_hessian(int resnum,
                       np.ndarray[double, ndim=1] x,
                       np.ndarray[double, ndim=1] y,
                       np.ndarray[double, ndim=1] z,
                       double cutoff=1.2,
                       double spring_constant=1000,
                       int dd=0):
    """
    Calculate the position-position Hessian matrix based on individual coordinate arrays.

    Parameters
    ----------
    resnum : int
        Number of residues (atoms).
    x : ndarray of double, 1D
        Array of x coordinates.
    y : ndarray of double, 1D
        Array of y coordinates.
    z : ndarray of double, 1D
        Array of z coordinates.
    cutoff : double, optional
        Distance cutoff threshold. Interactions beyond this distance are ignored (default is 1.2).
    spring_constant : double, optional
        Base spring constant used in the calculation (default is 1000).
    dd : int, optional
        Exponent modifier applied to the inverse distance factor (default is 0).

    Returns
    -------
    hessian : ndarray of double, 2D
        Hessian matrix of shape (3*resnum, 3*resnum) representing the second derivatives 
        of the system energy with respect to positions.
    """
    cdef int i, j
    cdef double x_ij, y_ij, z_ij, r, invr, gamma
    cdef np.ndarray[double, ndim=2] hessian = np.zeros((3 * resnum, 3 * resnum), dtype=np.float64)
    
    for i in range(resnum):
        for j in range(resnum):
            if j == i:
                continue
            x_ij = x[i] - x[j]
            y_ij = y[i] - y[j]
            z_ij = z[i] - z[j]
            r = sqrt(x_ij*x_ij + y_ij*y_ij + z_ij*z_ij)
            if r < cutoff:
                invr = 1.0 / r
                gamma = spring_constant * pow(invr, 2 + dd)
            else:
                continue
            # Update diagonal elements (Hii)
            hessian[3 * i, 3 * i]         += gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * i + 1]   += gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * i + 2]   += gamma * z_ij * z_ij
            hessian[3 * i, 3 * i + 1]       += gamma * x_ij * y_ij
            hessian[3 * i, 3 * i + 2]       += gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * i]       += gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * i + 2]   += gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * i]       += gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * i + 1]   += gamma * y_ij * z_ij
            # Update off-diagonal elements (Hij)
            hessian[3 * i, 3 * j]         -= gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * j + 1]   -= gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * j + 2]   -= gamma * z_ij * z_ij
            hessian[3 * i, 3 * j + 1]       -= gamma * x_ij * y_ij
            hessian[3 * i, 3 * j + 2]       -= gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * j]       -= gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * j + 2]   -= gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * j]       -= gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * j + 1]   -= gamma * y_ij * z_ij
    return hessian


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def hessian(np.ndarray[double, ndim=2] vec,
             double cutoff=1.2,
             double spring_constant=1000,
             int dd=0):
    """
    Calculate the position-position Hessian matrix from a coordinate matrix.

    Parameters
    ----------
    vec : ndarray of double, 2D
        A coordinate matrix where each residue's coordinates are provided.
        Note: The function computes n = vec.shape[0] // 3; thus, vec.shape[0]
        should be a multiple of 3. Each residue is expected to have three entries
        corresponding to its x, y, and z coordinates.
    cutoff : double, optional
        Distance cutoff threshold. Interactions beyond this distance are ignored (default is 1.2).
    spring_constant : double, optional
        Base spring constant used in the calculation (default is 1000).
    dd : int, optional
        Exponent modifier applied to the inverse distance factor (default is 0).

    Returns
    -------
    hessian : ndarray of double, 2D
        Hessian matrix of shape (3*n, 3*n) representing the second derivatives of the system energy,
        where n is derived from the input coordinate matrix.
    """
    cdef int i, j
    cdef double x_ij, y_ij, z_ij, r, invr, gamma
    cdef int n = vec.shape[0] 
    cdef np.ndarray[double, ndim=2] hessian = np.zeros((3 * n, 3 * n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            x_ij = vec[i, 0] - vec[j, 0]
            y_ij = vec[i, 1] - vec[j, 1]
            z_ij = vec[i, 2] - vec[j, 2]
            r = sqrt(x_ij*x_ij + y_ij*y_ij + z_ij*z_ij)
            if r < cutoff:
                invr = 1.0 / r
                gamma = spring_constant * pow(invr, 2 + dd)
            else:
                continue
            # Update diagonal elements (Hii)
            hessian[3 * i, 3 * i]         += gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * i + 1]   += gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * i + 2]   += gamma * z_ij * z_ij
            hessian[3 * i, 3 * i + 1]       += gamma * x_ij * y_ij
            hessian[3 * i, 3 * i + 2]       += gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * i]       += gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * i + 2]   += gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * i]       += gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * i + 1]   += gamma * y_ij * z_ij
            # Update off-diagonal elements (Hij)
            hessian[3 * i, 3 * j]         -= gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * j + 1]   -= gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * j + 2]   -= gamma * z_ij * z_ij
            hessian[3 * i, 3 * j + 1]       -= gamma * x_ij * y_ij
            hessian[3 * i, 3 * j + 2]       -= gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * j]       -= gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * j + 2]   -= gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * j]       -= gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * j + 1]   -= gamma * y_ij * z_ij
    return hessian


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def perturbation_matrix_old(np.ndarray[double, ndim=2] covariance_matrix,
                             int resnum):
    """
    Compute a perturbation matrix from a covariance matrix using an older method.

    Parameters
    ----------
    covariance_matrix : ndarray of double, 2D
        A covariance matrix of shape (3*resnum, 3*resnum) computed from position data.
    resnum : int
        Number of residues (atoms).

    Returns
    -------
    perturbation_matrix : ndarray of double, 2D
        A normalized perturbation matrix of shape (resnum, resnum), where each element 
        represents the cumulative perturbation contribution from directional projections.
    """
    cdef int i, j, k, d, n = resnum
    cdef double norm, sum_val, s
    cdef np.ndarray[double, ndim=2] perturbation_matrix = np.zeros((n, n), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] directions
    cdef double f0, f1, f2
    cdef double delta0, delta1, delta2

    directions = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float64
    )
    for k in range(directions.shape[0]):
        norm = 0.0
        for d in range(3):
            norm += directions[k, d] * directions[k, d]
        norm = sqrt(norm)
        for d in range(3):
            directions[k, d] /= norm

    for k in range(directions.shape[0]):
        f0 = directions[k, 0]
        f1 = directions[k, 1]
        f2 = directions[k, 2]
        for j in range(n):
            for i in range(n):
                delta0 = (covariance_matrix[3*i,   3*j]   * f0 +
                          covariance_matrix[3*i,   3*j+1] * f1 +
                          covariance_matrix[3*i,   3*j+2] * f2)
                delta1 = (covariance_matrix[3*i+1, 3*j]   * f0 +
                          covariance_matrix[3*i+1, 3*j+1] * f1 +
                          covariance_matrix[3*i+1, 3*j+2] * f2)
                delta2 = (covariance_matrix[3*i+2, 3*j]   * f0 + 
                          covariance_matrix[3*i+2, 3*j+1] * f1 +
                          covariance_matrix[3*i+2, 3*j+2] * f2)
                s = sqrt(delta0*delta0 + delta1*delta1 + delta2*delta2)
                perturbation_matrix[i, j] += s

    sum_val = 0.0
    for i in range(n):
        for j in range(n):
            sum_val += perturbation_matrix[i, j]
    if sum_val != 0:
        for i in range(n):
            for j in range(n):
                perturbation_matrix[i, j] /= sum_val

    return perturbation_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def perturbation_matrix(np.ndarray[double, ndim=2] covariance_matrix, bint normalize=True) -> np.ndarray:
    """
    Compute a perturbation matrix from a covariance matrix. Same as the old one but no need 
    to specify resnum and works with rectangular matrices

    Parameters
    ----------
    covariance_matrix : ndarray of double, 2D
        A covariance matrix computed from positional data with shape (3*m, 3*n), where m and n
        are the numbers of residues in two systems.

    Returns
    -------
    perturbation_matrix : ndarray of double, 2D
        A normalized perturbation matrix of shape (m, n). Each element represents the aggregated
        perturbation computed from the directional components of the corresponding 3x3 block.
    """
    cdef int i, j, k, d
    cdef int m = covariance_matrix.shape[0] // 3
    cdef int n = covariance_matrix.shape[1] // 3
    cdef double norm, sum_val
    cdef np.ndarray[double, ndim=2] perturbation_matrix = np.zeros((m, n), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] directions
    cdef double f0, f1, f2
    cdef double delta0, delta1, delta2

    # Create and normalize an array of 7 directional vectors.
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                          dtype=np.float64)
    for k in range(directions.shape[0]):
        norm = 0.0
        for d in range(3):
            norm += directions[k, d] * directions[k, d]
        norm = sqrt(norm)
        for d in range(3):
            directions[k, d] /= norm

    for k in range(directions.shape[0]):
        f0 = directions[k, 0]
        f1 = directions[k, 1]
        f2 = directions[k, 2]
        for j in range(n):
            for i in range(m):
                delta0 = (covariance_matrix[3*i,   3*j]   * f0 +
                          covariance_matrix[3*i,   3*j+1] * f1 +
                          covariance_matrix[3*i,   3*j+2] * f2)
                delta1 = (covariance_matrix[3*i+1, 3*j]   * f0 +
                          covariance_matrix[3*i+1, 3*j+1] * f1 +
                          covariance_matrix[3*i+1, 3*j+2] * f2)
                delta2 = (covariance_matrix[3*i+2, 3*j]   * f0 +
                          covariance_matrix[3*i+2, 3*j+1] * f1 +
                          covariance_matrix[3*i+2, 3*j+2] * f2)
                perturbation_matrix[i, j] += sqrt(delta0*delta0 + delta1*delta1 + delta2*delta2)


    if normalize:
        sum_val = np.sum(perturbation_matrix)
        perturbation_matrix *= n * m / sum_val

    return perturbation_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def perturbation_matrix_par(np.ndarray[double, ndim=2] covariance_matrix, bint normalize=True) -> np.ndarray:
    """
    Compute a perturbation matrix from a covariance matrix. Same as the old one but no need 
    to specify resnum and works with rectangular matrices

    Parameters
    ----------
    covariance_matrix : ndarray of double, 2D
        A covariance matrix computed from positional data with shape (3*m, 3*n), where m and n
        are the numbers of residues in two systems.

    Returns
    -------
    perturbation_matrix : ndarray of double, 2D
        A normalized perturbation matrix of shape (m, n). Each element represents the aggregated
        perturbation computed from the directional components of the corresponding 3x3 block.
    """
    cdef int m = covariance_matrix.shape[0] // 3
    cdef int n = covariance_matrix.shape[1] // 3
    cdef int k, i, j, tid
    cdef double sum_val, norm, f0, f1, f2, delta0, delta1, delta2
    # Allocate the final output array.
    cdef np.ndarray[np.double_t, ndim=2] perturbation_matrix = np.zeros((m, n), dtype=np.float64)
    # Allocate a thread-local accumulation array.
    cdef int num_threads = omp_get_max_threads()
    print(num_threads)
    cdef np.ndarray[np.double_t, ndim=3] local_acc = np.zeros((num_threads, m, n), dtype=np.float64)
    # Create typed memoryviews for fast access.
    cdef double[:, :] cov_view = covariance_matrix
    cdef double[:, :] pert_view = perturbation_matrix
    cdef double[:, :, :] local_acc_view = local_acc

    # Create and normalize an array of 7 directional vectors.
    cdef np.ndarray[double, ndim=2] directions
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                          dtype=np.float64)
    for k in range(directions.shape[0]):
        norm = 0.0
        for d in range(3):
            norm += directions[k, d] * directions[k, d]
        norm = sqrt(norm)
        for d in range(3):
            directions[k, d] /= norm

    # Parallel loop over directions.
    cdef int num_directions = directions.shape[0]
    cdef double[:, :] directions_view = directions
    for k in prange(num_directions, nogil=True, schedule='static'):
        f0 = directions_view[k, 0]
        f1 = directions_view[k, 1]
        f2 = directions_view[k, 2]
        tid = threadid() # Get thread id to index into the local accumulator.
        for j in range(n):
            for i in range(m):
                delta0 = (cov_view[3*i,   3*j]   * f0 +
                          cov_view[3*i,   3*j+1] * f1 +
                          cov_view[3*i,   3*j+2] * f2)
                delta1 = (cov_view[3*i+1, 3*j]   * f0 +
                          cov_view[3*i+1, 3*j+1] * f1 +
                          cov_view[3*i+1, 3*j+2] * f2)
                delta2 = (cov_view[3*i+2, 3*j]   * f0 +
                          cov_view[3*i+2, 3*j+1] * f1 +
                          cov_view[3*i+2, 3*j+2] * f2)
                local_acc_view[tid, i, j] += sqrt(delta0*delta0 + delta1*delta1 + delta2*delta2)
    
    perturbation_matrix = np.sum(local_acc, axis=0)
    if normalize:
        sum_val = np.sum(perturbation_matrix)
        perturbation_matrix *= n * m / sum_val

    return perturbation_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def perturbation_matrix_iso(np.ndarray[double, ndim=2] ccf, bint normalize=True) -> np.ndarray:
    """
    Calculate the perturbation matrix from a covariance (or Hessian) matrix using block-wise norms.

    The input covariance matrix 'ccf' is expected to have shape (3*m, 3*n). For each block (i,j),
    the perturbation value is computed as the square root of the sum of the squares of the corresponding
    3x3 block elements.

    Parameters
    ----------
    ccf : ndarray of double, 2D
        Input covariance matrix with shape (3*m, 3*n).
    normalize : bool, optional
        If True, the output perturbation matrix is normalized so that the total sum of its elements equals 1
        (default is True).

    Returns
    -------
    perturbation_matrix : ndarray of double, 2D
        An (m, n) matrix of perturbation values computed from the blocks of ccf.
    """
    cdef int m = ccf.shape[0] // 3
    cdef int n = ccf.shape[1] // 3
    cdef int i, j, a, b
    cdef double temp, sum_val = 0.0
    cdef np.ndarray[double, ndim=2] perturbation_matrix = np.empty((m, n), dtype=np.float64)
    
    # Compute the block-wise norm for each (i,j) block.
    for i in range(m):
        for j in range(n):
            temp = 0.0
            for a in range(3):
                for b in range(3):
                    temp += ccf[3*i + a, 3*j + b] * ccf[3*i + a, 3*j + b]
            perturbation_matrix[i, j] = sqrt(temp)
            sum_val += perturbation_matrix[i, j]
    
    if normalize and sum_val != 0.0:
        norm = n * m / sum_val  
        for i in range(m):
            for j in range(n):
                perturbation_matrix[i, j] *= norm
                
    return perturbation_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def perturbation_matrix_iso_par(np.ndarray[double, ndim=2] ccf, bint normalize=True) -> np.ndarray:
    """
    Parallel computation of perturbation matrix using OpenMP. 
    Calculate the perturbation matrix from a covariance (or Hessian) matrix using block-wise norms.

    The input covariance matrix 'ccf' is expected to have shape (3*m, 3*n). For each block (i,j),
    the perturbation value is computed as the square root of the sum of the squares of the corresponding
    3x3 block elements.

    Parameters
    ----------
    ccf : ndarray of double, 2D
        Input covariance matrix with shape (3*m, 3*n).
    normalize : bool, optional
        If True, the output perturbation matrix is normalized so that the total sum of its elements equals 1
        (default is True).

    Returns
    -------
    perturbation_matrix : ndarray of double, 2D
        An (m, n) matrix of perturbation values computed from the blocks of ccf.
    """
    cdef int m = ccf.shape[0] // 3
    cdef int n = ccf.shape[1] // 3
    cdef int i, j, a, b
    cdef double sum_val = 0.0
    cdef np.ndarray[double, ndim=2] perturbation_matrix = np.zeros((m, n), dtype=np.float64)
    # Compute the block-wise norm for each (i,j) block.
    for i in prange(m, nogil=True, schedule='static'):
        for j in range(n):
            for a in range(3):
                for b in range(3):
                    perturbation_matrix[i, j] += ccf[3*i + a, 3*j + b] * ccf[3*i + a, 3*j + b]
            perturbation_matrix[i, j] = sqrt(perturbation_matrix[i, j])
            sum_val += perturbation_matrix[i, j] 
    if normalize and sum_val != 0.0:
        norm = n * m / sum_val  
        for i in prange(m, nogil=True, schedule='static'):
            for j in range(n):
                perturbation_matrix[i, j] *= norm               
    return perturbation_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def td_perturbation_matrix(np.ndarray[double, ndim=3] ccf, bint normalize=True) -> np.ndarray:
    """
    Calculate the time-dependent perturbation matrix from a td-correlation matrix using block-wise norms.

    The input covariance matrix 'ccf' is expected to have shape (3*m, 3*n, nt). For each block (i,j, it),
    the perturbation value is computed as the square root of the sum of the squares of the corresponding
    3x3 block elements.

    Parameters
    ----------
    ccf : ndarray of double, 3D
        Input covariance matrix with shape (3*m, 3*n, nt).
    normalize : bool, optional
        If True, the output perturbation matrix is normalized so that the total sum of its elements equals 1
        (default is True).

    Returns
    -------
    perturbation_matrix : ndarray of double, 3D
        An (m, n, nt) matrix of perturbation values computed from the blocks of ccf.
    """
    cdef int m = ccf.shape[0] // 3
    cdef int n = ccf.shape[1] // 3
    cdef int nt = ccf.shape[2]
    cdef int i, j, it, a, b
    cdef double temp, sum_val = 0.0
    cdef np.ndarray[double, ndim=3] perturbation_matrix = np.empty((m, n, nt), dtype=np.float64)
    
    # Compute the block-wise norm for each (i,j,it) block.
    for it in range(nt):
        for i in range(m):
            for j in range(n):
                temp = 0.0
                for a in range(3):
                    for b in range(3):
                        temp += ccf[3*i + a, 3*j + b, it] * ccf[3*i + a, 3*j + b, it]
                perturbation_matrix[i, j, it] = sqrt(temp)
                if it == 0:
                    sum_val += perturbation_matrix[i, j, it]
    
    if normalize and sum_val != 0.0:
        norm = n * m / sum_val  
        for i in range(m):
            for j in range(n):
                for it in range(nt):
                    perturbation_matrix[i, j, it] *= norm
                
    return perturbation_matrix    


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def td_perturbation_matrix_par(np.ndarray[double, ndim=3] ccf, bint normalize=True) -> np.ndarray:
    """
    Calculate the time-dependent perturbation matrix from a td-correlation matrix using block-wise norms.

    The input covariance matrix 'ccf' is expected to have shape (3*m, 3*n, nt). For each block (i,j, it),
    the perturbation value is computed as the square root of the sum of the squares of the corresponding
    3x3 block elements.

    Parameters
    ----------
    ccf : ndarray of double, 3D
        Input covariance matrix with shape (3*m, 3*n, nt).
    normalize : bool, optional
        If True, the output perturbation matrix is normalized so that the total sum of its elements equals 1
        (default is True).

    Returns
    -------
    perturbation_matrix : ndarray of double, 3D
        An (m, n, nt) matrix of perturbation values computed from the blocks of ccf.
    """
    cdef int m = ccf.shape[0] // 3
    cdef int n = ccf.shape[1] // 3
    cdef int nt = ccf.shape[2]
    cdef int i, j, it, a, b
    cdef double temp, sum_val = 0.0
    cdef np.ndarray[double, ndim=3] perturbation_matrix = np.empty((m, n, nt), dtype=np.float64)
    
    # Compute the block-wise norm for each (i,j,it) block.
    for it in range(nt):
        for i in range(m):
            for j in range(n):
                temp = 0.0
                for a in range(3):
                    for b in range(3):
                        temp += ccf[3*i + a, 3*j + b, it] * ccf[3*i + a, 3*j + b, it]
                perturbation_matrix[i, j, it] = sqrt(temp)
                if it == 0:
                    sum_val += perturbation_matrix[i, j, it]
    
    if normalize and sum_val != 0.0:
        norm = n * m / sum_val  
        for i in range(m):
            for j in range(n):
                for it in range(nt):
                    perturbation_matrix[i, j, it] *= norm
                
    return perturbation_matrix    


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def covmat(np.ndarray[double, ndim=2] X):
    """
    Naive mimicing of:
    
        covmat = np.cov(centered_positions, rowvar=True, dtype=np.float64)
    
    Much slower, as expected.
    """
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]
    cdef np.ndarray[double, ndim=2] cov = np.zeros((m, m), dtype=np.float64)
    cdef int i, j, k
    # Loop over the variables (rows) to compute covariance elements.
    for i in range(m):
        for j in range(i, m):  # use symmetry: cov[i, j] == cov[j, i]
            for k in range(n):
                cov[i, j] += X[i, k] * X[j, k]
            cov[i, j] /= (n - 1)
            cov[j, i] = cov[i, j]
    return cov


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def pcovmat(np.ndarray[double, ndim=2] X):
    """
    Parallel computation of covariance matrix using OpenMP (via prange).
    Assumes X is already centered (mean subtracted).
    """
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]
    cdef np.ndarray[double, ndim=2] cov = np.zeros((m, m), dtype=np.float64)
    cdef int i, j, k
    cdef int num_threads = omp_get_max_threads()
    # Create typed memoryviews for fast access.
    cdef double[:, ::1] x_view = X
    cdef double[:, ::1] cov_view = cov
    for i in prange(m, schedule='static', nogil=True):
        for j in range(i, m): #, schedule='static'): 
            for k in range(n):
                cov_view[i, j] += x_view[i, k] * x_view[j, k]
            cov_view[i, j] /= (n - 1)
            cov_view[j, i] = cov_view[i, j]  # mirror
    return cov

