"""Cython Math Wrapper

Description:
    This module provides a unified interface to the Cython-optimized mathematical
    operations for both 32-bit (float) and 64-bit (double) precision. Functions
    automatically dispatch to the appropriate implementation based on the input
    array dtype.

    The module wraps rcymath32 and rcymath64, which contain the actual Cython
    implementations for 32-bit and 64-bit precision respectively.

Usage Example:
    >>> import numpy as np
    >>> from reforge.rfgmath import rcmath
    >>> 
    >>> # Use 64-bit precision (default)
    >>> vecs64 = np.random.rand(10, 3).astype(np.float64)
    >>> hessian64 = rcmath.hessian(vecs64, cutoff=1.2)
    >>> 
    >>> # Use 32-bit precision for faster computation / less memory
    >>> vecs32 = np.random.rand(10, 3).astype(np.float32)
    >>> hessian32 = rcmath.hessian(vecs32, cutoff=1.2)

Requirements:
    - Python 3.x
    - NumPy
    - Cython-compiled rcymath32 and rcymath64 modules

Author: Your Name
Date: 2025-11-28
"""

import numpy as np

try:
    from . import rcymath32
except ImportError:
    rcymath32 = None
    
try:
    from . import rcymath64
except ImportError:
    rcymath64 = None


def _get_implementation(arr, dtype=None):
    """
    Determine which implementation to use based on array dtype.
    
    Parameters
    ----------
    arr : ndarray
        Input array to check dtype
    dtype : dtype, optional
        Override dtype to use for implementation selection
    
    Returns
    -------
    module
        Either rcymath32 or rcymath64 module
    
    Raises
    ------
    ValueError
        If the dtype is not supported or the required module is not compiled
    """
    if dtype is None:
        dtype = arr.dtype
    
    # Normalize dtype
    dtype = np.dtype(dtype)
    
    if dtype == np.float32:
        if rcymath32 is None:
            raise ImportError("rcymath32 module is not compiled. Cannot use float32 precision.")
        return rcymath32
    elif dtype == np.float64:
        if rcymath64 is None:
            raise ImportError("rcymath64 module is not compiled. Cannot use float64 precision.")
        return rcymath64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Only float32 and float64 are supported.")


def calculate_hessian(resnum, x, y, z, cutoff=1.2, spring_constant=1000, dd=0, dtype=None):
    """
    Calculate the position-position Hessian matrix based on individual coordinate arrays.
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    resnum : int
        Number of residues (atoms).
    x : ndarray, 1D
        Array of x coordinates.
    y : ndarray, 1D
        Array of y coordinates.
    z : ndarray, 1D
        Array of z coordinates.
    cutoff : float, optional
        Distance cutoff threshold (default is 1.2).
    spring_constant : float, optional
        Base spring constant (default is 1000).
    dd : int, optional
        Exponent modifier (default is 0).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses x.dtype.

    Returns
    -------
    hessian : ndarray, 2D
        Hessian matrix of shape (3*resnum, 3*resnum).
    """
    impl = _get_implementation(x, dtype)
    
    # Ensure all arrays have the same dtype
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    x = np.asarray(x, dtype=target_dtype)
    y = np.asarray(y, dtype=target_dtype)
    z = np.asarray(z, dtype=target_dtype)
    
    return impl.calculate_hessian(resnum, x, y, z, cutoff, spring_constant, dd)


def hessian(vec, cutoff=1.2, spring_constant=1000, dd=0, dtype=None):
    """
    Calculate the position-position Hessian matrix from a coordinate matrix.
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    vec : ndarray, 2D
        Coordinate matrix (n_residues, 3).
    cutoff : float, optional
        Distance cutoff threshold (default is 1.2).
    spring_constant : float, optional
        Base spring constant (default is 1000).
    dd : int, optional
        Exponent modifier (default is 0).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses vec.dtype.

    Returns
    -------
    hessian : ndarray, 2D
        Hessian matrix of shape (3*n, 3*n).
    """
    impl = _get_implementation(vec, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    vec = np.asarray(vec, dtype=target_dtype)
    
    return impl.hessian(vec, cutoff, spring_constant, dd)


def perturbation_matrix_old(covariance_matrix, resnum, dtype=None):
    """
    Compute a perturbation matrix from a covariance matrix (older method).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    covariance_matrix : ndarray, 2D
        Covariance matrix of shape (3*resnum, 3*resnum).
    resnum : int
        Number of residues.
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses covariance_matrix.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 2D
        Normalized perturbation matrix of shape (resnum, resnum).
    """
    impl = _get_implementation(covariance_matrix, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    covariance_matrix = np.asarray(covariance_matrix, dtype=target_dtype)
    
    return impl.perturbation_matrix_old(covariance_matrix, resnum)


def perturbation_matrix(covariance_matrix, normalize=True, dtype=None):
    """
    Compute a perturbation matrix from a covariance matrix.
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    covariance_matrix : ndarray, 2D
        Covariance matrix of shape (3*m, 3*n).
    normalize : bool, optional
        Whether to normalize the output (default is True).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses covariance_matrix.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 2D
        Perturbation matrix of shape (m, n).
    """
    impl = _get_implementation(covariance_matrix, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    covariance_matrix = np.asarray(covariance_matrix, dtype=target_dtype)
    
    return impl.perturbation_matrix(covariance_matrix, normalize)


def perturbation_matrix_par(covariance_matrix, normalize=True, dtype=None):
    """
    Compute a perturbation matrix from a covariance matrix (parallel version).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    covariance_matrix : ndarray, 2D
        Covariance matrix of shape (3*m, 3*n).
    normalize : bool, optional
        Whether to normalize the output (default is True).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses covariance_matrix.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 2D
        Perturbation matrix of shape (m, n).
    """
    impl = _get_implementation(covariance_matrix, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    covariance_matrix = np.asarray(covariance_matrix, dtype=target_dtype)
    
    return impl.perturbation_matrix_par(covariance_matrix, normalize)


def perturbation_matrix_iso(ccf, normalize=True, dtype=None):
    """
    Calculate perturbation matrix using block-wise norms (isotropic method).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    ccf : ndarray, 2D
        Covariance matrix of shape (3*m, 3*n).
    normalize : bool, optional
        Whether to normalize the output (default is True).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses ccf.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 2D
        Perturbation matrix of shape (m, n).
    """
    impl = _get_implementation(ccf, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    ccf = np.asarray(ccf, dtype=target_dtype)
    
    return impl.perturbation_matrix_iso(ccf, normalize)


def perturbation_matrix_iso_par(ccf, normalize=True, dtype=None):
    """
    Calculate perturbation matrix using block-wise norms (parallel version).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    ccf : ndarray, 2D
        Covariance matrix of shape (3*m, 3*n).
    normalize : bool, optional
        Whether to normalize the output (default is True).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses ccf.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 2D
        Perturbation matrix of shape (m, n).
    """
    impl = _get_implementation(ccf, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    ccf = np.asarray(ccf, dtype=target_dtype)
    
    return impl.perturbation_matrix_iso_par(ccf, normalize)


def td_perturbation_matrix(ccf, normalize=True, dtype=None):
    """
    Calculate time-dependent perturbation matrix using block-wise norms.
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    ccf : ndarray, 3D
        Covariance matrix of shape (3*m, 3*n, nt).
    normalize : bool, optional
        Whether to normalize the output (default is True).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses ccf.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 3D
        Perturbation matrix of shape (m, n, nt).
    """
    impl = _get_implementation(ccf, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    ccf = np.asarray(ccf, dtype=target_dtype)
    
    return impl.td_perturbation_matrix(ccf, normalize)


def td_perturbation_matrix_par(ccf, normalize=True, dtype=None):
    """
    Calculate time-dependent perturbation matrix (parallel version).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    ccf : ndarray, 3D
        Covariance matrix of shape (3*m, 3*n, nt).
    normalize : bool, optional
        Whether to normalize the output (default is True).
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses ccf.dtype.

    Returns
    -------
    perturbation_matrix : ndarray, 3D
        Perturbation matrix of shape (m, n, nt).
    """
    impl = _get_implementation(ccf, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    ccf = np.asarray(ccf, dtype=target_dtype)
    
    return impl.td_perturbation_matrix_par(ccf, normalize)


def covmat(X, dtype=None):
    """
    Calculate covariance matrix (naive implementation).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    X : ndarray, 2D
        Centered data matrix.
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses X.dtype.

    Returns
    -------
    cov : ndarray, 2D
        Covariance matrix.
    """
    impl = _get_implementation(X, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    X = np.asarray(X, dtype=target_dtype)
    
    return impl.covmat(X)


def pcovmat(X, dtype=None):
    """
    Calculate covariance matrix (parallel implementation).
    
    Automatically dispatches to 32-bit or 64-bit implementation based on input dtype.

    Parameters
    ----------
    X : ndarray, 2D
        Centered data matrix.
    dtype : dtype, optional
        Force a specific dtype (float32 or float64). If None, uses X.dtype.

    Returns
    -------
    cov : ndarray, 2D
        Covariance matrix.
    """
    impl = _get_implementation(X, dtype)
    
    target_dtype = np.float32 if impl == rcymath32 else np.float64
    X = np.asarray(X, dtype=target_dtype)
    
    return impl.pcovmat(X)


# Expose all functions in __all__
__all__ = [
    'calculate_hessian',
    'hessian',
    'perturbation_matrix_old',
    'perturbation_matrix',
    'perturbation_matrix_par',
    'perturbation_matrix_iso',
    'perturbation_matrix_iso_par',
    'td_perturbation_matrix',
    'td_perturbation_matrix_par',
    'covmat',
    'pcovmat',
]
