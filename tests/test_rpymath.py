"""
===============================================================================
File: test_rpymath.py
Description:
    This file contains unit tests for the 'rpymath' module in the 
    reforge.rfgmath package. The tests verify various mathematical 
    routines including covariance matrix computation, Fourier transform based 
    correlation functions, and matrix inversion (both CPU and GPU implementations).

Usage:
    Run the tests with pytest:
        pytest -v tests/test_rpymath.py

Requirements:
    - NumPy
    - pytest
    - CuPy (for GPU tests; tests will be skipped if not installed)

Author: DY
Date: 2025-02-27
===============================================================================
"""

import os
import numpy as np
import pytest
from reforge.rfgmath import rpymath

# Skip GPU tests if CUDA is not available
try:
    import cupy as cp
    # Check if CUDA is actually available and accessible
    cuda_available = cp.cuda.is_available()
    if cuda_available:
        # Try to access a CUDA device to ensure it's working
        try:
            cp.cuda.Device(0).use()
            cuda_available = True
        except cp.cuda.runtime.CUDARuntimeError:
            cuda_available = False
    else:
        cuda_available = False
except ImportError:
    cp = None
    cuda_available = False


def test_covariance_matrix():
    """
    Verify the covariance matrix computation from the rpymath module.

    This test performs the following steps:
      - Constructs a simple 3x3 positions array and replicates it to simulate
        multiple frames.
      - Computes the covariance matrix using the internal _covariance_matrix function.
      - Asserts that the resulting covariance matrix has the expected shape.
      - Verifies that for the degenerate (linearly dependent) data, the determinant 
        of the covariance matrix is nearly zero.

    Returns:
        None
    """
    # Create a simple positions array: shape (n_coords, n_frames)
    positions = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=np.float64)
    nt = 10
    positions = np.tile(positions, (nt, nt))
    covmat = rpymath.covariance_matrix(positions, dtype=np.float64)
    # Expect a (3*nt, 3*nt) covariance matrix when rowvar=True.
    assert covmat.shape == (3*nt, 3*nt)
    # For this degenerate data (linearly dependent), the determinant should be ~0.
    np.testing.assert_almost_equal(np.linalg.det(covmat), 0, decimal=5)


def test_sfft_ccf():
    """
    Validate the sliding Fourier transform correlation function (_sfft_corr).

    This test:
      - Generates two random arrays and centers them by subtracting the mean.
      - Computes the correlation using _sfft_corr with a sliding average approach.
      - Manually calculates the reference correlation for each time shift.
      - Compares the FFT-based correlation result with the manual calculation,
        ensuring they match within a very tight tolerance.

    Returns:
        None
    """
    n_coords = 10
    n_samples = 256
    ntmax = 64
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    x_centered = x - np.mean(x, axis=-1, keepdims=True)
    y_centered = y - np.mean(y, axis=-1, keepdims=True)
    corr_fft = rpymath.sfft_ccf(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)
    ref_corr = np.empty((n_coords, n_coords, n_samples), dtype=np.float64)
    # Manually compute the sliding average correlation.
    for i in range(n_coords):
        for j in range(n_coords):
            for t in range(n_samples):
                ref_corr[i, j, t] = np.average(x_centered[i, t:] * y_centered[j, :n_samples-t])
    ref_corr = ref_corr[:, :, :ntmax]
    np.testing.assert_allclose(corr_fft, ref_corr, rtol=1e-10, atol=1e-10)


def test_pfft_ccf():
    """
    Compare the parallel (_pfft_corr) and serial (_sfft_corr) FFT-based correlation methods.

    This test:
      - Generates two random input signals.
      - Computes the correlation using both the parallel and serial methods.
      - Asserts that the results from both methods are nearly identical within 
        a very tight numerical tolerance.

    Returns:
        None
    """
    n_coords = 10
    n_samples = 256
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    ntmax = 64
    corr_par = rpymath.pfft_ccf(x, y, ntmax=ntmax, center=True, dtype=np.float64)
    corr_ser = rpymath.sfft_ccf(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)
    np.testing.assert_allclose(corr_par, corr_ser, rtol=1e-10, atol=1e-10)


def test_ccf():
    """
    Test the cross-correlation function (ccf) against a manual sliding average calculation.

    This test:
      - Splits random signals into several segments.
      - For each segment, computes the centered signals and manually averages 
        the product over a range of time shifts.
      - Averages the results over all segments to obtain a manual cross-correlation.
      - Compares this manual computation with the output of the ccf function in both 
        parallel and serial modes.
      - Asserts that the results match within the specified tolerance.

    Returns:
        None
    """
    n_coords = 10
    n_samples = 256
    n_seg = 4  # number of segments
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    segments_x = np.array_split(x, n_seg, axis=-1)
    segments_y = np.array_split(y, n_seg, axis=-1)
    manual_corr_sum = None
    for seg_x, seg_y in zip(segments_x, segments_y):
        x_centered = seg_x - np.mean(seg_x, axis=-1, keepdims=True)
        y_centered = seg_y - np.mean(seg_y, axis=-1, keepdims=True)
        nt_seg = seg_x.shape[-1]
        ntmax_seg = (nt_seg + 1) // 2
        manual_corr_seg = np.empty((n_coords, n_coords, ntmax_seg), dtype=np.float64)
        for i in range(n_coords):
            for j in range(n_coords):
                for tau in range(ntmax_seg):
                    window = x_centered[i, tau:nt_seg] * y_centered[j, :nt_seg-tau]
                    manual_corr_seg[i, j, tau] = np.average(window)
        if manual_corr_sum is None:
            manual_corr_sum = manual_corr_seg
        else:
            manual_corr_sum += manual_corr_seg
    manual_ccf = manual_corr_sum / n_seg
    par_ccf = rpymath.ccf(x, y, ntmax=None, n=n_seg, mode='parallel', center=True, dtype=np.float64)
    ser_ccf = rpymath.ccf(x, y, ntmax=None, n=n_seg, mode='serial', center=True, dtype=np.float64)
    np.testing.assert_allclose(manual_ccf, par_ccf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(manual_ccf, ser_ccf, rtol=1e-6, atol=1e-6)


def test_inverse_sparse_matrix_cpu():
    """
    Verify the CPU-based sparse matrix inversion function (_inverse_sparse_matrix_cpu).

    This test:
      - Constructs a diagonal matrix with known values.
      - Inverts the matrix using _inverse_sparse_matrix_cpu.
      - Compares the computed inverse with the expected inverse (reciprocals of the diagonal elements).
    
    Returns:
        None
    """
    N = 100
    diag_vals = np.linspace(1, 1e7, N)
    matrix = np.diag(diag_vals)
    # Invert the matrix with all eigenvalues inverted.
    inv_matrix = rpymath.inverse_sparse_matrix_cpu(matrix, k_singular=0, n_modes=N-1)
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=1e-6, atol=1e-6)


def test_inverse_matrix_cpu():
    """
    Verify the CPU-based sparse matrix inversion function (_inverse_sparse_matrix_cpu).

    This test:
      - Constructs a diagonal matrix with known values.
      - Inverts the matrix using _inverse_sparse_matrix_cpu.
      - Compares the computed inverse with the expected inverse (reciprocals of the diagonal elements).
    
    Returns:
        None
    """
    N = 100
    diag_vals = np.linspace(1, 1e7, N)
    matrix = np.diag(diag_vals)
    # Invert the matrix with all eigenvalues inverted.
    inv_matrix = rpymath.inverse_matrix_cpu(matrix, k_singular=0, n_modes=N)
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=1e-6, atol=1e-6)

#############################
## GPU tests ##
#############################

@pytest.mark.skipif(not cuda_available, reason='CUDA not available or not accessible')
def test_gfft_ccf():
    """
    Validate the GPU-based FFT correlation function (gfft_corr).

    This test:
      - Uses CuPy to generate random input signals on the GPU.
      - Computes the correlation with gfft_corr and transfers the result to the CPU.
      - Compares the GPU-computed correlation with the serial CPU implementation (sfft_corr).
    
    Returns:
        None
    """
    n_coords = 10
    n_samples = 128
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    ntmax = 64
    corr_gpu = rpymath.gfft_ccf(x, y, ntmax=ntmax, center=True)
    corr = corr_gpu.get()
    corr_ser = rpymath.sfft_ccf(x, y, ntmax=ntmax, center=True, loop=True)
    np.testing.assert_allclose(corr, corr_ser, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not cuda_available, reason='CUDA not available or not accessible')
def test_gfft_ccf_auto():
    """
    Test that gfft_ccf_auto returns the same result as gfft_ccf
    """
    n_coords = 10
    n_samples = 128
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    ntmax = 64
    ref_gpu = rpymath.gfft_ccf(x, y, ntmax=ntmax, center=True)
    ref = ref_gpu.get()
    test = rpymath.gfft_ccf_auto(x, y, ntmax=ntmax, center=True)
    np.testing.assert_allclose(test, ref, rtol=1e-10, atol=1e-10)    


@pytest.mark.skipif(not cuda_available, reason='CUDA not available or not accessible')
def test_inverse_sparse_matrix_gpu():
    """
    Test the GPU-based sparse matrix inversion (_inverse_sparse_matrix_gpu).

    This test:
      - Creates a diagonal matrix.
      - Inverts it using the GPU implementation.
      - Converts the GPU result to a NumPy array and compares it with the CPU-based inversion.
    
    Returns:
        None
    """
    N = 200
    diag_vals = np.linspace(1, 10, N)
    matrix = np.diag(diag_vals)
    inv_matrix_gpu = rpymath.inverse_sparse_matrix_gpu(matrix, k_singular=0, n_modes=N//10, dtype=cp.float64)
    inv_matrix = cp.asnumpy(inv_matrix_gpu)
    expected_inv = rpymath.inverse_sparse_matrix_cpu(matrix, k_singular=0, n_modes=N//10)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=0, atol=1e-6)


@pytest.mark.skipif(not cuda_available, reason='CUDA not available or not accessible')
def test_inverse_matrix_gpu():
    """
    Verify the GPU matrix inversion function (_inverse_matrix_gpu).

    This test:
      - Constructs a diagonal matrix.
      - Inverts it using the GPU-based method that leverages cupy.linalg.eigh.
      - Converts the GPU result back to a NumPy array and compares it with the expected inverse.
    
    Returns:
        None
    """
    N = 200
    diag_vals = np.linspace(1, 10, N)
    matrix = np.diag(diag_vals).astype(np.float64)
    inv_matrix_gpu = rpymath.inverse_matrix_gpu(matrix, k_singular=0, n_modes=N)
    inv_matrix = cp.asnumpy(inv_matrix_gpu)
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
    # test_gfft_ccf()
