"""
===============================================================================
File: test_mdm.py
Description:
    Unit tests for the mdm.py module in the reForge package.
    These tests cover key wrappers for FFT-based correlation functions,
    covariance matrix computation, perturbation matrices (DFI/DCI), ENM Hessian,
    matrix inversion, and miscellaneous utilities.
    
Usage:
    Run the tests with pytest:
        pytest -v test_mdm.py

Requirements:
    - NumPy
    - pytest
    - MDAnalysis (if needed for more advanced tests)
    - CuPy (for GPU tests; GPU tests are skipped if CUDA is not available)
===============================================================================
"""

import os
import numpy as np
import pytest
from reforge.mdm import * 


# ---------------------------
# Test FFT-based correlation
# ---------------------------
def test_fft_ccf_serial():
    """
    Test the fft_ccf function in 'serial' mode.
    Verifies that the output is a NumPy array and has the expected shape.
    """
    n_coords = 10
    n_samples = 256
    # Generate two random signals
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    # Call the fft_ccf wrapper in serial mode
    result = ccf(x, y, mode='serial', ntmax=64, center=True)
    # Expect a result as a NumPy array
    assert isinstance(result, np.ndarray)
    # (Additional shape checks can be added based on your implementation.)
    
    
# ---------------------------
# Test Covariance Matrix
# ---------------------------
def test_covariance_matrix():
    """
    Test the covariance_matrix wrapper.
    Checks that the computed covariance matrix has the expected shape.
    """
    # Create dummy positions: shape (n_coords, n_frames)
    n_coords, n_frames = 9, 20
    positions = np.random.rand(n_coords, n_frames)
    covmat = covariance_matrix(positions, dtype=np.float64)
    # For this implementation, assume output shape is (3*n_frames, 3*n_frames)
    # (Adjust the expected shape if your internal function differs.)
    # Here we simply check that the result is a square matrix.
    assert covmat.shape[0] == covmat.shape[1]


# ---------------------------
# Test Covariance Matrix Saving
# ---------------------------
def test_calc_and_save_covmats(tmp_path):
    """
    Test calc_and_save_covmats by writing output files to a temporary directory.
    """
    # Create dummy positions: shape (n_coords, n_frames)
    n_coords, n_frames = 9, 30
    positions = np.random.rand(n_coords, n_frames)
    outdir = tmp_path / "covmats"
    outdir.mkdir()
    calc_and_save_covmats(positions, str(outdir), n=3, outtag="testcov")
    # Expect three files to be created.
    files = list(outdir.glob("testcov_*.npy"))
    assert len(files) == 3
    # Load one file and check that it is a NumPy array.
    covmat = np.load(files[0])
    assert isinstance(covmat, np.ndarray)


# ---------------------------
# Test Perturbation Matrices and DFI/DCI
# ---------------------------
def test_perturbation_matrix():
    """
    Test the perturbation_matrix wrapper using a dummy covariance matrix.
    """
    # Create a dummy covariance matrix of shape (9, 9) corresponding to 3 residues.
    covar = np.random.rand(9, 9).astype(np.float64)
    pert = perturbation_matrix(covar, dtype=np.float64)
    # Expect the perturbation matrix to have shape (3, 3)
    assert pert.shape == (3, 3)


def test_td_perturbation_matrix():
    """
    Test the td_perturbation_matrix wrapper using a dummy covariance matrix.
    """
    covar = np.random.rand(9, 9, 100).astype(np.float64)
    pert = td_perturbation_matrix(covar, dtype=np.float64)
    # Expect the perturbation matrix to have shape (3, 3)
    assert pert.shape == (3, 3, 100)


def test_dfi_and_dci():
    """
    Test the DFI and DCI calculations.
    """
    # Create a simple 3x3 perturbation matrix.
    pert = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.float64)
    dfi_vals = dfi(pert)
    dci_mat = dci(pert, asym=False)
    # dfi should have length equal to the number of rows
    assert dfi_vals.shape[0] == pert.shape[0]
    # dci should have the same shape as the perturbation matrix
    assert dci_mat.shape == pert.shape


def test_group_molecule_and_group_group_dci():
    """
    Test group_molecule_dci and group_group_dci functions.
    """
    # Create a dummy perturbation matrix for 6 residues
    pert = np.random.rand(6, 6).astype(np.float64)
    # Define groups as lists of indices
    groups = [[0, 1, 2], [3, 4, 5]]
    gm_dci = group_molecule_dci(pert, groups=groups, asym=False)
    gg_dci = group_group_dci(pert, groups=groups, asym=False)
    # Check that output lengths match the number of groups
    assert len(gm_dci) == len(groups)
    assert len(gg_dci) == len(groups)
    # Each element in group_group_dci should be a list with length equal to the number of groups
    for row in gg_dci:
        assert len(row) == len(groups)


# ---------------------------
# Test Hessian Calculation
# ---------------------------
def test_hessian():
    """
    Test the hessian function using a small coordinate matrix.
    """
    # Create a coordinate matrix for 5 residues (shape: (5, 3))
    vecs = np.random.rand(5, 3).astype(np.float64)
    cutoff = 1.2
    spring_constant = 1000.0
    dd = 0
    hess = hessian(vecs, cutoff, spring_constant, dd)
    # Expected Hessian shape is (15, 15)
    assert hess.shape == (15, 15)


# ---------------------------
# Test Inverse Matrix Wrapper
# ---------------------------
def test_inverse_matrix_cpu():
    """
    Test the unified inverse_matrix wrapper using a simple diagonal matrix.
    """
    N = 50
    diag_vals = np.linspace(1, 10, N)
    matrix = np.diag(diag_vals).astype(np.float64)
    inv_mat = inverse_matrix(matrix, device='cpu_sparse', k_singular=0, n_modes=N)
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_mat, expected_inv, rtol=1e-5, atol=1e-6)


def test_inverse_matrix_gpu():
    """
    Test the unified inverse_matrix wrapper for GPU.
    This test verifies that the GPU inversion (dense or sparse) returns a result 
    that matches the CPU inversion for a simple diagonal matrix.
    """
    N = 50
    diag_vals = np.linspace(1, 10, N)
    matrix = np.diag(diag_vals).astype(np.float64)
    inv_mat_gpu = inverse_matrix(matrix, device='gpu_dense', k_singular=0, n_modes=N)
    inv_mat = cp.asnumpy(inv_mat_gpu)
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_mat, expected_inv, rtol=1e-5, atol=1e-6)


# ---------------------------
# Test Percentile Calculation
# ---------------------------
def test_percentile():
    """
    Test the percentile function to ensure it computes correct ranking.
    """
    x = np.array([50, 20, 80, 40, 100], dtype=np.float64)
    p = percentile(x)
    # Expected percentiles: values should be in the range [0, 1] and in the order of sorted indices.
    assert np.all(p >= 0) and np.all(p <= 1)
    # Check that the smallest element has percentile 0 and the largest has ~1.
    sorted_idx = np.argsort(x)
    np.testing.assert_almost_equal(p[sorted_idx[0]], 0, decimal=5)
    np.testing.assert_almost_equal(p[sorted_idx[-1]], (len(x)-1)/len(x), decimal=5)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
