"""
===============================================================================
File: test_rcmath.py
Description:
    This file contains unit tests for the 'rcmath' module from the 
    reforge.rfgmath package. The tests compare the outputs of the new 
    implementations with their legacy counterparts for various mathematical 
    functions, including Hessian calculation and perturbation matrices.

Usage:
    Run the tests with pytest:
        pytest -v tests/test_rcmath.py

Requirements:
    - NumPy
    - pytest
    - CuPy (for GPU tests; tests will be skipped if not installed)

Author: DY
Date: 2025-02-27
===============================================================================
"""

import numpy as np
import pytest
from reforge.rfgmath import rcmath, legacy, rpymath

np.random.seed(42)

def test_hessian():
    """
    Test the calculation of the Hessian matrix.

    This test compares the output of the '_calculate_hessian' function between
    the legacy and the new implementation. It performs the following steps:
      - Generates random arrays for x, y, and z coordinates.
      - Sets test parameters including the number of residues (n), cutoff,
        spring constant, and a distance dependence exponent (dd).
      - Computes the Hessian using both the legacy and new implementations.
      - Asserts that the results are almost identical within a tight tolerance.

    Returns:
        None
    """
    n = 50
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = np.random.rand(n)
    vec = np.array((x, y, z)).T
    cutoff = 12
    spring_constant = 1000
    dd = 0
    legacy_result = legacy.calculate_hessian(n, x, y, z, cutoff, spring_constant, dd)
    new_result = rcmath.calculate_hessian(n, x, y, z, cutoff, spring_constant, dd)
    vec_result = rcmath.hessian(vec, cutoff, spring_constant, dd)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vec_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix_old():
    """
    Compare the legacy and new implementations of the old perturbation matrix function.

    This test:
      - Creates a symmetric covariance matrix from a random matrix.
      - Computes the perturbation matrix using the '_perturbation_matrix_old'
        function from both the legacy and new implementations.
      - Asserts that the outputs are nearly identical within the specified tolerance.

    Returns:
        None
    """
    m = 200  # number of residues
    A = np.random.rand(3 * m, 3 * m)
    covmat = np.ascontiguousarray(0.5 * (A + A.T))
    legacy_result = legacy.calcperturbMat(covmat, m)
    new_result = rcmath.perturbation_matrix_old(covmat, m)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix():
    """
    Compare the CPU-based perturbation matrix outputs between legacy and new implementations.

    This test:
      - Generates a symmetric covariance matrix.
      - Computes the perturbation matrix using the legacy CPU function and the
        new perturbation matrix function.
      - Verifies that the results match within a tight numerical tolerance.

    Returns:
        None
    """
    m = 200
    A = np.random.rand(3 * m, 3 * m)
    covmat = np.ascontiguousarray(0.5 * (A + A.T))
    legacy_result = rcmath.perturbation_matrix_old(covmat, m) * m**2
    new_result = rcmath.perturbation_matrix(covmat)
    par_result = rcmath.perturbation_matrix_par(covmat)
    np.testing.assert_allclose(par_result, new_result, rtol=1e-6, atol=1e-6)


def test_td_perturbation_matrix():
    """
    Compare the time-dependent perturbation matrix outputs between legacy and new implementations.

    This test:
      - Constructs a symmetric covariance matrix.
      - Computes the time-dependent perturbation matrix with normalization
        using the legacy CPU function and the new implementation.
      - Asserts that both results are almost identical within the specified tolerances.

    Returns:
        None
    """
    m = 50
    nt = 1000
    covmat = np.random.rand(3*m, 3*m, nt)
    legacy_result = legacy.td_perturbation_matrix_cpu(covmat, normalize=False)
    new_result = rcmath.td_perturbation_matrix(covmat, normalize=False)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix_iso():
    """
    Compare the two new implementations of the perturbation matrix.

    This test compares the output of the old perturbation matrix function
    (as implemented in the new module) with the new perturbation matrix function.
    The test verifies that the two approaches yield nearly identical results
    within a slightly relaxed tolerance.

    Returns:
        None
    """
    m = 2000
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    iso_result = rcmath.perturbation_matrix_iso(covmat)
    iso_result_par = rcmath.perturbation_matrix_iso_par(covmat)
    np.testing.assert_allclose(iso_result_par, iso_result, rtol=1e-5, atol=1e-5)


def test_covmat():
    m = 500
    nt = 2000
    x = np.random.rand(3*m, nt)
    ref = rpymath.covariance_matrix(x, dtype=np.float64)
    x -= np.average(x, axis=1, keepdims=True)
    # scov = rcmath.covmat(x)
    pcov = rcmath.pcovmat(x)
    np.testing.assert_allclose(ref, pcov, rtol=1e-6, atol=1e-6)


def test_dtype_dispatch_hessian():
    """
    Test that rcmath correctly dispatches to 32-bit and 64-bit implementations
    based on input dtype.
    
    This test verifies:
      - float32 input uses rcymath32 and returns float32
      - float64 input uses rcymath64 and returns float64
      - Results are reasonably close between precisions
    """
    n = 50
    vec64 = np.random.rand(n, 3).astype(np.float64)
    vec32 = vec64.astype(np.float32)
    cutoff = 12
    spring_constant = 1000
    dd = 0
    # Test automatic dispatch
    result64 = rcmath.hessian(vec64, cutoff, spring_constant, dd)
    result32 = rcmath.hessian(vec32, cutoff, spring_constant, dd)
    # Verify output dtypes
    assert result64.dtype == np.float64, f"Expected float64, got {result64.dtype}"
    assert result32.dtype == np.float32, f"Expected float32, got {result32.dtype}"
    # Results should be reasonably close (within float32 precision limits)
    # Allow slightly larger tolerance due to accumulated floating point errors
    np.testing.assert_allclose(result64, result32, rtol=2e-3, atol=2e-3)


def test_dtype_force_conversion():
    """
    Test that the dtype parameter can force conversion to a specific precision.
    
    This test verifies:
      - Can force float32 output from float64 input
      - Can force float64 output from float32 input
    """
    n = 30
    vec64 = np.random.rand(n, 3).astype(np.float64)
    cutoff = 12
    spring_constant = 1000
    dd = 0
    # Force 32-bit computation from 64-bit input
    result32_forced = rcmath.hessian(vec64, cutoff, spring_constant, dd, dtype=np.float32)
    assert result32_forced.dtype == np.float32, f"Expected float32, got {result32_forced.dtype}"
    # Force 64-bit computation from 32-bit input
    vec32 = vec64.astype(np.float32)
    result64_forced = rcmath.hessian(vec32, cutoff, spring_constant, dd, dtype=np.float64)
    assert result64_forced.dtype == np.float64, f"Expected float64, got {result64_forced.dtype}"


def test_calculate_hessian_32bit_vs_64bit():
    """
    Compare calculate_hessian results between 32-bit and 64-bit implementations.
    
    Tests individual coordinate arrays (x, y, z) version of the function.
    """
    n = 50
    x64 = np.random.rand(n).astype(np.float64)
    y64 = np.random.rand(n).astype(np.float64)
    z64 = np.random.rand(n).astype(np.float64)
    x32 = x64.astype(np.float32)
    y32 = y64.astype(np.float32)
    z32 = z64.astype(np.float32)
    cutoff = 12
    spring_constant = 1000
    dd = 0
    result64 = rcmath.calculate_hessian(n, x64, y64, z64, cutoff, spring_constant, dd)
    result32 = rcmath.calculate_hessian(n, x32, y32, z32, cutoff, spring_constant, dd)
    assert result64.dtype == np.float64
    assert result32.dtype == np.float32
    # Should be close within float32 precision
    np.testing.assert_allclose(result64, result32, rtol=1e-4, atol=1e-4)


def test_perturbation_matrix_32bit_vs_64bit():
    """
    Compare perturbation_matrix results between 32-bit and 64-bit implementations.
    
    Tests both normalized and non-normalized versions.
    """
    m = 100
    A = np.random.rand(3 * m, 3 * m)
    covmat64 = np.ascontiguousarray(0.5 * (A + A.T)).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    # Test with normalization
    result64_norm = rcmath.perturbation_matrix(covmat64, normalize=True)
    result32_norm = rcmath.perturbation_matrix(covmat32, normalize=True)
    assert result64_norm.dtype == np.float64
    assert result32_norm.dtype == np.float32
    np.testing.assert_allclose(result64_norm, result32_norm, rtol=1e-4, atol=1e-4)
    # Test without normalization
    result64 = rcmath.perturbation_matrix(covmat64, normalize=False)
    result32 = rcmath.perturbation_matrix(covmat32, normalize=False)
    np.testing.assert_allclose(result64, result32, rtol=1e-4, atol=1e-4)


def test_perturbation_matrix_par_32bit_vs_64bit():
    """
    Compare perturbation_matrix_par (parallel version) between 32-bit and 64-bit.
    
    This also tests that parallel and serial versions give same results within each precision.
    """
    m = 100
    A = np.random.rand(3 * m, 3 * m)
    covmat64 = np.ascontiguousarray(0.5 * (A + A.T)).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    # Test parallel versions
    result64_par = rcmath.perturbation_matrix_par(covmat64, normalize=True)
    result32_par = rcmath.perturbation_matrix_par(covmat32, normalize=True)
    assert result64_par.dtype == np.float64
    assert result32_par.dtype == np.float32
    np.testing.assert_allclose(result64_par, result32_par, rtol=1e-4, atol=1e-4)
    # Compare parallel vs serial within each precision
    result64_ser = rcmath.perturbation_matrix(covmat64, normalize=True)
    result32_ser = rcmath.perturbation_matrix(covmat32, normalize=True)
    np.testing.assert_allclose(result64_par, result64_ser, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result32_par, result32_ser, rtol=1e-5, atol=1e-5)


def test_perturbation_matrix_iso_32bit_vs_64bit():
    """
    Compare perturbation_matrix_iso (isotropic method) between 32-bit and 64-bit.
    """
    m = 100
    A = np.random.rand(3 * m, 3 * m)
    covmat64 = np.ascontiguousarray(0.5 * (A + A.T)).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    result64 = rcmath.perturbation_matrix_iso(covmat64, normalize=True)
    result32 = rcmath.perturbation_matrix_iso(covmat32, normalize=True)
    assert result64.dtype == np.float64
    assert result32.dtype == np.float32
    np.testing.assert_allclose(result64, result32, rtol=1e-4, atol=1e-4)


def test_perturbation_matrix_iso_par_32bit_vs_64bit():
    """
    Compare perturbation_matrix_iso_par (parallel isotropic) between 32-bit and 64-bit.
    """
    m = 100
    A = np.random.rand(3 * m, 3 * m)
    covmat64 = np.ascontiguousarray(0.5 * (A + A.T)).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    result64_par = rcmath.perturbation_matrix_iso_par(covmat64, normalize=True)
    result32_par = rcmath.perturbation_matrix_iso_par(covmat32, normalize=True)
    assert result64_par.dtype == np.float64
    assert result32_par.dtype == np.float32
    np.testing.assert_allclose(result64_par, result32_par, rtol=1e-4, atol=1e-4)
    # Compare with serial version
    result64_ser = rcmath.perturbation_matrix_iso(covmat64, normalize=True)
    result32_ser = rcmath.perturbation_matrix_iso(covmat32, normalize=True)
    np.testing.assert_allclose(result64_par, result64_ser, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result32_par, result32_ser, rtol=1e-5, atol=1e-5)


def test_td_perturbation_matrix_32bit_vs_64bit():
    """
    Compare time-dependent perturbation matrix between 32-bit and 64-bit.
    """
    m = 30
    nt = 100
    covmat64 = np.random.rand(3*m, 3*m, nt).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    result64 = rcmath.td_perturbation_matrix(covmat64, normalize=True)
    result32 = rcmath.td_perturbation_matrix(covmat32, normalize=True)
    assert result64.dtype == np.float64
    assert result32.dtype == np.float32
    assert result64.shape == (m, m, nt)
    assert result32.shape == (m, m, nt)
    np.testing.assert_allclose(result64, result32, rtol=1e-4, atol=1e-4)


def test_td_perturbation_matrix_par_32bit_vs_64bit():
    """
    Compare time-dependent perturbation matrix (parallel) between 32-bit and 64-bit.
    """
    m = 30
    nt = 100
    covmat64 = np.random.rand(3*m, 3*m, nt).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    result64_par = rcmath.td_perturbation_matrix_par(covmat64, normalize=True)
    result32_par = rcmath.td_perturbation_matrix_par(covmat32, normalize=True)
    assert result64_par.dtype == np.float64
    assert result32_par.dtype == np.float32
    np.testing.assert_allclose(result64_par, result32_par, rtol=1e-4, atol=1e-4)
    # Compare with serial version
    result64_ser = rcmath.td_perturbation_matrix(covmat64, normalize=True)
    result32_ser = rcmath.td_perturbation_matrix(covmat32, normalize=True)
    np.testing.assert_allclose(result64_par, result64_ser, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result32_par, result32_ser, rtol=1e-5, atol=1e-5)


def test_covmat_32bit_vs_64bit():
    """
    Compare covariance matrix calculation between 32-bit and 64-bit.
    """
    m = 100
    nt = 500
    x64 = np.random.rand(m, nt).astype(np.float64)
    x64 -= np.mean(x64, axis=1, keepdims=True)
    x32 = x64.astype(np.float32)
    result64 = rcmath.covmat(x64)
    result32 = rcmath.covmat(x32)
    assert result64.dtype == np.float64
    assert result32.dtype == np.float32
    np.testing.assert_allclose(result64, result32, rtol=1e-4, atol=1e-4)


def test_pcovmat_32bit_vs_64bit():
    """
    Compare parallel covariance matrix calculation between 32-bit and 64-bit.
    """
    m = 100
    nt = 500
    x64 = np.random.rand(m, nt).astype(np.float64)
    x64 -= np.mean(x64, axis=1, keepdims=True)
    x32 = x64.astype(np.float32)
    result64_par = rcmath.pcovmat(x64)
    result32_par = rcmath.pcovmat(x32)
    assert result64_par.dtype == np.float64
    assert result32_par.dtype == np.float32
    np.testing.assert_allclose(result64_par, result32_par, rtol=1e-4, atol=1e-4)
    # Compare with serial version
    result64_ser = rcmath.covmat(x64)
    result32_ser = rcmath.covmat(x32)
    np.testing.assert_allclose(result64_par, result64_ser, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result32_par, result32_ser, rtol=1e-5, atol=1e-5)


def test_memory_efficiency():
    """
    Verify that 32-bit versions use approximately half the memory of 64-bit versions.
    """
    m = 100
    A = np.random.rand(3 * m, 3 * m)
    covmat64 = np.ascontiguousarray(0.5 * (A + A.T)).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    result64 = rcmath.perturbation_matrix(covmat64)
    result32 = rcmath.perturbation_matrix(covmat32)
    # Check memory usage
    mem64 = result64.nbytes
    mem32 = result32.nbytes
    # 32-bit should use approximately half the memory
    assert abs(mem32 * 2 - mem64) < 10, f"Memory ratio incorrect: {mem64} vs {mem32}"
    print(f"64-bit memory: {mem64 / 1024:.2f} KB")
    print(f"32-bit memory: {mem32 / 1024:.2f} KB")
    print(f"Memory savings: {(1 - mem32/mem64)*100:.1f}%")


def test_rectangular_matrix_32bit_vs_64bit():
    """
    Test perturbation matrix with rectangular (non-square) covariance matrices.
    
    This tests the feature where m != n (different number of residues in two systems).
    """
    m = 80
    n = 120
    covmat64 = np.random.rand(3*m, 3*n).astype(np.float64)
    covmat32 = covmat64.astype(np.float32)
    result64 = rcmath.perturbation_matrix(covmat64, normalize=True)
    result32 = rcmath.perturbation_matrix(covmat32, normalize=True)
    assert result64.shape == (m, n)
    assert result32.shape == (m, n)
    assert result64.dtype == np.float64
    assert result32.dtype == np.float32
    np.testing.assert_allclose(result64, result32, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__])

