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


if __name__ == '__main__':
    pytest.main([__file__])

