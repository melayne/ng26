"""Tests for the framework-independent linear-system solvers."""

import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from src.linear_solver.linear_solver import (
    LinearSystem,
    cg_solve,
    direct_solve,
    jacobi_preconditioner,
    solve_linear_system,
)


@pytest.mark.parametrize(
    "matrix_factory",
    [np.asarray, sparse.csr_matrix, sparse.csr_array],
)
def test_direct_solve_handles_dense_and_sparse_matrices(matrix_factory):
    A = matrix_factory([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    solution = direct_solve(A, b)

    assert solution.shape == b.shape
    assert np.allclose(A @ solution, b)


def test_direct_solve_preserves_complex_values():
    A = np.array([[2.0 + 1.0j, 0.0], [0.0, 3.0 - 1.0j]])
    b = np.array([1.0j, 2.0 + 1.0j])

    solution = direct_solve(A, b)

    assert np.iscomplexobj(solution)
    assert np.allclose(A @ solution, b)


def test_direct_solve_rejects_multiple_right_hand_sides():
    with pytest.raises(ValueError, match="one-dimensional"):
        direct_solve(np.eye(2), np.ones((2, 2)))


def test_direct_solve_rejects_linear_operator():
    A = LinearOperator((2, 2), matvec=lambda x: x)

    with pytest.raises(TypeError, match="direct solver"):
        direct_solve(A, np.ones(2))


def test_direct_method_does_not_warn_for_none_string():
    system = LinearSystem(A=np.eye(2), b=np.ones(2))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = solve_linear_system(
            system,
            method="direct",
            preconditioner="none",
        )

    assert result.success


def test_direct_method_warns_for_an_actual_preconditioner():
    system = LinearSystem(A=np.eye(2), b=np.ones(2))

    with pytest.warns(UserWarning, match="ignored"):
        solve_linear_system(
            system,
            method="direct",
            preconditioner="jacobi",
        )


def test_cg_with_jacobi_converges_without_default_history():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    result = cg_solve(A, b, preconditioner="jacobi")

    assert result.success
    assert result.iterations is not None and result.iterations > 0
    assert result.residual_history == []
    assert np.allclose(A @ result.solution, b)


def test_cg_can_record_residual_history():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    result = cg_solve(A, b, record_residual_history=True)

    assert result.iterations == len(result.residual_history)
    assert result.residual_history


def test_cg_preserves_complex_values():
    A = np.array([[4.0 + 0.0j, 1.0j], [-1.0j, 3.0 + 0.0j]])
    b = np.array([1.0 + 1.0j, 2.0 - 0.5j])

    result = cg_solve(A, b, preconditioner="jacobi")

    assert result.success
    assert np.iscomplexobj(result.solution)
    assert np.allclose(A @ result.solution, b)


def test_jacobi_rejects_rectangular_matrix():
    with pytest.raises(ValueError, match="square"):
        jacobi_preconditioner(np.ones((2, 3)))


def test_gmres_is_explicitly_unimplemented():
    system = LinearSystem(A=np.eye(2), b=np.ones(2))

    with pytest.raises(NotImplementedError, match="GMRES"):
        solve_linear_system(system, method="gmres")


def test_unsupported_solver_method_is_rejected():
    system = LinearSystem(A=np.eye(2), b=np.ones(2))

    with pytest.raises(ValueError, match="direct.*cg.*gmres"):
        solve_linear_system(system, method="bicgstab")  # type: ignore[arg-type]
