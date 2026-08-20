#===============================================================================
# Imports
#===============================================================================
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal, TypeAlias

import numpy as np
from scipy import sparse
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import (
    LinearOperator,
    cg,
    gmres,
    spsolve,
)

#===============================================================================
# Type Aliases
#===============================================================================
MatrixLike: TypeAlias = (
    np.ndarray
    | sparse.spmatrix
    | sparse.sparray
    | LinearOperator
)

SolverMethod: TypeAlias = Literal["direct", "cg", "gmres"]

PreconditionerSpec: TypeAlias = (
    None
    | Literal["none", "jacobi", "gauss_seidel"]
    | MatrixLike
    | Callable[[np.ndarray], np.ndarray]
)
#===============================================================================
# Classes
#===============================================================================

@dataclass
class LinearSystem:
    """Representation of A x = b."""

    A: MatrixLike
    b: np.ndarray
    x0: np.ndarray | None = None

@dataclass
class SolveResult:
    """Information returned by a linear solve."""

    solution: np.ndarray
    success: bool
    solver_name: str
    iterations: int | None
    initial_residual_norm: float
    final_residual_norm: float
    residual_history: list[float]
    solve_time: float
    message: str

#===============================================================================
# Helper Functions
#===============================================================================
def _matrix_shape(A: MatrixLike) -> tuple[int, int]:
    """Return the shape of a supported two-dimensional matrix object."""
    if not (
        isinstance(A, (np.ndarray, LinearOperator))
        or sparse.issparse(A)
    ):
        raise TypeError(f"Unsupported matrix type: {type(A)}")

    shape = getattr(A, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("A must be two-dimensional.")

    return int(shape[0]), int(shape[1])


def _validate_linear_system(
    A: MatrixLike,
    b: np.ndarray,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate dimensions and normalize the system vectors."""
    n_rows, n_cols = _matrix_shape(A)
    if n_rows != n_cols:
        raise ValueError("A must be square.")

    b_vector = np.asarray(b)
    if b_vector.ndim != 1:
        raise ValueError("b must be one-dimensional.")
    if b_vector.shape[0] != n_rows:
        raise ValueError("The dimensions of A and b do not agree.")

    if x0 is None:
        return b_vector, None

    x0_vector = np.asarray(x0)
    if x0_vector.shape != b_vector.shape:
        raise ValueError("x0 must have the same shape as b.")

    return b_vector, x0_vector


def _solver_dtype(A: MatrixLike, vector: np.ndarray) -> np.dtype:
    """Choose an inexact solver dtype without discarding complex values."""
    matrix_dtype = getattr(A, "dtype", None)
    dtype = (
        np.result_type(vector.dtype)
        if matrix_dtype is None
        else np.result_type(matrix_dtype, vector.dtype)
    )

    if not np.issubdtype(dtype, np.inexact):
        return np.dtype(float)
    return np.dtype(dtype)


def as_csr_matrix(A: MatrixLike) -> sparse.csr_matrix:
    """Convert a MatrixLike object to a CSR sparse matrix."""
    if isinstance(A, LinearOperator):
        raise TypeError(
            "LinearOperator has no sparse matrix form; "
            "use iterative solvers with matvecs instead."
        )

    if isinstance(A, np.ndarray):
        return sparse.csr_matrix(A)

    if sparse.issparse(A):
        # csr_matrix accepts dense and sparse; avoids .tocsr() which
        # basedpyright cannot prove after issparse() (not a TypeGuard).
        return sparse.csr_matrix(A)

    raise TypeError(f"Unsupported matrix type: {type(A)}")

def matrix_density(A: np.ndarray) -> float:
    """Return the fraction of entries that are nonzero."""
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional.")

    if A.size == 0:
        return 0.0

    return np.count_nonzero(A) / A.size

def convert_dense_to_sparse(
    A: np.ndarray,
    maximum_density: float = 0.1,
) -> sparse.csr_matrix:
    """Convert a dense matrix to a sparse matrix."""

    if not 0.0 <= maximum_density <= 1.0:
        raise ValueError(
            "maximum_density must be between 0 and 1 inclusive."
        )

    density = matrix_density(A)

    if density > maximum_density:
        raise ValueError(
            f"Matrix density is {density:.1%}; "
            "sparse conversion may not be beneficial."
        )

    return sparse.csr_matrix(A)

#===============================================================================
# Preconditioners
#===============================================================================
def jacobi_preconditioner(
    A: np.ndarray | sparse.spmatrix | sparse.sparray,
) -> LinearOperator:
    r"""
    Return M^{-1} ≈ D^{-1} as a LinearOperator (D = diag(A)).

    Parameters
    ----------
    A
        The coefficient matrix A, A \in \mathbb{R}^{n \times n}.

    Returns
    -------
    LinearOperator
        The Jacobi preconditioner M^{-1} ≈ D^{-1}, D = diag(A).
    """

    n_rows, n_cols = _matrix_shape(A)
    if n_rows != n_cols:
        raise ValueError("Jacobi requires a square matrix.")

    if isinstance(A, np.ndarray):
        diagonal = np.asarray(np.diag(A)).copy()
    elif sparse.issparse(A):
        # Convert first: SciPy stubs omit `.diagonal` on the generic
        # spmatrix/sparray types after issparse().
        A = sparse.csr_matrix(A)
        diagonal = np.asarray(A.diagonal())
    else:
        raise TypeError(f"Unsupported matrix type for Jacobi: {type(A)}")

    if np.any(diagonal == 0):
        raise ValueError("Jacobi requires a nonzero diagonal.")

    inverse_diagonal = 1.0 / diagonal
    return LinearOperator(
        shape=(n_rows, n_cols),
        matvec=lambda r: inverse_diagonal * np.asarray(r),
        dtype=inverse_diagonal.dtype,
    )

def gauss_seidel_preconditioner():
    ...

def build_preconditioner(
    A: MatrixLike,
    preconditioner: PreconditionerSpec = None,
) -> MatrixLike | None:
    """Normalize a preconditioner specification for SciPy solvers.

    A matrix or ``LinearOperator`` supplied here must approximate ``A^{-1}``.
    SciPy applies it directly to each residual; SciPy does not solve a system
    with the supplied preconditioner matrix.
    """

    if preconditioner is None:
        return None

    if isinstance(preconditioner, str):
        if preconditioner == "none":
            return None
        if preconditioner == "jacobi":
            if isinstance(A, LinearOperator):
                raise TypeError(
                    "Jacobi needs an explicit matrix to read the diagonal."
                )
            return jacobi_preconditioner(A)
        if preconditioner == "gauss_seidel":
            raise NotImplementedError("Gauss-Seidel preconditioner not implemented.")
        raise ValueError(
            "String preconditioner must be 'none', 'jacobi', or 'gauss_seidel'."
        )

    if isinstance(preconditioner, LinearOperator):
        if getattr(preconditioner, "shape") != getattr(A, "shape"):
            raise ValueError("Preconditioner shape must match A.")
        return preconditioner

    if isinstance(preconditioner, np.ndarray):
        if preconditioner.shape != getattr(A, "shape"):
            raise ValueError("Preconditioner shape must match A.")
        return preconditioner

    if sparse.issparse(preconditioner):
        if getattr(preconditioner, "shape") != getattr(A, "shape"):
            raise ValueError("Preconditioner shape must match A.")
        return sparse.csr_matrix(preconditioner)

    if callable(preconditioner):
        return LinearOperator(
            shape=getattr(A, "shape"),
            matvec=preconditioner,
            dtype=getattr(A, "dtype", None),
        )

    raise TypeError(
        "preconditioner must be None, 'jacobi', a matrix, "
        "a LinearOperator, or a callable."
    )



#===============================================================================
# Solvers
#===============================================================================
def _direct_solve_validated(
    A: np.ndarray | sparse.spmatrix | sparse.sparray,
    b: np.ndarray,
) -> np.ndarray:
    """Solve a direct system after its dimensions have been validated."""
    if isinstance(A, np.ndarray):
        solution = dense_solve(A, b)
    else:
        solution = spsolve(sparse.csc_matrix(A), b)

    return np.asarray(solution)

def direct_solve(
    A: MatrixLike,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
) -> SolveResult:
    r"""Solve a linear system A x = b using the direct solver.

    Parameters
    ----------
    A
        The real or complex coefficient matrix A.
    b
        The compatible real or complex right-hand side b.
    x0
        Optional vector used to calculate the initial residual.
        It does not affect the direct-solver solution.

    Returns
    -------
    SolveResult
         The solution and diagnostic information.

    Raises
    ------
    TypeError
        If A is a LinearOperator or another unsupported matrix type.
    ValueError
        If A is not square or b is not a compatible one-dimensional vector.

    """

    if isinstance(A, LinearOperator):
        raise TypeError(
            "A LinearOperator cannot be used with a direct solver. "
            "Use an iterative solver instead."
        )

    b_vector, x0_vector = _validate_linear_system(A, b, x0)

    initial_vector = (
        np.zeros_like(b_vector)
        if x0_vector is None
        else x0_vector
    )
    initial_residual_norm = float(
        np.linalg.norm(b_vector - A @ initial_vector)
    )

    start_time = perf_counter()
    solution = _direct_solve_validated(A, b_vector)
    solve_time = perf_counter() - start_time

    solution = np.asarray(solution)
    final_residual_norm = float(
        np.linalg.norm(b_vector - A @ solution)
    )

    success = bool(np.all(np.isfinite(solution)))
    message = (
        "Direct solve completed."
        if success
        else "Direct solve produced nonfinite values."
    )

    return SolveResult(
        solution=solution,
        success=success,
        solver_name="direct",
        iterations=None,
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=final_residual_norm,
        residual_history=[],
        solve_time=solve_time,
        message=message,
    )

def cg_solve(
    A: MatrixLike,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: PreconditionerSpec = None,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    maximum_iterations: int | None = None,
    record_residual_history: bool = False,
) -> SolveResult:
    """Solve A x = b with conjugate gradients (SPD matrices).

    Parameters
    ----------
    A : MatrixLike
        Square operator or matrix. May be a LinearOperator.
    b : numpy.ndarray
        One-dimensional right-hand side.
    x0 : numpy.ndarray, optional
        Initial guess. Defaults to zeros.
    preconditioner : PreconditionerSpec, optional
        Approximation of A^{-1}. A dense or sparse matrix is applied directly
        to each residual; SciPy does not solve a system with it. The string
        "jacobi" constructs a diagonal approximate inverse.
    relative_tolerance, absolute_tolerance : float
        SciPy CG stopping tolerances.
    maximum_iterations : int, optional
        Iteration cap. SciPy default if None.
    record_residual_history : bool, optional
        Record the exact residual after every iteration. Disabled by default
        because it requires one additional matrix-vector product per iteration.

    Returns
    -------
    SolveResult
    """
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("Solver tolerances must be nonnegative.")
    if maximum_iterations is not None and maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive or None.")

    b_vector, x0_vector = _validate_linear_system(A, b, x0)
    dtype = _solver_dtype(A, b_vector)
    b_vector = np.asarray(b_vector, dtype=dtype)

    if x0_vector is None:
        x0_vector = np.zeros_like(b_vector)
    else:
        x0_vector = np.asarray(x0_vector, dtype=dtype)

    initial_residual_norm = float(
        np.linalg.norm(b_vector - A @ x0_vector)
    )

    residual_history: list[float] = []
    iteration_count = 0

    def callback(xk: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1
        if record_residual_history:
            residual = b_vector - A @ xk
            residual_history.append(float(np.linalg.norm(residual)))

    start_time = perf_counter()
    M = build_preconditioner(A, preconditioner)
    solution, info = cg(
        A,
        b_vector,
        x0=x0_vector,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxiter=maximum_iterations,
        M=M,
        callback=callback,
    )
    solve_time = perf_counter() - start_time

    solution = np.asarray(solution)
    final_residual_norm = float(
        np.linalg.norm(b_vector - A @ solution)
    )

    finite_solution = bool(np.all(np.isfinite(solution)))
    if info == 0 and finite_solution:
        success = True
        message = "CG converged."
    elif info == 0:
        success = False
        message = "CG produced nonfinite values."
    elif info > 0:
        success = False
        message = (
            f"CG did not converge within the iteration limit "
            f"(info={info})."
        )
    else:
        success = False
        message = f"CG failed with numerical breakdown (info={info})."

    return SolveResult(
        solution=solution,
        success=success,
        solver_name="cg",
        iterations=iteration_count,
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=final_residual_norm,
        residual_history=residual_history,
        solve_time=solve_time,
        message=message,
    )
def gmres_solve():
    ...

def iterative_solve():
    ...


#===============================================================================
# Linear System Solver
#===============================================================================
def solve_linear_system(
    system: LinearSystem,
    method: SolverMethod = "direct",
    preconditioner: PreconditionerSpec = None,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    maximum_iterations: int | None = None,
    gmres_restart: int | None = None,
    record_residual_history: bool = False,
) -> SolveResult:
    """
    Solve one linear system A x = b.

    Parameters
    ----------
    system
        Linear system containing A, b, and an optional x0.
    method
        One of "direct", "cg", or "gmres".
    preconditioner
        None, "jacobi", a LinearOperator, or a callable implementing
        r -> P^{-1} r.
    relative_tolerance
        Relative convergence tolerance for iterative methods.
    absolute_tolerance
        Absolute convergence tolerance for iterative methods.
    maximum_iterations
        Maximum number of iterative-solver iterations.
    gmres_restart
        Number of GMRES inner iterations before restarting.
    record_residual_history
        Whether to compute and retain exact residual norms after every
        iterative-solver step. This adds a matrix-vector product per step.
    """
    if method not in {"direct", "cg", "gmres"}:
        raise ValueError(
            "method must be 'direct', 'cg', or 'gmres'."
        )

    if method == "direct":
        if not (
            preconditioner is None
            or (
                isinstance(preconditioner, str)
                and preconditioner == "none"
            )
        ):
            warnings.warn(
                "The preconditioner is ignored by the direct solver.",
                UserWarning,
                stacklevel=2,
            )

        return direct_solve(
            system.A,
            system.b,
            x0=system.x0,
        )
        
    if method == "cg":
        return cg_solve(
            system.A,
            system.b,
            x0=system.x0,
            preconditioner=preconditioner,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_iterations=maximum_iterations,
            record_residual_history=record_residual_history,
        )

    if method == "gmres":
        raise NotImplementedError("GMRES solver not implemented.")

