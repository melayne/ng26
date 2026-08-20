#===============================================================================
# Imports
#===============================================================================
from dataclasses import dataclass
from typing import TypeAlias, Literal, Callable
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import (
    LinearOperator,
    cg,
    gmres,
    spsolve,
)
import warnings

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

    if isinstance(A, LinearOperator):
        raise TypeError(
            "Jacobi needs an explicit matrix to read the diagonal."
        )
        
    if isinstance(A, np.ndarray):
        diagonal = np.diag(A).astype(float, copy=True)
    elif sparse.issparse(A):
        # Convert first: SciPy stubs omit `.diagonal` on the generic
        # spmatrix/sparray types after issparse().
        A = sparse.csr_matrix(A)
        diagonal = np.asarray(A.diagonal(), dtype=float)
    else:
        raise TypeError(f"Unsupported matrix type for Jacobi: {type(A)}")

    if np.any(diagonal == 0.0):
        raise ValueError("Jacobi requires a nonzero diagonal.")

    inverse_diagonal = 1.0 / diagonal
    n = diagonal.shape[0]
    return LinearOperator(
        shape=(n, n),
        matvec=lambda r: inverse_diagonal * np.asarray(r, dtype=float),
        dtype=float,
    )

def gauss_seidel_preconditioner():
    ...

def build_preconditioner(
    A: MatrixLike,
    preconditioner: PreconditionerSpec = None,
) -> MatrixLike | None:
    """Normalize a preconditioner spec for SciPy iterative solvers."""
    
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
            dtype=float,
        )

    raise ValueError(
        "preconditioner must be None, 'jacobi', a matrix, "
        "a LinearOperator, or a callable."
    )



#===============================================================================
# Solvers
#===============================================================================
def direct_solve(
    A: MatrixLike, 
    b: np.ndarray, 
) -> np.ndarray:
    r"""Solve a linear system A x = b using the direct solver.

    Parameters
    ----------
    A
        The coefficient matrix A, A \in \mathbb{R}^{n \times n}.
    b
        The right-hand side b, b \in \mathbb{R}^n.

    Returns
    -------
    solution
        The solution x, x \in \mathbb{R}^n.
    
    """

    if isinstance(A, LinearOperator):
        raise TypeError(
            "A LinearOperator cannot be used with a direct solver. "
            "Use an iterative solver instead."
        )

    if isinstance(A, np.ndarray):
        solution = dense_solve(A, b)

    elif sparse.issparse(A):
        A_solve = sparse.csc_matrix(A)
        solution = spsolve(A_solve, b)

    else:
        raise TypeError(f"Unsupported matrix type: {type(A)}")

    return np.asarray(solution).reshape(-1)

def cg_solve(
    A: MatrixLike,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: MatrixLike | None = None,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    maximum_iterations: int | None = None,
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
    preconditioner : MatrixLike, optional
        Left preconditioner for SciPy CG. May be a dense/sparse matrix
        (SciPy solves M z = r each step) or a LinearOperator whose
        matvec applies M^{-1}.
    relative_tolerance, absolute_tolerance : float
        SciPy CG stopping tolerances.
    maximum_iterations : int, optional
        Iteration cap. SciPy default if None.

    Returns
    -------
    SolveResult
    """
    b = np.asarray(b, dtype=float)
    if b.ndim != 1:
        raise ValueError("b must be one-dimensional.")

    n_rows, n_cols = getattr(A, "shape")
    if n_rows != n_cols:
        raise ValueError("A must be square.")
    if b.shape[0] != n_rows:
        raise ValueError("The dimensions of A and b do not agree.")

    if preconditioner is not None:
        m_rows, m_cols = getattr(preconditioner, "shape")
        if (m_rows, m_cols) != (n_rows, n_cols):
            raise ValueError(
                "The preconditioner must have the same shape as A."
            )

    if x0 is None:
        x0_vec = np.zeros_like(b)
    else:
        x0_vec = np.asarray(x0, dtype=float)
        if x0_vec.shape != b.shape:
            raise ValueError("x0 must have the same shape as b.")

    initial_residual_norm = float(np.linalg.norm(b - A @ x0_vec))

    residual_history: list[float] = []
    iteration_count = 0

    def callback(xk: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1
        residual_history.append(float(np.linalg.norm(b - A @ xk)))

    start_time = perf_counter()
    solution, info = cg(
        A,
        b,
        x0=x0_vec,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxiter=maximum_iterations,
        M=preconditioner,
        callback=callback,
    )
    solve_time = perf_counter() - start_time

    solution = np.asarray(solution, dtype=float).reshape(-1)
    final_residual_norm = float(np.linalg.norm(b - A @ solution))

    if info == 0:
        success = True
        message = "CG converged."
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
    preconditioner=None,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    maximum_iterations: int | None = None,
    gmres_restart: int | None = None,
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
    """
    A = system.A
    b = np.asarray(system.b)
    x0 = None if system.x0 is None else np.asarray(system.x0)

    #Note: SciPy's sparray stubs omit `.shape`; getattr keeps runtime behavior.
    n_rows, n_cols = getattr(A, "shape")
    
    if n_rows != n_cols:
        raise ValueError("A must be square.")

    if b.ndim != 1:
        raise ValueError("b must be one-dimensional.")

    if b.shape[0] != n_rows:
        raise ValueError("The dimensions of A and b do not agree.")

    if x0 is not None and x0.shape != b.shape:
        raise ValueError("x0 must have the same shape as b.")

    if method not in {"direct", "cg", "gmres"}:
        raise ValueError(
            "method must be 'direct', 'cg', or 'gmres'."
        )

    initial_vector = np.zeros_like(b) if x0 is None else x0
    initial_residual = b - A @ initial_vector
    initial_residual_norm = np.linalg.norm(initial_residual)

    residual_history = []
    iteration_count = 0

    start_time = perf_counter()

    if method == "direct":
        if preconditioner is not None:
            warnings.warn(
                "The preconditioner is ignored by the direct solver.",
                UserWarning,
                stacklevel=2,
            )
            
        solution = direct_solve(A, b)
        success = bool(np.all(np.isfinite(solution)))
        message = (
            "Direct solve completed."
            if success
            else "Direct solve produced nonfinite values."
        )
        iteration_count = None
    
    elif method == "cg":
        M = build_preconditioner(A, preconditioner)
        return cg_solve(
            A,
            b,
            x0=x0,
            preconditioner=M,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_iterations=maximum_iterations,
        )

    elif method == "gmres":
        raise NotImplementedError("GMRES solver not implemented.")

    # else:
    #     M = make_preconditioner(A, preconditioner)

    #     if method == "cg":

    #         def callback(xk):
    #             nonlocal iteration_count
    #             iteration_count += 1

    #             residual = b - A @ xk
    #             residual_history.append(np.linalg.norm(residual))

    #         solution, info = cg(
    #             A,
    #             b,
    #             x0=x0,
    #             rtol=relative_tolerance,
    #             atol=absolute_tolerance,
    #             maxiter=maximum_iterations,
    #             M=M,
    #             callback=callback,
    #         )

    #     else:

    #         def callback(relative_residual):
    #             nonlocal iteration_count
    #             iteration_count += 1
    #             residual_history.append(float(relative_residual))

    #         solution, info = gmres(
    #             A,
    #             b,
    #             x0=x0,
    #             rtol=relative_tolerance,
    #             atol=absolute_tolerance,
    #             maxiter=maximum_iterations,
    #             restart=gmres_restart,
    #             M=M,
    #             callback=callback,
    #             callback_type="pr_norm",
    #         )

    #     if info == 0:
    #         success = True
    #         message = "Iterative solver converged."
    #     elif info > 0:
    #         success = False
    #         message = (
    #             f"Solver did not converge within its iteration limit "
    #             f"(info={info})."
    #         )
    #     else:
    #         success = False
    #         message = f"Solver failed because of a numerical breakdown (info={info})."

    solve_time = perf_counter() - start_time

    # SciPy sparse/LinearOperator stubs omit mature ``__matmul__`` typing.
    final_residual = b - (A @ solution)  # type: ignore[operator]
    final_residual_norm = float(np.linalg.norm(final_residual))

    return SolveResult(
        solution=np.asarray(solution),
        success=success,
        solver_name=method,
        iterations=iteration_count,
        initial_residual_norm=float(initial_residual_norm),
        final_residual_norm=float(final_residual_norm),
        residual_history=residual_history,
        solve_time=solve_time,
        message=message,
    )


# def jacobi_preconditioner(A: sparse.spmatrix) -> LinearOperator:
#     """
#     Construct the Jacobi approximate-inverse operator

#         r -> D^{-1} r,

#     where D is the diagonal of A.
#     """
#     if not sparse.issparse(A):
#         raise TypeError(
#             "Jacobi preconditioning requires an explicit SciPy sparse matrix."
#         )

#     diagonal = np.asarray(A.diagonal())

#     if np.any(diagonal == 0):
#         raise ValueError(
#             "Jacobi preconditioning cannot be used when A has a zero diagonal entry."
#         )

#     inverse_diagonal = 1.0 / diagonal

#     return LinearOperator(
#         shape=A.shape,
#         matvec=lambda r: inverse_diagonal * r,
#         dtype=A.dtype,
#     )


# def make_preconditioner(
#     A: sparse.spmatrix | LinearOperator,
#     preconditioner: (
#         None
#         | str
#         | LinearOperator
#         | Callable[[np.ndarray], np.ndarray]
#     ),
# ) -> LinearOperator | None:
#     """Convert a preconditioner specification to a LinearOperator."""

#     if preconditioner is None or preconditioner == "none":
#         return None

#     if preconditioner == "jacobi":
#         return jacobi_preconditioner(A)

#     if isinstance(preconditioner, LinearOperator):
#         if preconditioner.shape != A.shape:
#             raise ValueError(
#                 "The preconditioner must have the same shape as A."
#             )
#         return preconditioner

#     if callable(preconditioner):
#         return LinearOperator(
#             shape=A.shape,
#             matvec=preconditioner,
#             dtype=A.dtype,
#         )

#     raise ValueError(
#         "preconditioner must be None, 'jacobi', "
#         "a LinearOperator, or a callable."
#     )


# def solve_linear_system(
#     system: LinearSystem,
#     method: str = "direct",
#     preconditioner=None,
#     relative_tolerance: float = 1e-8,
#     absolute_tolerance: float = 0.0,
#     maximum_iterations: int | None = None,
#     gmres_restart: int | None = None,
# ) -> SolveResult:
#     """
#     Solve one linear system A x = b.

#     Parameters
#     ----------
#     system
#         Linear system containing A, b, and an optional x0.
#     method
#         One of "direct", "cg", or "gmres".
#     preconditioner
#         None, "jacobi", a LinearOperator, or a callable implementing
#         r -> P^{-1} r.
#     relative_tolerance
#         Relative convergence tolerance for iterative methods.
#     absolute_tolerance
#         Absolute convergence tolerance for iterative methods.
#     maximum_iterations
#         Maximum number of iterative-solver iterations.
#     gmres_restart
#         Number of GMRES inner iterations before restarting.
#     """
#     A = system.A
#     b = np.asarray(system.b)
#     x0 = None if system.x0 is None else np.asarray(system.x0)

#     if A.shape[0] != A.shape[1]:
#         raise ValueError("A must be square.")

#     if b.ndim != 1:
#         raise ValueError("b must be one-dimensional.")

#     if b.shape[0] != A.shape[0]:
#         raise ValueError("The dimensions of A and b do not agree.")

#     if x0 is not None and x0.shape != b.shape:
#         raise ValueError("x0 must have the same shape as b.")

#     if method not in {"direct", "cg", "gmres"}:
#         raise ValueError(
#             "method must be 'direct', 'cg', or 'gmres'."
#         )

#     initial_vector = np.zeros_like(b) if x0 is None else x0
#     initial_residual = b - A @ initial_vector
#     initial_residual_norm = np.linalg.norm(initial_residual)

#     residual_history = []
#     iteration_count = 0

#     start_time = perf_counter()

#     if method == "direct":
#         if not sparse.issparse(A):
#             raise TypeError(
#                 "The direct solver requires an explicit SciPy sparse matrix."
#             )

#         if preconditioner is not None:
#             raise ValueError(
#                 "The direct solver does not use a preconditioner."
#             )

#         solution = spsolve(A, b)
#         success = np.all(np.isfinite(solution))
#         message = (
#             "Direct solve completed."
#             if success
#             else "Direct solve produced nonfinite values."
#         )
#         iteration_count = None

#     else:
#         M = make_preconditioner(A, preconditioner)

#         if method == "cg":

#             def callback(xk):
#                 nonlocal iteration_count
#                 iteration_count += 1

#                 residual = b - A @ xk
#                 residual_history.append(np.linalg.norm(residual))

#             solution, info = cg(
#                 A,
#                 b,
#                 x0=x0,
#                 rtol=relative_tolerance,
#                 atol=absolute_tolerance,
#                 maxiter=maximum_iterations,
#                 M=M,
#                 callback=callback,
#             )

#         else:

#             def callback(relative_residual):
#                 nonlocal iteration_count
#                 iteration_count += 1
#                 residual_history.append(float(relative_residual))

#             solution, info = gmres(
#                 A,
#                 b,
#                 x0=x0,
#                 rtol=relative_tolerance,
#                 atol=absolute_tolerance,
#                 maxiter=maximum_iterations,
#                 restart=gmres_restart,
#                 M=M,
#                 callback=callback,
#                 callback_type="pr_norm",
#             )

#         if info == 0:
#             success = True
#             message = "Iterative solver converged."
#         elif info > 0:
#             success = False
#             message = (
#                 f"Solver did not converge within its iteration limit "
#                 f"(info={info})."
#             )
#         else:
#             success = False
#             message = f"Solver failed because of a numerical breakdown (info={info})."

#     solve_time = perf_counter() - start_time

#     final_residual = b - A @ solution
#     final_residual_norm = np.linalg.norm(final_residual)

#     return SolveResult(
#         solution=np.asarray(solution),
#         success=success,
#         solver_name=method,
#         iterations=iteration_count,
#         initial_residual_norm=float(initial_residual_norm),
#         final_residual_norm=float(final_residual_norm),
#         residual_history=residual_history,
#         solve_time=solve_time,
#         message=message,
#     )
