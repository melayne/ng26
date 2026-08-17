from dataclasses import dataclass
from typing import TypeAlias
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import (
    LinearOperator,
    cg,
    gmres,
    spsolve,
)

MatrixLike: TypeAlias = (
    np.ndarray
    | sparse.spmatrix
    | sparse.sparray
    | LinearOperator
)

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


def solve_linear_system(
    system: LinearSystem,
    method: str = "direct",
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

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if b.ndim != 1:
        raise ValueError("b must be one-dimensional.")

    if b.shape[0] != A.shape[0]:
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
        if not sparse.issparse(A):
            raise TypeError(
                "The direct solver requires an explicit SciPy sparse matrix."
            )

        if preconditioner is not None:
            raise ValueError(
                "The direct solver does not use a preconditioner."
            )

        solution = spsolve(A, b)
        success = np.all(np.isfinite(solution))
        message = (
            "Direct solve completed."
            if success
            else "Direct solve produced nonfinite values."
        )
        iteration_count = None

    else:
        M = make_preconditioner(A, preconditioner)

        if method == "cg":

            def callback(xk):
                nonlocal iteration_count
                iteration_count += 1

                residual = b - A @ xk
                residual_history.append(np.linalg.norm(residual))

            solution, info = cg(
                A,
                b,
                x0=x0,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                maxiter=maximum_iterations,
                M=M,
                callback=callback,
            )

        else:

            def callback(relative_residual):
                nonlocal iteration_count
                iteration_count += 1
                residual_history.append(float(relative_residual))

            solution, info = gmres(
                A,
                b,
                x0=x0,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                maxiter=maximum_iterations,
                restart=gmres_restart,
                M=M,
                callback=callback,
                callback_type="pr_norm",
            )

        if info == 0:
            success = True
            message = "Iterative solver converged."
        elif info > 0:
            success = False
            message = (
                f"Solver did not converge within its iteration limit "
                f"(info={info})."
            )
        else:
            success = False
            message = f"Solver failed because of a numerical breakdown (info={info})."

    solve_time = perf_counter() - start_time

    final_residual = b - A @ solution
    final_residual_norm = np.linalg.norm(final_residual)

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