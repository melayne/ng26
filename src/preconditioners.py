import numpy as np
import scipy.sparse as sp
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Gauss–Seidel (scipy CSR, free-DOF updates only)
# ---------------------------------------------------------------------------
def gauss_seidel_sweeps(
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    free_ids: np.ndarray,
    *,
    nsweeps: int = 5,
    omega: float = 1.0,
    verbose: bool = False,
    callback: Optional[Callable[[np.ndarray, int, float], None]] = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Apply forward Gauss-Seidel sweeps on the linear system Ax = b.
    Only the free degrees of freedom (DOFs) are updated; fixed DOFs
    (such as Dirichlet nodes) remain unchanged.

    Parameters
    ----------
    A : scipy.spmatrix
        Sparse (square) system matrix in any scipy sparse format. Will be
        converted to CSR internally.
    b : np.ndarray
        Right-hand side vector (ndarray of shape (ndof,)).
    x0 : np.ndarray
        Initial guess for the coefficient vector (ndarray of shape (ndof,)).
    free_ids : np.ndarray
        Indices of DOFs that should be updated (all others are fixed).
        Can be a 1D numpy array of integer indices.
    nsweeps : int, optional
        Number of Gauss-Seidel passes to apply (default: 5).
    omega : float, optional
        Relaxation parameter (1.0 = classical Gauss-Seidel, <1.0 under-relaxation).
    verbose : bool, optional
        If True, print free-DOF residual norm at every sweep.
    callback : callable(x, sweep_number, residual_norm), optional
        Called after each sweep with the current iterate ``x``, 1-based sweep
        number, and free-DOF residual norm. Useful for live plotting.

    Returns
    -------
    x : np.ndarray
        Updated coefficient vector after Gauss-Seidel iterations (ndarray, shape (ndof,)).
    history : list of float
        List of the free-DOF 2-norm of the residual (b - Ax) after each sweep.

    Notes
    -----
    The Gauss-Seidel iterative method for solving a linear system Ax = b proceeds 
    by iteratively updating each component of the solution vector x according to:

        x[i] := (1/ A[i, i]) * (b[i] - sum_{j != i} A[i, j] * x[j])

    for each degree of freedom (DOF) i, in sequence. In this implementation, only a subset 
    of "free" DOFs, as indicated by `free_ids`, are updated, while all other DOFs (such as 
    those associated with Dirichlet boundary conditions) remain fixed. The method supports
    a relaxation parameter `omega` (with `omega=1.0` recovering the classical Gauss-Seidel
    scheme, and `omega<1.0` giving under-relaxation).

    After each sweep, the 2-norm of the free-DOF residual (restricted to the variable indices) 
    is recorded and returned for monitoring convergence.
    """
    A = A.tocsr()
    x = np.asarray(x0, dtype=float).copy()
    b = np.asarray(b, dtype=float)
    diag = A.diagonal()

    if np.any(np.abs(diag[free_ids]) < 1e-14):
        raise ValueError("Zero or near-zero diagonal on a free DOF.")

    history: list[float] = []
    for sweep in range(nsweeps):
        for i in free_ids:
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]
            row_dot = vals @ x[cols]
            sigma_i = row_dot - diag[i] * x[i]
            x_gs = (b[i] - sigma_i) / diag[i]
            x[i] = (1.0 - omega) * x[i] + omega * x_gs

        residual = b - A @ x
        rnorm = float(np.linalg.norm(residual[free_ids]))
        history.append(rnorm)
        if verbose:
            print(f"sweep {sweep + 1:3d}, free residual norm = {rnorm:.3e}")
        if callback is not None:
            callback(x, sweep + 1, rnorm)

    return x, history


def smooth_custom_gs_residual(
    A, b, x, free_ids, *,
    nsweeps=5, omega=1.0, verbose=False
) -> tuple[np.ndarray, list[float]]:
    """
    Residual-correction smoother using a custom forward GS solve for Ae = r.
    Updates x in place and returns residual history on free DOFs.
    """
    A = A.tocsr()
    x = x.copy()
    diag = A.diagonal()

    if np.any(np.abs(diag[free_ids]) < 1e-14):
        raise ValueError("Zero or near-zero diagonal on a free DOF.")

    history = []

    for sweep in range(1, nsweeps + 1):
        # 1) residual r = b - A x
        r = b - A @ x

        # 2) approximately solve A e = r by ONE forward GS sweep on e
        e = np.zeros_like(x)
        for i in free_ids:
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]

            sigma = vals @ e[cols] - diag[i] * e[i]   # sum_{j != i} Aij*e_j
            e[i] = (r[i] - sigma) / diag[i]

        # 3) correction update x <- x + omega * e
        x[free_ids] += omega * e[free_ids]

        # 4) monitor free residual norm
        r_new = b - A @ x
        rnorm = float(np.linalg.norm(r_new[free_ids]))
        history.append(rnorm)

        if verbose:
            print(f"sweep {sweep:3d}  ||r_free||_2 = {rnorm:.6e}")

    return x, history