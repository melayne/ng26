import numpy as np
import scipy.sparse as sp
from typing import Optional
from ngsolve import BilinearForm, LinearForm, GridFunction, InnerProduct, grad, dx


# ---------------------------------------------------------------------------
# Matrix / DOF utilities
# ---------------------------------------------------------------------------
def get_free_fixed_ids(fes) -> tuple[np.ndarray, np.ndarray]:
    """
    Get free and fixed DOF IDs for an NGSolve FESpace.

    Parameters
    ----------
    fes : ngsolve.FESpace
        The finite element space to get free and fixed DOF IDs for.

    Returns
    -------
    free_ids : np.ndarray
        Array of indices of free DOFs.
    fixed_ids : np.ndarray
        Array of indices of fixed DOFs.
    """
    free_dofs = fes.FreeDofs()
    free = np.array([bool(free_dofs[i]) for i in range(fes.ndof)], dtype=bool)
    free_ids = np.flatnonzero(free)
    fixed_ids = np.flatnonzero(~free)
    return free_ids, fixed_ids


def boundary_dof_ids(fes, name: str) -> np.ndarray:
    """Return indices of DOFs lying on the named boundary region.

    Parameters
    ----------
    fes : ngsolve.FESpace
        The finite element space.
    name : str
        Boundary name or regex, same syntax as the ``dirichlet=`` argument and
        ``mesh.Boundaries`` (e.g. ``"left"`` or ``"left|top"``).

    Returns
    -------
    np.ndarray
        Integer indices of the DOFs associated with that boundary. This is
        purely geometric membership; it does not depend on whether those DOFs
        are constrained (Dirichlet) or free.

    Notes
    -----
    A DOF on a shared corner belongs to every boundary it touches, so it will
    appear in the result for each of those boundary names.
    """
    ba = fes.GetDofs(fes.mesh.Boundaries(name))
    return np.flatnonzero([bool(ba[i]) for i in range(len(ba))])


def apply_dirichlet(vec, fixed_ids, values: "float | np.ndarray" = 0.0):
    """Write Dirichlet ``values`` into the ``fixed_ids`` entries of ``vec``.

    Generic, stateless helper: it does not know about any FE level, it only
    pins the given indices of a vector to the given values. Class methods such
    as ``Level.enforce_dirichlet`` wrap this with their own ``fixed_ids`` and
    default value.

    Parameters
    ----------
    vec : ngsolve.BaseVector or np.ndarray
        Vector to modify in place. NGSolve vectors are accessed via
        ``vec.FV().NumPy()``; anything else is treated as a numpy array.
    fixed_ids : np.ndarray
        Indices of the (Dirichlet/fixed) DOFs to overwrite.
    values : float or np.ndarray, optional
        Either a scalar (broadcast to every fixed DOF) or a 1-D array of shape
        ``(len(fixed_ids),)`` giving one value per fixed DOF, in the same order
        as ``fixed_ids``. Defaults to ``0.0`` (homogeneous). A full-length
        ``(ndof,)`` array is *not* accepted; slice it yourself with
        ``values[fixed_ids]``.

    Returns
    -------
    The same ``vec`` object (for chaining), modified in place.

    Raises
    ------
    ValueError
        If ``values`` is an array whose shape is neither scalar nor
        ``(len(fixed_ids),)``.
    """
    arr = vec.FV().NumPy() if hasattr(vec, "FV") else np.asarray(vec)
    values = np.asarray(values, dtype=float)
    n_fixed = len(fixed_ids)
    if values.ndim != 0 and values.shape != (n_fixed,):
        raise ValueError(
            f"values must be a scalar or a 1-D array of shape ({n_fixed},) "
            f"matching fixed_ids; got shape {values.shape}. "
            f"If you have a full-length (ndof,) array, pass values[fixed_ids]."
        )
    arr[fixed_ids] = values
    return vec


def vector_norm(vec, mat=None, *, free_ids: Optional[np.ndarray] = None) -> float:
    """Norm of an NGSolve vector, optionally weighted by a matrix.

    Generic, stateless helper that knows nothing about FE levels: it only
    needs a vector and (optionally) a metric matrix.

    Parameters
    ----------
    vec : ngsolve.BaseVector
        The vector to measure.
    mat : ngsolve matrix / operator, optional
        Metric ``B``. If given, returns ``sqrt(vec^T B vec)``. With the
        stiffness matrix ``A``, that is the discrete ``A``-seminorm ``||v||_A``
        (for an error vector ``e`` this is the usual energy error norm). It is
        **not** ``sqrt(v^T A^{-1} v)``. For a constrained problem, ``vec``
        should be zero on fixed DOFs. If ``mat`` is ``None``, the Euclidean
        norm on ``free_ids`` (if given) or the full vector is returned.
    free_ids : np.ndarray, optional
        Only used for the Euclidean case (``mat is None``); restricts the norm to
        these entries. Ignored when ``mat`` is given.

    Returns
    -------
    float
        ``sqrt(vec^T B vec)`` if ``mat`` is given, else the Euclidean norm.

    Notes
    -----
    For an SPD metric the quadratic form is non-negative; tiny negative values
    from round-off are clamped to zero before the square root.
    """
    if mat is None:
        arr = vec.FV().NumPy() if hasattr(vec, "FV") else np.asarray(vec)
        if free_ids is not None:
            arr = arr[free_ids]
        return float(np.linalg.norm(arr))

    Bv = vec.CreateVector()
    Bv.data = mat * vec
    quad = float(InnerProduct(vec, Bv))
    return float(np.sqrt(max(quad, 0.0)))


def ng_matrix_to_csr(ng_mat) -> sp.csr_matrix:
    """Convert an NGSolve BaseMatrix COO representation to scipy CSR."""
    rows, cols, vals = ng_mat.COO()
    height, width = ng_mat.shape
    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(height, width),
    ).tocsr()


def bilinear_form_to_csr(bf: BilinearForm) -> sp.csr_matrix:
    """Export assembled BilinearForm.mat to scipy CSR."""
    return ng_matrix_to_csr(bf.mat)


def get_prolongation_operators(fes, level: Optional[int] = None):
    """
    Get prolongation operators, P and PT,for a given finite element 
    space. The space must have multiple levels.

    Parameters
    ----------
    fes : ngsolve.FESpace
        The finite element space to get prolongation operators for.
    level : int, optional
        The level of the mesh to get prolongation operators for. If 
        None, the highest level is used.

    Returns
    -------
    Return (P, PT) NGSolve operators for coarse -> fine and fine -> coarse.

    Notes
    -----
    P  : ndof_fine x ndof_coarse
    PT : ndof_coarse x ndof_fine
    """
    if level is None:
        level = fes.mesh.levels - 1
    P = fes.Prolongation().CreateMatrix(level)
    PT = P.CreateTranspose()
    return P, PT

