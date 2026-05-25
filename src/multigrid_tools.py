#%%

"""Reusable two-level multigrid tools for NGSolve finite element problems.

Notes
-----
Building bilinear and linear forms
    NGSolve forms are usually written directly in a notebook::

        u, v = fes.TnT()
        a = BilinearForm(fes)
        a += InnerProduct(grad(u), grad(v)) * dx
        a.Assemble()
        f = LinearForm(fes)
        f.Assemble()

    For multigrid you need the same PDE on a coarse mesh and a fine mesh.
    This module offers two styles:

    1. **Plain function (recommended for coursework)** — define something
       like ``assemble_poisson(fes)`` that returns ``(a, f)`` after
       ``Assemble()``, or use :func:`poisson_setup` which is the same idea.

    2. **Factory** — :func:`add_integrators` and :func:`poisson_setup` return
       a ``setup`` function (see below). Optional; useful if you want one
       recipe object passed into a driver.

Callback
    A **callback** is a function you pass in so **another function calls it
    later** when the right arguments exist.

    In :func:`add_integrators`, ``bilinear`` and ``linear`` are callbacks.
    You write the ``a += ...`` and ``f += ...`` logic; ``add_integrators``
    calls your functions inside ``setup(fes)`` after creating empty forms::

        def bilinear_fn(a, u, v):
            a += InnerProduct(grad(u), grad(v)) * dx

        setup = add_integrators(bilinear=bilinear_fn)

    There are **no input forms**: ``a`` and ``f`` start empty inside
    ``setup(fes)``. Your callback only adds integrators.

    The same idea appears in :func:`gauss_seidel_sweeps` with ``callback=``
    for plotting — GS calls your function each sweep.

Closure
    A **closure** is an inner function that **remembers variables from the
    outer function** after the outer function has returned.

    :func:`add_integrators` defines an inner ``setup(fes)`` that uses
    ``bilinear`` and ``linear`` from the outer call, then returns ``setup``::

        setup = add_integrators(bilinear=bilinear_fn)
        # add_integrators is finished, but setup still remembers bilinear_fn

        a, f = setup(fes_fine)
        a.Assemble(); f.Assemble()

    :func:`poisson_setup` is the same pattern with Poisson integrators
    hard-coded inside ``setup``.

End-to-end flow with ``setup``
    1. Define recipe once::

           setup = poisson_setup(rhs_cf=None)
           # or setup = add_integrators(bilinear=..., linear=...)

    2. Per mesh level::

           fes = H1(mesh, order=1, dirichlet="...")
           a, f = setup(fes)
           a.Assemble(); f.Assemble()

    3. Wrap in :class:`PoissonLevel` (pass ``mesh``, ``fes``, ``a``, ``f``)
       or use :meth:`PoissonLevel.assemble` for Poisson-only convenience.

    4. :class:`TwoLevelSetup` + :class:`TwoLevelSolver` for V-cycles.

    Multigrid logic (GS, ``P``, ``PT``, residuals) does not depend on how
    ``a`` and ``f`` were built — only that they are assembled.

When *not* to use the factory
    For a single two-level Poisson problem, writing forms inline or using
    a plain ``assemble_poisson(fes)`` helper is simpler and easier to debug
    than callbacks and closures. Prefer :func:`poisson_setup` over
    :func:`add_integrators` unless you need a custom PDE via callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import scipy.sparse as sp
import time
import ngsolve as ng
from ngsolve import BilinearForm, GridFunction, H1, InnerProduct, LinearForm, grad, dx

# Builds (BilinearForm, LinearForm) for a given FESpace — plug in Poisson, elasticity, etc.
FormSetupFn = Callable[[object], tuple[BilinearForm, LinearForm]]


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


# ---------------------------------------------------------------------------
# Problem-specific form builders (plug into FELevel.from_space)
# ---------------------------------------------------------------------------
def build_poisson_setup(rhs_cf=None) -> FormSetupFn:
    """
    Return a Poisson form-builder ``setup(fes)`` for ``-Δu = rhs``.

    The returned function creates an unassembled pair ``(a, f)`` with
    ``a += InnerProduct(grad(u), grad(v)) * dx`` and, when ``rhs_cf`` is
    provided, ``f += rhs_cf * v * dx``. If ``rhs_cf`` is ``None``, no RHS
    integrator is added (homogeneous right-hand side).

    Parameters
    ----------
    rhs_cf : CoefficientFunction, optional
        Source term coefficient used in ``f += rhs_cf * v * dx``.

    Examples
    --------
    >>> setup = poisson_setup()
    >>> a, f = setup(fes)
    >>> a.Assemble(); f.Assemble()
    """
    def bilinear_poisson(a, u, v):
        a += InnerProduct(grad(u), grad(v)) * dx

    def linear_poisson(f, u, v):
        if rhs_cf is not None:
            f += rhs_cf * v * dx

    return build_form_setup(bilinear=bilinear_poisson, linear=linear_poisson)


def build_form_setup(
    *,
    bilinear: Optional[Callable[[BilinearForm, object, object], None]] = None,
    linear: Optional[Callable[[LinearForm, object, object], None]] = None,
) -> FormSetupFn:
    """
    Generic form builder from callbacks that add integrators.

    Returns ``setup(fes) -> (a, f)`` (unassembled). See module docstring Notes
    for callbacks, closures, and when to use a plain function instead.

    Parameters
    ----------
    bilinear : callable(a, u, v), optional
        Called with an empty :class:`BilinearForm` and trial/test functions
        ``u, v = fes.TnT()``. Add integrators with ``a += ...``.
    linear : callable(f, u, v), optional
        Same for the :class:`LinearForm` RHS.

    Examples
    --------
    >>> def bilinear_fn(a, u, v):
    ...     a += InnerProduct(grad(u), grad(v)) * dx
    >>> def linear_fn(f, u, v):
    ...     f += 1 * v * dx
    >>> setup = add_integrators(bilinear=bilinear_fn, linear=linear_fn)
    >>> a, f = setup(fes)
    >>> a.Assemble(); f.Assemble()
    """

    def setup(fes) -> tuple[BilinearForm, LinearForm]:
        u, v = fes.TnT()
        a = BilinearForm(fes)
        if bilinear is not None:
            bilinear(a, u, v)
        f = LinearForm(fes)
        if linear is not None:
            linear(f, u, v)
        return a, f

    return setup


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


# ---------------------------------------------------------------------------
# Single FE level
# ---------------------------------------------------------------------------
@dataclass
class FELevel:
    """One mesh level: spaces, forms, DOF layout, optional scipy export."""

    mesh: ng.Mesh
    fes: object
    a: BilinearForm
    f: LinearForm
    gfu: GridFunction
    free_ids: np.ndarray
    fixed_ids: np.ndarray
    A_csr: sp.csr_matrix = field(repr=False)
    b_np: np.ndarray = field(repr=False)
    dirichlet_value: float | np.ndarray = field(default = 0.0)

    # dirichlet_bcs: str = field(default = "left|right|top|bottom")

    @classmethod
    def from_forms(
        cls,
        mesh,
        fes,
        a,
        f,
        *,
        gfu=None,
        dirichlet_value: float | np.ndarray = 0.0,
    ) -> "FELevel":
        if gfu is None:
            gfu = GridFunction(fes)
        free_ids, fixed_ids = get_free_fixed_ids(fes)
        return cls(
            mesh=mesh,
            fes=fes,
            a=a,
            f=f,
            gfu=gfu,
            free_ids=free_ids,
            fixed_ids=fixed_ids,
            dirichlet_value=dirichlet_value,
            A_csr=bilinear_form_to_csr(a),
            b_np=f.vec.FV().NumPy().copy(),
        )

    @classmethod
    def from_mesh(
        cls,
        mesh,
        form_setup: FormSetupFn,
        *,
        order: int = 1,
        dirichlet: str = "left|right|top|bottom",
        dirichlet_value: float | np.ndarray = 0.0,
    ) -> "FELevel":
        fes = H1(mesh, order=order, dirichlet=dirichlet)
        a, f = form_setup(fes)
        a.Assemble()
        f.Assemble()
        return cls.from_forms(
            mesh,
            fes,
            a,
            f,
            dirichlet_value=dirichlet_value,
        )

    @property
    def ndof(self) -> int:
        return self.fes.ndof

    def vec_np(self) -> np.ndarray:
        return self.gfu.vec.FV().NumPy()

    def set_vec_np(self, x: np.ndarray) -> None:
        self.gfu.vec.FV().NumPy()[:] = x

    def enforce_dirichlet(self, values: float | np.ndarray | None = None) -> None:
        """Set fixed DOF coefficients using class default unless overridden."""
        if values is None:
            values = self.dirichlet_value
        self.vec_np()[self.fixed_ids] = values

    def set_initial_guess(self, cf, *, zero_boundary: bool = True) -> None:
        """Interpolate a coefficient function and optionally zero boundary DOFs."""
        self.gfu.Set(cf)
        if zero_boundary:
            self.enforce_dirichlet()

    def compute_residual_vector(self):
        """
        Return NGSolve vector r = f - A u (full algebraic residual).

        Note: r[fixed_ids] is generally **nonzero** — boundary rows are not
        the discrete PDE residual. Use ``free_residual_norm()`` for convergence.
        """
        r = self.f.vec.CreateVector()
        r.data = self.f.vec - self.a.mat * self.gfu.vec
        return r

    def compute_residual_for_restriction(self):
        """
        Residual prepared for coarse-grid restriction.

        Zeros fixed-DOF entries so boundary artifacts do not pollute r_c = P^T r.
        This edit is for multigrid only, not a claim that r = 0 on the boundary.
        """
        r = self.compute_residual_vector()
        r.FV().NumPy()[self.fixed_ids] = 0.0
        return r

    def free_residual_norm(self) -> float:
        r_np = self.compute_residual_vector().FV().NumPy()
        return float(np.linalg.norm(r_np[self.free_ids]))

    def fixed_residual_norm(self) -> float:
        """Norm of r on Dirichlet DOFs (often nonzero even when BCs are correct)."""
        r_np = self.compute_residual_vector().FV().NumPy()
        return float(np.linalg.norm(r_np[self.fixed_ids]))

    def max_on_fixed(self) -> float:
        return float(np.max(np.abs(self.vec_np()[self.fixed_ids])))

    def max_on_free(self) -> float:
        return float(np.max(np.abs(self.vec_np()[self.free_ids])))

    def direct_solve(self) -> GridFunction:
        """NGSolve direct solve with Dirichlet BCs."""
        u = GridFunction(self.fes)
        u.vec.data = self.a.mat.Inverse(self.fes.FreeDofs()) * self.f.vec
        return u

    def smooth(
        self,
        *,
        nsweeps: int = 5,
        omega: float = 1.0,
        verbose: bool = False,
        scene=None,
        plot_every: int = None,
        plot_callback: Optional[Callable[[object], None]] = None,
    ) -> list[float]:
        """
        Apply GS sweeps in-place on ``gfu``.

        Parameters
        ----------
        scene : optional
            NGSolve Draw scene object to refresh during smoothing.
        plot_every : int, optional
            Redraw cadence in sweeps when ``scene`` is provided. If ``None``,
            plotting is disabled even when ``scene`` is passed.
        plot_callback : callable(scene), optional
            Custom redraw callback. If omitted and ``scene`` has ``Redraw()``,
            ``scene.Redraw()`` is called.
        """
        if plot_every is not None and plot_every <= 0:
            raise ValueError("plot_every must be >= 1")

        plotting_enabled = scene is not None and plot_every is not None

        def _on_sweep(x_now: np.ndarray, sweep_number: int, _rnorm: float) -> None:
            self.set_vec_np(x_now)
            self.enforce_dirichlet()
            if not plotting_enabled:
                return
            if sweep_number % plot_every != 0:
                return
            if plot_callback is not None:
                plot_callback(scene)
            elif hasattr(scene, "Redraw"):
                scene.Redraw()
                time.sleep(0.5)

        x, hist = gauss_seidel_sweeps(
            self.A_csr,
            self.b_np,
            self.vec_np(),
            self.free_ids,
            nsweeps=nsweeps,
            omega=omega,
            verbose=verbose,
            callback=_on_sweep if plotting_enabled else None,
        )
        self.set_vec_np(x)
        self.enforce_dirichlet()
        return hist


if __name__ == "__main__":
    import ngsolve as ng
    from ngsolve import BilinearForm, GridFunction, H1, InnerProduct, LinearForm, grad, dx, Mesh, x, y, sin, pi
    from ngsolve.webgui import Draw
    from netgen.geom2d import unit_square

    def poisson_bilinear_fn(a, u, v):
        """Add Poisson bilinear form in place"""
        a += InnerProduct(grad(u), grad(v)) * dx

    def homogeneous_linear_fn(f, u, v):
        f += 0 * v * dx


    poisson_setup = build_form_setup(bilinear=poisson_bilinear_fn, linear=homogeneous_linear_fn)
    
    mesh_c = Mesh(unit_square.GenerateMesh(maxh=0.04))
    mesh_f = Mesh(unit_square.GenerateMesh(maxh=0.04))
    mesh_f.Refine()

    fes_f = H1(mesh_f, order=1, dirichlet="left|right|top|bottom")
    fes_c = H1(mesh_c, order=1, dirichlet="left|right|top|bottom")
    a_f, f_f = poisson_setup(fes_f)
    a_c, f_c = poisson_setup(fes_c)
    a_f.Assemble()
    f_f.Assemble()
    a_c.Assemble()
    f_c.Assemble()

    poisson_f = FELevel.from_forms(mesh_f, fes_f, a_f, f_f)
    poisson_c = FELevel.from_forms(mesh_c, fes_c, a_c, f_c)

    x0_slow = sin(pi*x) * sin(pi*y)
    x0_fast = 0.1 * sin(30*pi*x/2) * sin(30*pi*y/2)
    x0_initial = x0_slow + x0_fast

    _scene = Draw(
        poisson_f.gfu,
        poisson_f.mesh,
        "Gauss-Seidel smoothing",
        deformation=True,
        radius=1.2,
        settings={
            "camera": {
                "transformations": [
                    {"type": "rotateX", "angle": -45},
                    # {"type": "rotateY", "angle": 30},
                    # {"type": "rotateZ", "angle": 10},
                ]
            },
            "Misc": {"line_thickness": 0.01}
        },
    )

    poisson_f.set_initial_guess(x0_initial)
    poisson_f.smooth(scene=_scene, plot_every=1)


# %%
    x0_grf = GridFunction(poisson_f.fes)
    x0_grf.Set(x0_initial)                  # interpolate CF onto FE space
    x0_grf.vec.FV().NumPy()[poisson_f.fixed_ids] = 0.0  # optional, enforce BC
    Draw(x0_grf, poisson_f.mesh, "Initial guess", deformation=True, radius=1.2)

    Draw(poisson_f.gfu, poisson_f.mesh, "Gauss-Seidel smoothing", deformation=True, radius=1.2)

# %%
    P, PT = get_prolongation_operators(poisson_f.fes)
    r_f = poisson_f.compute_residual_for_restriction()
    r_c = PT.CreateColVector()
    r_c.data = PT * r_f

    # Coarse-grid direct solve: A_c e_c = r_c (free DOFs only)
    e_c = r_c.CreateVector()
    e_c.data = poisson_c.a.mat.Inverse(poisson_c.fes.FreeDofs()) * r_c

    # Prolongate correction and update fine iterate
    e_f = P.CreateColVector()
    e_f.data = P * e_c
    poisson_f.gfu.vec.data += e_f
    poisson_f.enforce_dirichlet()

    Draw(poisson_f.gfu, poisson_f.mesh, "After coarse correction", deformation=True, radius=1.2)
# %%


### SCRATCH WORK


# ---------------------------------------------------------------------------
# Two-level problem builder + solver
# ---------------------------------------------------------------------------


# @dataclass
# class TwoLevelSetup:
#     """Fine/coarse Poisson levels plus prolongation / restriction operators."""

#     fine: PoissonLevel
#     coarse: PoissonLevel
#     P: object
#     PT: object
#     level: Optional[int] = None

#     @classmethod
#     def from_separate_meshes(
#         cls,
#         mesh_coarse: ng.Mesh,
#         mesh_fine: ng.Mesh,
#         *,
#         order: int = 1,
#         dirichlet: str = "left|right|top|bottom",
#         rhs_cf=None,
#     ) -> "TwoLevelSetup":
#         """
#         Build coarse and fine levels on separate meshes.

#         Prolongation comes from the *fine* mesh hierarchy (one refinement
#         above the coarse level of that hierarchy).
#         """
#         coarse = PoissonLevel.assemble(
#             mesh_coarse, order=order, dirichlet=dirichlet, rhs_cf=rhs_cf,
#         )
#         fine = PoissonLevel.assemble(
#             mesh_fine, order=order, dirichlet=dirichlet, rhs_cf=rhs_cf,
#         )
#         level = fine.fes.mesh.levels - 1
#         P, PT = get_prolongation_operators(fine.fes, level=level)
#         return cls(fine=fine, coarse=coarse, P=P, PT=PT, level=level)

#     @classmethod
#     def from_unit_square(
#         cls,
#         maxh: float = 0.04,
#         *,
#         order: int = 1,
#         dirichlet: str = "left|right|top|bottom",
#         rhs_cf=None,
#     ) -> "TwoLevelSetup":
#         """Coarse mesh + one uniform refinement for the fine mesh."""
#         from netgen.geom2d import unit_square

#         mesh_c = ng.Mesh(unit_square.GenerateMesh(maxh=maxh))
#         mesh_f = ng.Mesh(unit_square.GenerateMesh(maxh=maxh))
#         mesh_f.Refine()
#         return cls.from_separate_meshes(
#             mesh_c, mesh_f, order=order, dirichlet=dirichlet, rhs_cf=rhs_cf,
#         )

#     def validate(self, *, atol: float = 1e-10) -> None:
#         """Run standard consistency checks; raise ValueError on failure."""
#         f, c = self.fine, self.coarse
#         P_shape = (f.ndof, c.ndof)
#         PT_shape = (c.ndof, f.ndof)

#         P_mat = ng_matrix_to_csr(self.P)
#         if P_mat.shape != P_shape:
#             raise ValueError(f"P shape {P_mat.shape}, expected {P_shape}")

#         PT_mat = ng_matrix_to_csr(self.PT)
#         if PT_mat.shape != PT_shape:
#             raise ValueError(f"PT shape {PT_mat.shape}, expected {PT_shape}")

#         A_galerkin = ng_matrix_to_csr(self.PT @ f.a.mat @ self.P)
#         A_coarse = bilinear_form_to_csr(c.a)
#         if not np.allclose(A_coarse.toarray(), A_galerkin.toarray(), atol=atol):
#             raise ValueError("Galerkin check failed: PT @ A_f @ P != A_c")


# @dataclass
# class CycleStats:
#     """Diagnostics collected during one V-cycle."""

#     pre_smooth_history: list[float] = field(default_factory=list)
#     residual_before: float = 0.0
#     residual_after_correction: float = 0.0
#     post_smooth_history: list[float] = field(default_factory=list)
#     max_u_before: float = 0.0
#     max_u_after_correction: float = 0.0
#     max_u_after: float = 0.0


# class TwoLevelSolver:
#     """Two-level multigrid with custom GS smoothing and NGSolve coarse solve."""

#     def __init__(self, setup: TwoLevelSetup):
#         self.setup = setup
#         self._coarse_inv = setup.coarse.a.mat.Inverse(setup.coarse.fes.FreeDofs())

#     @property
#     def fine(self) -> PoissonLevel:
#         return self.setup.fine

#     @property
#     def coarse(self) -> PoissonLevel:
#         return self.setup.coarse

#     def restrict(self, r_fine):
#         """Restrict fine residual to coarse grid."""
#         r_c = self.setup.PT.CreateColVector()
#         r_c.data = self.setup.PT * r_fine
#         return r_c

#     def coarse_correction(self, r_c):
#         """Solve A_c e_c = r_c on free DOFs."""
#         e_c = r_c.CreateVector()
#         e_c.data = self._coarse_inv * r_c
#         return e_c

#     def prolongate(self, e_c):
#         """Prolongate coarse correction to fine grid."""
#         e_f = self.setup.P.CreateColVector()
#         e_f.data = self.setup.P * e_c
#         e_f.FV().NumPy()[self.fine.fixed_ids] = 0.0
#         return e_f

#     def apply_correction(self, e_f) -> None:
#         """Add prolongated correction and re-enforce BCs."""
#         self.fine.gfu.vec.data += e_f
#         self.fine.enforce_dirichlet(0.0)

#     def v_cycle(
#         self,
#         *,
#         n_pre: int = 5,
#         n_post: int = 3,
#         omega: float = 1.0,
#         verbose: bool = False,
#     ) -> CycleStats:
#         """One multiplicative two-level V-cycle in-place on ``fine.gfu``."""
#         stats = CycleStats()
#         stats.max_u_before = self.fine.max_on_free()
#         stats.residual_before = self.fine.free_residual_norm()

#         stats.pre_smooth_history = self.fine.smooth(
#             nsweeps=n_pre, omega=omega, verbose=verbose,
#         )

#         r_f = self.fine.compute_defect_for_restriction()
#         e_c = self.coarse_correction(self.restrict(r_f))
#         self.apply_correction(self.prolongate(e_c))

#         stats.residual_after_correction = self.fine.free_residual_norm()
#         stats.max_u_after_correction = self.fine.max_on_free()

#         stats.post_smooth_history = self.fine.smooth(
#             nsweeps=n_post, omega=omega, verbose=verbose,
#         )
#         stats.max_u_after = self.fine.max_on_free()
#         return stats

#     def solve(
#         self,
#         *,
#         max_cycles: int = 20,
#         tol: float = 1e-8,
#         n_pre: int = 5,
#         n_post: int = 3,
#         omega: float = 1.0,
#         verbose: bool = True,
#     ) -> tuple[list[CycleStats], list[float]]:
#         """
#         Repeat V-cycles until ``free ||r||_2 < tol`` or ``max_cycles`` reached.

#         Returns (cycle_stats, residual_history).
#         """
#         history: list[float] = []
#         all_stats: list[CycleStats] = []

#         for k in range(max_cycles):
#             stats = self.v_cycle(
#                 n_pre=n_pre, n_post=n_post, omega=omega, verbose=verbose,
#             )
#             all_stats.append(stats)
#             rnorm = self.fine.free_residual_norm()
#             history.append(rnorm)
#             if verbose:
#                 print(
#                     f"cycle {k + 1:2d}: ||r||={rnorm:.3e}, "
#                     f"max|u|={self.fine.max_on_free():.3e}"
#                 )
#             if rnorm < tol:
#                 break

#         return all_stats, history

#     def diagnostics(self) -> dict[str, float]:
#         """Print-friendly error / residual metrics vs direct solve."""
#         u = self.fine.vec_np()
#         r_np = self.fine.compute_residual_vector().FV().NumPy()
#         u_ref = self.fine.direct_solve().vec.FV().NumPy()

#         return {
#             "max_u_all": float(np.max(np.abs(u))),
#             "max_u_free": float(np.max(np.abs(u[self.fine.free_ids]))),
#             "max_u_fixed": float(np.max(np.abs(u[self.fine.fixed_ids]))),
#             "free_residual_norm": float(np.linalg.norm(r_np[self.fine.free_ids])),
#             "fixed_residual_norm": float(np.linalg.norm(r_np[self.fine.fixed_ids])),
#             "direct_max_u": float(np.max(np.abs(u_ref))),
#             "error_vs_direct": float(np.max(np.abs(u - u_ref))),
#         }

#     def print_diagnostics(self) -> None:
#         d = self.diagnostics()
#         print(f"max |u| (free DOFs):     {d['max_u_free']:.6e}")
#         print(f"free ||r||_2:            {d['free_residual_norm']:.6e}")
#         print(f"fixed ||r||_2:           {d['fixed_residual_norm']:.6e}  (may be nonzero)")
#         print(f"max |u| on fixed DOFs:   {d['max_u_fixed']:.6e}")
#         print(f"direct solve max |u|:    {d['direct_max_u']:.6e}")
#         print(f"error vs direct:         {d['error_vs_direct']:.6e}")
