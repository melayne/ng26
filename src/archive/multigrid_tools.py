#%%

"""Reusable two-level multigrid tools for NGSolve finite element problems.

Notes
-----
Building bilinear and linear forms
    NGSolve forms are usually written directly::

        u, v = fes.TnT()
        a = BilinearForm(fes)
        a += InnerProduct(grad(u), grad(v)) * dx
        a.Assemble()
        f = LinearForm(fes)
        f.Assemble()

    For multigrid we want the same PDE on multiple meshes, with varying mesh sizes.
    The factory pattern lets you define the PDE once and reuse it per level.

    callback-based factory example::

        def bilinear_poisson(a, u, v):
            a += InnerProduct(grad(u), grad(v)) * dx

        def linear_rhs(f, u, v):
            f += 1.0 * v * dx

        setup = build_form_setup(bilinear=bilinear_poisson, linear=linear_rhs)
        fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
        a, f = setup(fes)
        a.Assemble(); f.Assemble()

Callback
    A callback is a function passed into another function.
    The receiving function calls the callback later when the right
    arguments exist.

    In ``build_form_setup``, ``bilinear`` and ``linear`` are callbacks.
    You write the ``a += ...`` and ``f += ...`` logic; ``build_form_setup``
    calls those callbacks inside ``setup(fes)`` after creating empty forms.

        def bilinear_fn(a, u, v):
            a += InnerProduct(grad(u), grad(v)) * dx

        def linear_fn(f, u, v):
            f += 1.0 * v * dx

        setup = build_form_setup(bilinear=bilinear_fn, linear=linear_fn)

End-to-end flow with ``setup``
    1. Define recipe once::

           setup = build_form_setup(bilinear=..., linear=...)

    2. Per mesh level::

           fes_c = H1(mesh_c, order=1, dirichlet="...")
           a_c, f_c = setup(fes_c)
           a_c.Assemble(); f_c.Assemble()

           fes_f = H1(mesh_f, order=1, dirichlet="...")
           a_f, f_f = setup(fes_f)
           a_f.Assemble(); f_f.Assemble()

    3. Build level objects with ``FELevel``::

           level_f = FELevel.from_forms(mesh_f, fes_f, a_f, f_f)
           level_c = FELevel.from_forms(mesh_c, fes_c, a_c, f_c)

    Multigrid logic (GS, ``P``, ``PT``, residuals) does not depend on how
    ``a`` and ``f`` were built, only that they are assembled.

When *not* to use the factory
    For a single two-level Poisson problem, writing forms inline can be
    simpler and easier to debug than callbacks and closures.
    Prefer writing forms inline unless you need the same PDE across multiple
    mesh levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Literal, TypedDict
import numpy as np
import scipy.sparse as sp
import time
import warnings

import ngsolve as ng
from ngsolve import BilinearForm, GridFunction, H1, InnerProduct, LinearForm, grad, dx

from NGSolve_utils import *
from preconditioners import gauss_seidel_sweeps

FormSetupFn = Callable[[object], tuple[BilinearForm, LinearForm]]


class LevelDataDict(TypedDict):
    """Plain-dict form of :class:`LevelData`, for interop with dict-based code."""
    level:     int
    mesh:      ng.Mesh
    fes:       object
    ndof:      int
    a:         BilinearForm
    f:         LinearForm
    A:         object
    gfu:       GridFunction
    free_ids:  np.ndarray
    fixed_ids: np.ndarray
    P:         Optional[object]
    PT:        Optional[object]


@dataclass
class LevelData:
    """Per-level data returned by ``build_level_data``."""
    level:     int
    mesh:      ng.Mesh
    fes:       object          # H1 FE space
    ndof:      int
    a:         BilinearForm
    f:         LinearForm
    A:         object          # a.mat — NGSolve sparse matrix
    gfu:       GridFunction
    free_ids:  np.ndarray
    fixed_ids: np.ndarray
    P:         Optional[object] = None  # prolongation: coarse -> fine
    PT:        Optional[object] = None  # restriction:  fine -> coarse

    def to_typeddict(self) -> LevelDataDict:
        """Return a plain ``dict`` view of this level (typed as :class:`LevelDataDict`)."""
        return LevelDataDict(
            level=self.level,
            mesh=self.mesh,
            fes=self.fes,
            ndof=self.ndof,
            a=self.a,
            f=self.f,
            A=self.A,
            gfu=self.gfu,
            free_ids=self.free_ids,
            fixed_ids=self.fixed_ids,
            P=self.P,
            PT=self.PT,
        )

# ---------------------------------------------------------------------------
# Linear + Bilinaer form builder
# ---------------------------------------------------------------------------
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
    >>> setup = build_form_setup(bilinear=bilinear_fn, linear=linear_fn)
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
# Single FE level
# ---------------------------------------------------------------------------
@dataclass
class FELevel:
    """One mesh level: spaces, forms, DOF layout, optional scipy export."""

    mesh: ng.Mesh
    fes: object
    a: BilinearForm
    f: LinearForm
    gfu: GridFunction # Current iterate (current estimate) of the solution u
    free_ids: np.ndarray
    fixed_ids: np.ndarray
    dirichlet_value: float | np.ndarray = field(default = 0.0)

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

    @property
    def A_csr(self) -> sp.csr_matrix:
        """Return current assembled stiffness matrix as scipy CSR."""
        return bilinear_form_to_csr(self.a)

    def b_np(self, *, inplace: bool = False) -> np.ndarray:
        """Return assembled load vector as copy (default) or mutable view."""
        arr = self.f.vec.FV().NumPy()
        return arr if inplace else arr.copy()

    def gfu_np(self) -> np.ndarray:
        return self.gfu.vec.FV().NumPy()

    def set_gfu_np(self, x: np.ndarray) -> None:
        self.gfu.vec.FV().NumPy()[:] = x

    def enforce_dirichlet(self, values: float | np.ndarray | None = None) -> None:
        """Set fixed DOF coefficients using class default unless overridden."""
        if values is None:
            values = self.dirichlet_value
        self.gfu_np()[self.fixed_ids] = values

    def set_initial_guess(self, cf, *, zero_boundary: bool = True) -> None:
        """Interpolate a coefficient function and optionally zero boundary DOFs."""
        self.gfu.Set(cf)
        if zero_boundary:
            self.enforce_dirichlet()

    def compute_residual_vector(self):
        """
        Return NGSolve vector r = f - A u (full algebraic residual).

        Note: r[fixed_ids] is generally **nonzero** — boundary rows are not
        the discrete PDE residual. Use ``residual_norm(norm="l2")`` for convergence.
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

    def residual_norm(self, *, norm="l2") -> float:
        """Norm of r on free DOFs. Supports ``'l2'`` and ``'A'``/``'energy'``."""
        r = self.compute_residual_vector()
        r.FV().NumPy()[self.fixed_ids] = 0.0
        r_np = r.FV().NumPy()
        if isinstance(norm, str):
            key = norm.lower()
            if key in ("l2", "euclidean"):
                return float(np.linalg.norm(r_np[self.free_ids]))
            if key in ("a", "energy"):
                Ar = self.a.mat.CreateColVector()
                Ar.data = self.a.mat * r
                return float(np.sqrt(float(np.dot(r_np, Ar.FV().NumPy()))))
            raise ValueError(f"Unknown norm {norm!r}; use 'l2' or 'A'.")
        Br = self.a.mat.CreateColVector()
        Br.data = norm * r
        return float(np.sqrt(float(np.dot(r_np, Br.FV().NumPy()))))

    def fixed_residual_norm(self) -> float:
        """Norm of r on Dirichlet DOFs (often nonzero even when BCs are correct)."""
        r_np = self.compute_residual_vector().FV().NumPy()
        return float(np.linalg.norm(r_np[self.fixed_ids]))

    def max_on_fixed(self) -> float:
        return float(np.max(np.abs(self.gfu_np()[self.fixed_ids])))

    def max_on_free(self) -> float:
        return float(np.max(np.abs(self.gfu_np()[self.free_ids])))

    def direct_solve(self) -> GridFunction:
        """NGSolve direct solve with Dirichlet BCs."""

        # Check SPD

        u = GridFunction(self.fes)
        u.vec.data = self.a.mat.Inverse(self.fes.FreeDofs()) * self.f.vec
        return u

    def smooth_GS(
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
            self.set_gfu_np(x_now)
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
            self.b_np(),
            self.gfu_np(),
            self.free_ids,
            nsweeps=nsweeps,
            omega=omega,
            verbose=verbose,
            callback=_on_sweep if plotting_enabled else None,
        )
        self.set_gfu_np(x)
        self.enforce_dirichlet()
        return hist

    def smooth_ngsolve(
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
        Apply NGSolve's native smoother in-place on ``gfu``.

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
        smoother = self.a.mat.CreateSmoother(self.fes.FreeDofs(), GS=True)
        rhs = self.f.vec.CreateVector()
        correction = self.f.vec.CreateVector()
        hist: list[float] = []

        for sweep in range(1, nsweeps + 1):
            rhs.data = self.f.vec - self.a.mat * self.gfu.vec
            correction.data = omega * (smoother * rhs)
            self.gfu.vec.data += correction
            self.enforce_dirichlet()

            rnorm = self.residual_norm(norm="l2")
            hist.append(rnorm)
            if verbose:
                print(f"sweep {sweep:3d}  ||r_free||_2 = {rnorm:.6e}")

            if not plotting_enabled or sweep % plot_every != 0:
                continue
            if plot_callback is not None:
                plot_callback(scene)
            elif hasattr(scene, "Redraw"):
                scene.Redraw()
                time.sleep(0.5)

        self.enforce_dirichlet()
        return hist

# ---------------------------------------------------------------------------
# Multilevel data builder
# ---------------------------------------------------------------------------
def build_multilevel_data(
    coarse_mesh: ng.Mesh,
    form_setup: FormSetupFn,
    *,
    n_refines: int,
    order: int = 1,
    dirichlet: str = "left|right|top|bottom",
    verbose: bool = False,
) -> list[LevelData]:
    """
    Build and return per-level finite element data for a multigrid hierarchy.

    Starting from ``coarse_mesh``, refines a working mesh ``n_refines`` times,
    assembling forms and snapshotting mesh geometry at each level.

    Parameters
    ----------
    coarse_mesh : ng.Mesh
        Starting geometry.  Not modified.
    form_setup : FormSetupFn
        Callable ``setup(fes) -> (a, f)`` that adds integrators without
        calling ``.Assemble()``.  Create one with ``build_form_setup``.
    n_refines : int
        Number of uniform refinements.  Produces ``n_refines + 1`` levels.
    order : int
        Polynomial order for the H1 space.
    dirichlet : str
        Boundary names to pin as Dirichlet conditions.
    verbose : bool
        Print a summary table while building.

    Returns
    -------
    list[LevelData]
        One dict per level, ordered coarse (index 0) to fine (index ``n_refines``).
        Each dict contains:

        ``level``     — integer level index
        ``mesh``      — ``ng.Mesh`` geometry snapshot (independent object)
        ``fes``       — H1 FE space built on ``mesh``
        ``ndof``      — number of DOFs at this level
        ``a``         — assembled :class:`BilinearForm`
        ``f``         — assembled :class:`LinearForm`
        ``A``         — ``a.mat``, the assembled stiffness matrix
        ``gfu``       — :class:`GridFunction` for the current solution iterate
        ``free_ids``  — indices of free (non-Dirichlet) DOFs
        ``fixed_ids`` — indices of Dirichlet DOFs
        ``P``         — prolongation operator, coarse → fine (``None`` on coarsest)
        ``PT``        — restriction operator, fine → coarse (``None`` on coarsest)

    Examples
    --------
    ::

        from netgen.geom2d import unit_square

        def poisson(a, u, v):
            a += InnerProduct(grad(u), grad(v)) * dx
        def rhs(f, u, v):
            f += 1.0 * v * dx

        coarse = ng.Mesh(unit_square.GenerateMesh(maxh=0.4))
        levels = build_level_data(
            coarse,
            build_form_setup(bilinear=poisson, linear=rhs),
            n_refines=3,
            verbose=True,
        )
    """
    if n_refines < 1:
        raise ValueError("n_refines must be >= 1.")

    nlevels = n_refines + 1
    working = ng.Mesh(coarse_mesh.ngmesh.Copy())
    level_data: list[LevelData] = []
    pending_P = None

    for lev in range(nlevels):
        snapshot = ng.Mesh(working.ngmesh.Copy())
        fes = H1(snapshot, order=order, dirichlet=dirichlet)

        a, f = form_setup(fes)
        a.Assemble()
        f.Assemble()

        free_ids, fixed_ids = get_free_fixed_ids(fes)

        level_data.append(LevelData(
            level=lev,
            mesh=snapshot,
            fes=fes,
            ndof=int(fes.ndof),
            a=a,
            f=f,
            A=a.mat,
            gfu=GridFunction(fes),
            free_ids=free_ids,
            fixed_ids=fixed_ids,
            P=pending_P,
            PT=pending_P.CreateTranspose() if pending_P is not None else None,
        ))

        if verbose:
            if lev == 0:
                print(f"  {'lev':>3}  {'ndof':>7}  {'A':>13}  {'P (coarse->fine)':>16}  {'PT (fine->coarse)':>17}  {'nfree':>7}  {'nfixed':>6}")
                print(f"  {'---':>3}  {'-------':>7}  {'-------------':>13}  {'----------------':>16}  {'-----------------':>17}  {'-------':>7}  {'------':>6}")
            a_shape  = f"{a.mat.height}x{a.mat.width}"
            p_shape  = f"{pending_P.height}x{pending_P.width}"  if pending_P is not None else "-"
            pt_shape = f"{pending_P.width}x{pending_P.height}"  if pending_P is not None else "-"
            tag = "  (coarse)" if lev == 0 else ("  (fine)" if lev == n_refines else "")
            print(f"  {lev:>3}  {fes.ndof:>7}  {a_shape:>13}  {p_shape:>16}  {pt_shape:>17}  {len(free_ids):>7}  {len(fixed_ids):>6}{tag}")

        if lev < n_refines:
            working.Refine()
            fes_next = H1(working, order=order, dirichlet=dirichlet)
            pending_P = fes_next.Prolongation().CreateMatrix(working.levels - 1)

    return level_data






SmootherKind = Literal["gs", "ngsolve"]
LevelAction = Literal["smooth", "direct"]


@dataclass
class LevelRefinementPolicy:
    """
    Per-level multigrid refinement policy.

    Notes
    -----
    - ``down_*`` applies before restriction (fine -> coarse).
    - ``up_*`` applies after prolongation correction (coarse -> fine).
    - ``coarse_action`` controls behavior on the coarsest level.
    """

    down_action: LevelAction = "smooth"
    up_action: LevelAction = "smooth"
    coarse_action: LevelAction = "direct"

    down_smoother: SmootherKind = "gs"
    up_smoother: SmootherKind = "gs"
    coarse_smoother: SmootherKind = "gs"

    down_sweeps: int = 2
    up_sweeps: int = 2
    coarse_sweeps: int = 4

    down_omega: float = 1.0
    up_omega: float = 1.0
    coarse_omega: float = 1.0


@dataclass
class MultilevelHierarchy:
    """
    Mesh hierarchy data for multigrid.

    Conventions
    -----------
    Levels are ordered coarse -> fine:
    ``level_data[0]`` is coarsest, ``level_data[-1]`` is finest.

    P and PT are indexed by the **fine** end of the operator they connect:

    - ``level_data[i].P``  maps level ``i-1`` -> ``i``   (``None`` on coarsest).
    - ``level_data[i].PT`` maps level ``i``   -> ``i-1`` (``None`` on coarsest).

    To restrict from level ``i``:   use ``level_data[i].PT``.
    To prolongate to level ``i``:   use ``level_data[i].P``.  No index offset needed.

    Parameters
    ----------
    level_data : list[LevelData]
        Output of :func:`build_level_data`.
    """

    level_data: list[LevelData]
    levels: list[FELevel] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.level_data) < 2:
            raise ValueError("Multilevel hierarchy needs at least 2 levels.")
        if self.level_data[0].P is not None or self.level_data[0].PT is not None:
            raise ValueError("Coarsest level (index 0) must have P = PT = None.")
        for i, d in enumerate(self.level_data[1:], start=1):
            if d.P is None:
                raise ValueError(f"Level {i} is missing P (prolongation).")
            if d.PT is None:
                raise ValueError(f"Level {i} is missing PT (restriction).")
                
        self.levels = [
            FELevel.from_forms(d.mesh, d.fes, d.a, d.f) for d in self.level_data
        ]

    @property
    def nlevels(self) -> int:
        return len(self.level_data)

    @property
    def coarsest_idx(self) -> int:
        return 0

    @property
    def finest_idx(self) -> int:
        return self.nlevels - 1


class MultilevelSolver:
    """
    Initial multilevel V-cycle implementation with policy-based smoothing/actions.

    Design goal: provide one operator-style entry point ``apply_preconditioner``
    that can later be used by either a custom CG loop or an NGSolve wrapper.
    """

    def __init__(
        self,
        hierarchy: MultilevelHierarchy,
        *,
        policies: Optional[list[LevelRefinementPolicy]] = None,
    ) -> None:
        self.hierarchy = hierarchy
        if policies is None:
            policies = [LevelRefinementPolicy() for _ in range(hierarchy.nlevels)]
        if len(policies) != hierarchy.nlevels:
            raise ValueError("Expected one LevelPolicy per hierarchy level.")
        self.policies = policies

    def _apply_smoother(
        self,
        level: FELevel,
        *,
        smoother: SmootherKind,
        nsweeps: int,
        omega: float,
        verbose: bool = False,
    ) -> None:
        if nsweeps <= 0:
            warnings.warn(f"nsweeps must be > 0, got {nsweeps}", stacklevel=2)
            return
        if smoother == "gs":
            level.smooth_GS(nsweeps=nsweeps, omega=omega, verbose=verbose)
        elif smoother == "ngsolve":
            level.smooth_ngsolve(nsweeps=nsweeps, omega=omega, verbose=verbose)
        
        raise ValueError(f"Unknown smoother kind: {smoother}")

    def _apply_action(self, 
        level_idx: int, 
        phase: Literal["down", "up", "coarse"], 
        *, 
        verbose: bool
    ) -> list[float]:

        level = self.hierarchy.levels[level_idx]
        policy = self.policies[level_idx]

        if phase == "down":
            action = policy.down_action
            smoother = policy.down_smoother
            nsweeps = policy.down_sweeps
            omega = policy.down_omega
        elif phase == "up":
            action = policy.up_action
            smoother = policy.up_smoother
            nsweeps = policy.up_sweeps
            omega = policy.up_omega
        else:
            action = policy.coarse_action
            smoother = policy.coarse_smoother
            nsweeps = policy.coarse_sweeps
            omega = policy.coarse_omega

        if action == "direct":
            u = level.direct_solve()
            level.gfu.vec.data = u.vec
            level.enforce_dirichlet()
            return
        if action == "smooth":
            self._apply_smoother(level, smoother=smoother, nsweeps=nsweeps, omega=omega, verbose=verbose)
            return
      
        raise ValueError(f"Unknown action: {action}")

    def _zero_level_iterate(self, level_idx: int) -> None:
        level = self.hierarchy.levels[level_idx]
        level.gfu_np()[:] = 0.0
        level.enforce_dirichlet()

    def _set_level_rhs(self, level_idx: int, rhs_vec) -> None:
        level = self.hierarchy.levels[level_idx]
        level.f.vec.data = rhs_vec

    def _restrict_to_coarser(self, fine_idx: int):
        """Restrict residual from level ``fine_idx`` to level ``fine_idx - 1``."""
        fine = self.hierarchy.levels[fine_idx]
        R = self.hierarchy.level_data[fine_idx].PT  # PT indexed by fine end
        r_f = fine.compute_residual_for_restriction()
        r_c = R.CreateColVector()
        r_c.data = R * r_f
        return r_c

    def _prolongate_correction(self, coarse_idx: int):
        """Prolongate coarse correction from level ``coarse_idx`` to level ``coarse_idx + 1``."""
        fine_idx = coarse_idx + 1
        coarse = self.hierarchy.levels[coarse_idx]
        P = self.hierarchy.level_data[fine_idx].P  # P indexed by fine end
        e_f = P.CreateColVector()
        e_f.data = P * coarse.gfu.vec
        return e_f

    def _v_cycle_recursive(self, level_idx: int, *, verbose: bool = False) -> None:
        if level_idx == self.hierarchy.coarsest_idx:
            self._apply_action(level_idx, "coarse", verbose=verbose)
            return

        self._apply_action(level_idx, "down", verbose=verbose)

        r_c = self._restrict_to_coarser(level_idx)
        coarse_idx = level_idx - 1
        self._set_level_rhs(coarse_idx, r_c)
        self._zero_level_iterate(coarse_idx)

        self._v_cycle_recursive(coarse_idx, verbose=verbose)

        e_f = self._prolongate_correction(coarse_idx)
        fine = self.hierarchy.levels[level_idx]
        fine.gfu.vec.data += e_f
        fine.enforce_dirichlet()

        self._apply_action(level_idx, "up", verbose=verbose)

    def v_cycle(self, *, finest_level_idx: Optional[int] = None, verbose: bool = False) -> float:
        """
        Run one V-cycle in-place and return finest free residual norm.
        """
        if finest_level_idx is None:
            finest_level_idx = self.hierarchy.finest_idx
        self._v_cycle_recursive(finest_level_idx, verbose=verbose)
        return self.hierarchy.levels[finest_level_idx].residual_norm(norm="l2")

    def apply_preconditioner(self, rhs_vec, *, finest_level_idx: Optional[int] = None, verbose: bool = False):
        """
        Apply one multigrid preconditioner action ``z = M^{-1} rhs``.

        This method is state-safe: it restores level ``f.vec`` and ``gfu.vec``
        after application and returns the correction vector ``z`` on the finest
        level. That makes it suitable as a backend for either custom CG loops or
        future NGSolve preconditioner wrappers.
        """
        if finest_level_idx is None:
            finest_level_idx = self.hierarchy.finest_idx

        saved_rhs = []
        saved_u = []
        for level in self.hierarchy.levels:
            rhs_store = level.f.vec.CreateVector()
            rhs_store.data = level.f.vec
            saved_rhs.append(rhs_store)

            u_store = level.gfu.vec.CreateVector()
            u_store.data = level.gfu.vec
            saved_u.append(u_store)

        try:
            for i, level in enumerate(self.hierarchy.levels):
                level.f.vec.FV().NumPy()[:] = 0.0
                level.gfu.vec.FV().NumPy()[:] = 0.0
                level.enforce_dirichlet()
                if i == finest_level_idx:
                    level.f.vec.data = rhs_vec

            self.v_cycle(finest_level_idx=finest_level_idx, verbose=verbose)

            finest = self.hierarchy.levels[finest_level_idx]
            z = finest.gfu.vec.CreateVector()
            z.data = finest.gfu.vec
            return z
        finally:
            for level, rhs_store, u_store in zip(self.hierarchy.levels, saved_rhs, saved_u):
                level.f.vec.data = rhs_store
                level.gfu.vec.data = u_store

