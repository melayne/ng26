"""Geometric multigrid V-cycle for NGSolve finite element problems.

Notes
-----
Hierarchy: one ``Level`` per mesh (coarse to fine). Each level has assembled
``a``, ``f``, optional transfers ``P`` / ``PT``, and cached CSR / smoother data.

Vectors: level routines solve ``A x = b`` with ``level.a.mat``. ``b`` is never
modified by smoothing or the V-cycle. ``x`` is updated in place on free DOFs
only; Dirichlet entries are left unchanged.

``MultigridSolver.solve`` uses the finest ``f.vec`` as ``b`` and ``gfu.vec`` as
``x``. Recursive coarse calls use restricted residual ``r_c`` as ``b`` and a
zero-initialized correction as ``x``; ``f.vec`` on any level is not overwritten.

Restriction ``r_c = PT * r`` and prolongation ``x += P * e_c`` implement the
standard correction scheme.

Diagnostics (``MultigridSolver.solve`` / ``record_norms``): three norm families —
L2 residual ``||r||_2``, energy error ``||u_exact - x||_A``, and update-dual
``sqrt(r_before^T dx)``. See the comment above ``NormKind``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, overload

import numpy as np
import scipy.sparse as sp

import ngsolve as ng
from ngsolve import BaseMatrix, BilinearForm, FESpace, GridFunction, H1, InnerProduct, LinearForm

try:
    from .NGSolve_utils import (
        apply_dirichlet,
        bilinear_form_to_csr,
        boundary_dof_ids,
        get_free_fixed_ids,
        vector_norm as _vector_norm,
    )
    from .preconditioners import gauss_seidel_sweeps
except ImportError:
    from NGSolve_utils import (
        apply_dirichlet,
        bilinear_form_to_csr,
        boundary_dof_ids,
        get_free_fixed_ids,
        vector_norm as _vector_norm,
    )
    from preconditioners import gauss_seidel_sweeps


FormSetupFn = Callable[[FESpace], "tuple[BilinearForm, LinearForm]"]
DirichletValue = float | np.ndarray | ng.CoefficientFunction | dict
SmootherKind = Literal["gs", "native"]
# Norm names fall into three families (do not mix them up):
#
# 1) Residual L2 (stopping / algebraic):  "l2"  ->  ||r||_2
#    Use in solve(..., stop_norm="l2").  Level.residual_norm().
#
# 2) Energy ERROR (needs u_exact):  "energy", "A" in record_norms / error_norm
#    ->  ||e||_A = sqrt(e^T A e),  e = u_exact - x.
#    NOT a residual norm.  Requires level._gfu_exact (see solve() guard).
#
# 3) Dual / preconditioned (needs r_before, dx):  "update_dual", etc.
#    ->  sqrt(r_before^T dx)  ~  ||r||_{A^{-1}}  when dx ~ A^{-1} r_before.
#
# Residuals are measured in L2 only (Level.residual_norm, solve stop_norm).
# Energy norm sqrt(e^T A e) is only for the error e via error_norm / record_norms.
NormKind = Literal[
    "l2",
    "euclidean",
    "2",
    "A",
    "energy",
    "M",
    "mass",
    "update_dual",
    "dual",
    "mg-dual",
    "dual-residual",
    "approx-energy-error",
    "preconditioned",
]

_DUAL_NORM_KEYS = {
    "update_dual",
    "dual",
    "mg-dual",
    "dual-residual",
    "approx-energy-error",
    "preconditioned",
}

# Only for ||u_exact - x||_A in record_norms / error_norm — not for residuals.
_ENERGY_ERROR_NORM_KEYS = {
    "energy",
    "A",
}

_L2_NORM_KEYS = {
    "l2",
    "euclidean",
    "2",
}

_MASS_NORM_KEYS = {
    "M",
    "mass",
}
# ---------------------------------------------------------------------------
# Form factory
# ---------------------------------------------------------------------------
def build_form_setup(
    *,
    bilinear: Optional[Callable[[BilinearForm, object, object], None]] = None,
    linear: Optional[Callable[[LinearForm, object, object], None]] = None,
) -> FormSetupFn:
    """Build a form factory from bilinear and linear integrator callbacks.

    Parameters
    ----------
    bilinear : callable(a, u, v), optional
        Receives an empty BilinearForm and trial/test functions ``u, v = fes.TnT()``.
    linear : callable(f, u, v), optional
        Receives an empty LinearForm and the same ``u, v``.

    Returns
    -------
    setup
        Callable ``setup(fes) -> (a, f)`` (unassembled).

    Notes
    -----
    Assemble both forms before passing them to ``Level.from_forms`` or
    ``build_hierarchy``.
    """

    def setup(fes) -> "tuple[BilinearForm, LinearForm]":
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
# A single finite element level
# ---------------------------------------------------------------------------
@dataclass
class Level:
    """One mesh level: FE space, assembled forms, transfers, and caches.

    Attributes
    ----------
    mesh
        NGSolve mesh for this level.
    fes
        Finite element space on ``mesh``.
    a
        Assembled bilinear form; ``a.mat`` is the level operator.
    f
        Assembled linear form; ``f.vec`` is the load on this mesh.
    gfu
        Solution GridFunction; ``gfu.vec`` is the finest-level iterate in ``solve``.
    free_ids, fixed_ids
        Indices of free and Dirichlet DOFs in global ordering.
    P
        Prolongation from the immediately coarser level into this level; ``None`` on coarsest.
    PT
        Restriction from this level to the immediately coarser level; ``None`` on coarsest.
    dirichlet_value
        Scalar, array, CoefficientFunction, or boundary dict for BC enforcement.
    dirichlet
        Boundary pattern label from space construction.
    u_exact
        Optional exact solution (coefficient or grid function) for error norms.
    _gfu_exact
        Cached exact solution on this mesh when ``u_exact`` was passed to
        ``from_forms``.
    """

    mesh: ng.Mesh
    fes: FESpace
    a: BilinearForm
    f: LinearForm
    gfu: GridFunction
    free_ids: np.ndarray
    fixed_ids: np.ndarray
    P: Optional[BaseMatrix] = None   # coarse -> fine, into this level
    PT: Optional[BaseMatrix] = None  # fine -> coarse, out of this level

    dirichlet_value: DirichletValue = 0.0
    dirichlet: str = ""          # boundary pattern used to build the FE space

    _A_csr: Optional[sp.csr_matrix] = field(default=None, repr=False, compare=False)
    _smoother: Any | None = field(default=None, repr=False, compare=False)
    _boundary_ids: dict = field(default_factory=dict, repr=False, compare=False)

    u_exact: "ng.CoefficientFunction | GridFunction | None" = None
    _gfu_exact: GridFunction | None = None

    # -- construction -------------------------------------------------------
    @staticmethod
    def transfer_available(fes) -> bool:
        """Whether prolongation operators can be built for ``fes``.

        Returns
        -------
        bool
            True when ``fes.mesh.levels > 1``.

        Notes
        -----
        ``Prolongation()`` always exists; on an unrefined mesh ``CreateMatrix``
        returns a zero-width matrix instead of failing.
        """
        return fes.mesh.levels > 1

    @classmethod
    def from_forms(cls, mesh, fes, a, f, *, P=None, PT=None,
                   gfu=None,
                   dirichlet_value: DirichletValue = 0.0,
                   dirichlet="",
                   built_P: bool = True,
                   u_exact: "ng.CoefficientFunction | GridFunction | None" = None,
                ) -> "Level":
        """Construct one multigrid level from assembled forms on a single mesh.

        Parameters
        ----------
        mesh
            NGSolve mesh for this level.
        fes
            Finite element space on ``mesh``.
        a
            Assembled bilinear form; ``a.mat`` is the level operator.
        f
            Assembled linear form; ``f.vec`` is the load on this mesh.
        P, PT
            Prolongation (coarse -> fine) and restriction. If omitted and
            ``built_P`` is True, built when ``transfer_available(fes)``.
        gfu
            Solution GridFunction; default is a new zero field.
        dirichlet_value
            Scalar, array, CoefficientFunction, or ``{boundary: value}`` dict.
        dirichlet
            Boundary pattern string stored for later BC enforcement.
        built_P
            If True, auto-create ``P``/``PT`` when the mesh supports it.
        u_exact
            Manufactured or reference solution (``CoefficientFunction`` or
            ``GridFunction``). Stored as ``_gfu_exact`` for ``error_norm`` /
            ``record_norms`` energy keys.
        Returns
        -------
        Level

        Notes
        -----
        Call after ``a.Assemble()`` and ``f.Assemble()``. Each hierarchy level
        gets its own ``mesh``, ``a``, and ``f``. Only the finest ``f.vec`` is
        used as the global RHS in ``MultigridSolver.solve``; the V-cycle does not
        overwrite ``f.vec``. Set ``built_P=False`` on a mesh with no coarser neighbour.
        """
        if gfu is None:
            gfu = GridFunction(fes)
        free_ids, fixed_ids = get_free_fixed_ids(fes)
        if P is None and built_P and cls.transfer_available(fes):
            P = fes.Prolongation().CreateMatrix(fes.mesh.levels - 1)
        if PT is None and P is not None:
            PT = P.CreateTranspose()

        gfu_exact = None
        if u_exact is not None:
            gfu_exact = ng.GridFunction(fes)
            gfu_exact.Set(u_exact)

        return cls(mesh=mesh, fes=fes, a=a, f=f, gfu=gfu,
                   free_ids=free_ids, fixed_ids=fixed_ids,
                   P=P, PT=PT, dirichlet_value=dirichlet_value, dirichlet=dirichlet,
                   u_exact=u_exact, _gfu_exact=gfu_exact)

    # -- cached operators ---------------------------------------------------
    @property
    def ndof(self) -> int:
        return self.fes.ndof

    @property
    def A_csr(self) -> sp.csr_matrix:
        """CSR stiffness matrix (cached).

        Notes
        -----
        Invalid after reassembly; call ``refresh()`` first.
        """
        if self._A_csr is None:
            self._A_csr = bilinear_form_to_csr(self.a)
        return self._A_csr

    @property
    def smoother(self) -> Any:
        """NGSolve symmetric Gauss-Seidel preconditioner on free DOFs (cached).

        Notes
        -----
        Built with ``CreateSmoother(FreeDofs(), GS=True)``. Native sweeps use
        ``SmoothBack`` when available; otherwise the same ``smoother * residual``
        update for forward and backward. Invalid after reassembly; call ``refresh()``.
        """
        sm = self._smoother
        if sm is None:
            sm = self.a.mat.CreateSmoother(self.fes.FreeDofs(), GS=True)
            self._smoother = sm
        return sm

    def refresh(self) -> None:
        """Clear cached CSR matrix and native smoother.

        Notes
        -----
        Required after ``a.Assemble()`` if coefficients, mesh, or BCs changed.
        """
        self._A_csr = None
        self._smoother = None

    # -- DOF / vector helpers ----------------------------------------------
    def gfu_np(self) -> np.ndarray:
        return self.gfu.vec.FV().NumPy()

    def dirichlet_ids(self, name: str) -> np.ndarray:
        """Fixed DOF indices on a named boundary.

        Parameters
        ----------
        name
            Boundary label; regex allowed.

        Returns
        -------
        ndarray
            Subset of ``fixed_ids`` on that boundary.
        """
        if name not in self._boundary_ids:
            geom = boundary_dof_ids(self.fes, name)
            self._boundary_ids[name] = np.intersect1d(geom, self.fixed_ids)
        return self._boundary_ids[name]

    def enforce_dirichlet(self, x=None, values=None) -> None:
        """Set Dirichlet values on fixed DOFs.

        Parameters
        ----------
        x
            Vector to modify; default ``gfu.vec``.
        values
            Scalar, array, CoefficientFunction, or ``{boundary: value}`` dict;
            default ``dirichlet_value``.

        Notes
        -----
        Dict mode zeros all fixed DOFs first, then applies each boundary.
        Only ``fixed_ids`` are written; free DOFs are unchanged.
        """
        if values is None:
            values = self.dirichlet_value
        target = self.gfu.vec if x is None else x

        if isinstance(values, dict):
            self._enforce_dirichlet_dict(target, values)
        elif isinstance(values, ng.CoefficientFunction):
            self._project_cf_onto(target, values, self.dirichlet, self.fixed_ids)
        else:
            apply_dirichlet(target, self.fixed_ids, values)

    def _enforce_dirichlet_dict(self, target, spec: dict) -> None:
        arr = target.FV().NumPy()
        arr[self.fixed_ids] = 0.0  # deterministic default for unlisted boundaries
        covered: list[np.ndarray] = []
        for name, val in spec.items():
            ids = self.dirichlet_ids(name)
            if len(ids) == 0:
                warnings.warn(
                    f"Dirichlet spec key {name!r} matches no fixed DOFs on this "
                    f"level (known boundaries: {self.mesh.GetBoundaries()}).",
                    stacklevel=3,
                )
                continue
            covered.append(ids)
            if isinstance(val, ng.CoefficientFunction):
                self._project_cf_onto(target, val, name, ids)
            else:
                arr[ids] = val

        if covered:
            n_done = len(np.unique(np.concatenate(covered)))
            if n_done < len(self.fixed_ids):
                warnings.warn(
                    f"Dirichlet spec covers {n_done}/{len(self.fixed_ids)} fixed "
                    f"DOFs; the rest were set to 0.",
                    stacklevel=3,
                )

    def _project_cf_onto(self, target, cf, region_name: str, ids: np.ndarray) -> None:
        """Project a boundary coefficient onto selected DOFs of ``target``."""
        tmp = GridFunction(self.fes)
        pattern = region_name if region_name else ".*"
        tmp.Set(cf, definedon=self.mesh.Boundaries(pattern))
        target.FV().NumPy()[ids] = tmp.vec.FV().NumPy()[ids]

    def set_initial_guess(self, cf, *, enforce_bc: bool = True) -> None:
        """Set ``gfu`` from a coefficient function and optionally enforce BCs.

        Parameters
        ----------
        cf
            Initial guess or boundary-compatible field.
        enforce_bc
            If True, call ``enforce_dirichlet`` after interpolation.
        """
        self.gfu.Set(cf)
        if enforce_bc:
            self.enforce_dirichlet()

    def residual(self, b, x):
        """Compute residual ``b - A x``.

        Parameters
        ----------
        b, x
            RHS and iterate (read-only).

        Returns
        -------
        BaseVector
            New residual vector.
        """
        r = x.CreateVector()
        r.data = b - self.a.mat * x
        return r

    def _exact_vec(self, u_exact=None):
        """Coefficient vector for the exact solution on this level."""
        if u_exact is not None and hasattr(u_exact, "vec"):
            return u_exact.vec

        if u_exact is not None:
            tmp = GridFunction(self.fes)
            tmp.Set(u_exact)
            return tmp.vec

        if self._gfu_exact is not None:
            return self._gfu_exact.vec

        if self.u_exact is not None:
            tmp = GridFunction(self.fes)
            tmp.Set(self.u_exact)
            return tmp.vec

        raise ValueError("u_exact is not set on this level")


    def exact_error(self, x, u_exact=None):
        e = x.CreateVector()
        e.data = self._exact_vec(u_exact) - x
        return e


    def vector_norm(self, vec, *, op=None) -> float:
        """``sqrt(vec^T op vec)`` if ``op`` is given, else ``||vec||_2`` on free DOFs."""
        return _vector_norm(vec, op, free_ids=self.free_ids)

    def energy_norm(self, vec) -> float:
        """``||v||_A = sqrt(v^T A v)`` with this level's stiffness matrix."""
        return self.vector_norm(vec, op=self.a.mat)

    def residual_norm(self, b=None, x=None) -> float:
        """``||b - A x||_2`` on free DOFs (algebraic residual for stopping)."""
        b = self.f.vec if b is None else b
        x = self.gfu.vec if x is None else x
        r = self.residual(b, x)
        r.FV().NumPy()[self.fixed_ids] = 0.0
        return self.vector_norm(r)

    def error_norm(self, x=None, *, u_exact=None) -> float:
        """``||u_exact - x||_A`` on free DOFs."""
        x = self.gfu.vec if x is None else x
        e = self.exact_error(x, u_exact=u_exact)
        e.FV().NumPy()[self.fixed_ids] = 0.0
        return self.energy_norm(e)

    # -- solves / smoothing -------------------------------------------------
    def coarse_solve(self, b, x) -> None:
        """Direct solve ``A x = b`` on free DOFs.

        Parameters
        ----------
        b
            RHS (unchanged).
        x
            Solution; free entries overwritten, fixed entries preserved.
        """
        x.data = self.a.mat.Inverse(self.fes.FreeDofs()) * b

    def smooth(self, b, x, *, kind: str = "native",
               nsweeps: int = 2, omega: float = 1.0, verbose: bool = False,
               backward: bool = False) -> None:
        """Gauss-Seidel relaxation for ``A x = b``.

        Parameters
        ----------
        b, x
            RHS (read-only) and iterate (updated in place on free DOFs).
        kind
            ``"native"`` (NGSolve smoother) or ``"gs"`` (scipy CSR).
        nsweeps
            Number of sweeps.
        omega
            Relaxation factor (1.0 = none).
        verbose
            Print residual norm per sweep.
        backward
            Forward sweep if False; backward if True (post-smooth leg).

        Notes
        -----
        Fixed DOFs in ``x`` are not modified.
        """
        if nsweeps <= 0:
            return
        if kind == "native":
            self._smooth_native(b, x, nsweeps=nsweeps, omega=omega,
                                verbose=verbose, backward=backward)
        elif kind == "gs":
            self._smooth_scipy_gs(b, x, nsweeps=nsweeps, omega=omega,
                                  verbose=verbose, backward=backward)
        else:
            raise ValueError(f"Unknown smoother kind: {kind!r} (use 'native' or 'gs').")

    def _smooth_native(self, b, x, *, nsweeps, omega, verbose, backward) -> None:
        """NGSolve GS=True: ``SmoothBack`` if present, else ``sm * (b - A x)``."""
        sm = self.smoother
        if sm is None:
            raise RuntimeError("Native smoother is not available.")
        x0 = x.CreateVector()
        for sweep in range(1, nsweeps + 1):
            smooth_back = getattr(sm, "SmoothBack", None)
            if backward and smooth_back is not None:
                x0.data = x
                smooth_back(x, b)
                if omega != 1.0:
                    x.data = x0.data + omega * (x.data - x0.data)
            else:
                r = b - self.a.mat * x
                x.data += omega * (sm * r)
            if verbose:
                rn = self.residual_norm(b, x)
                tag = "back" if backward else "fwd"
                print(f"    [native {tag}] sweep {sweep:3d}  ||r_free|| = {rn:.6e}")

    def _smooth_scipy_gs(self, b, x, *, nsweeps, omega, verbose, backward) -> None:
        """Scipy CSR Gauss-Seidel implementation."""
        b_np = b.FV().NumPy()
        x_np = x.FV().NumPy().copy()
        free = self.free_ids[::-1] if backward else self.free_ids
        x_np, _ = gauss_seidel_sweeps(
            self.A_csr, b_np, x_np, free,
            nsweeps=nsweeps, omega=omega, verbose=verbose,
        )
        x.FV().NumPy()[:] = x_np


# ---------------------------------------------------------------------------
# Multigrid hierarchy
# ---------------------------------------------------------------------------
@dataclass
class MultigridHierarchy:
    """Coarse-to-fine stack of multigrid levels.

    Attributes
    ----------
    levels
        ``Level`` instances ordered coarse (index 0) to fine (index -1).
    """

    levels: list[Level]

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError("A multigrid hierarchy needs at least 2 levels.")
        if self.levels[0].P is not None or self.levels[0].PT is not None:
            raise ValueError("Coarsest level (index 0) must have P = PT = None.")
        for i, lvl in enumerate(self.levels[1:], start=1):
            if lvl.P is None or lvl.PT is None:
                raise ValueError(f"Level {i} is missing a grid-transfer operator.")

    @property
    def nlevels(self) -> int:
        return len(self.levels)

    @property
    def coarsest_idx(self) -> int:
        return 0

    @property
    def finest_idx(self) -> int:
        return self.nlevels - 1

    @property
    def finest(self) -> Level:
        return self.levels[-1]

    @property
    def coarsest(self) -> Level:
        return self.levels[0]

    def info(self) -> None:
        """Print ndof, matrix shapes, and transfer sizes per level."""
        print(f"  {'lev':>3}  {'ndof':>7}  {'A':>13}  {'P(c->f)':>12}  {'nfree':>7}  {'nfixed':>6}")
        print(f"  {'---':>3}  {'-------':>7}  {'-------------':>13}  {'------------':>12}  {'-------':>7}  {'------':>6}")
        for lev, level in enumerate(self.levels):
            a_shape = f"{level.a.mat.height}x{level.a.mat.width}"
            p_shape = f"{level.P.height}x{level.P.width}" if level.P is not None else "-"
            tag = "  (coarse)" if lev == self.coarsest_idx else ("  (fine)" if lev == self.finest_idx else "")
            print(f"  {lev:>3}  {level.fes.ndof:>7}  {a_shape:>13}  {p_shape:>12}  "
                  f"{len(level.free_ids):>7}  {len(level.fixed_ids):>6}{tag}")


def build_hierarchy(
    coarse_mesh: ng.Mesh,
    form_setup: FormSetupFn,
    *,
    n_refines: int,
    order: int = 1,
    dirichlet: str = "left|right|top|bottom",
    dirichlet_value: DirichletValue = 0.0,
    u_exact: "ng.CoefficientFunction | GridFunction | None" = None,
    verbose: bool = False,
) -> MultigridHierarchy:
    """Uniform refinement hierarchy with per-level forms and transfers.

    Parameters
    ----------
    coarse_mesh
        Starting mesh (copied; original unchanged).
    form_setup
        ``setup(fes) -> (a, f)`` from ``build_form_setup``.
    n_refines
        Number of refinements (>= 1); produces ``n_refines + 1`` levels.
    order
        H1 polynomial order on every level.
    dirichlet
        Boundary pattern for the FE space.
    dirichlet_value
        Stored on each level for BC enforcement.
    u_exact
        Optional exact solution, forwarded to ``Level.from_forms`` on every
        level (interpolated onto each mesh). Enables ``solve(..., norms=...,
        "energy")`` and ``error_norm`` on the finest level.
    verbose
        Print ``info()`` after construction.

    Returns
    -------
    MultigridHierarchy

    Notes
    -----
    Each level assembles the same PDE on its mesh. Coarsest level has no ``P``/``PT``.
    """
    if n_refines < 1:
        raise ValueError("n_refines must be >= 1.")

    working = ng.Mesh(coarse_mesh.ngmesh.Copy())
    levels: list[Level] = []
    pending_P = None

    for lev in range(n_refines + 1):
        snapshot = ng.Mesh(working.ngmesh.Copy())
        fes = H1(snapshot, order=order, dirichlet=dirichlet)
        a, f = form_setup(fes)
        a.Assemble()
        f.Assemble()

        level = Level.from_forms(
            snapshot, fes, a, f,
            P=pending_P,
            PT=pending_P.CreateTranspose() if pending_P is not None else None,
            dirichlet_value=dirichlet_value,
            dirichlet=dirichlet,
            u_exact=u_exact,
        )
        levels.append(level)

        if lev < n_refines:
            working.Refine()
            fes_next = H1(working, order=order, dirichlet=dirichlet)
            pending_P = fes_next.Prolongation().CreateMatrix(working.levels - 1)
    hierarchy = MultigridHierarchy(levels)
    if verbose:
        hierarchy.info()
    return hierarchy


# Backwards-compatible alias for the name used in the test script.
build_multilevel_data = build_hierarchy


# ---------------------------------------------------------------------------
# V-cycle solver
# ---------------------------------------------------------------------------
@dataclass
class VCycleConfig:
    """V-cycle and outer iteration settings.

    Attributes
    ----------
    smoother
        ``"native"`` (NGSolve) or ``"gs"`` (scipy CSR).
    pre_sweeps, post_sweeps
        Forward GS sweeps before restriction; backward sweeps after prolongation.
    omega
        Relaxation factor passed to each smooth call.
    coarse_direct
        If True, exact solve on the coarsest level; otherwise smooth only.
    coarse_sweeps
        Sweeps on coarsest when ``coarse_direct`` is False.
    """

    smoother: SmootherKind = "native"
    pre_sweeps: int = 2
    post_sweeps: int = 2
    omega: float = 1.0
    coarse_direct: bool = True
    coarse_sweeps: int = 20  # used only when coarse_direct is False


class MultigridSolver:
    """Geometric V-cycle on a ``MultigridHierarchy``."""

    def __init__(self, hierarchy: MultigridHierarchy,
                 config: Optional[VCycleConfig] = None) -> None:
        """Attach a hierarchy and optional cycle configuration.

        Parameters
        ----------
        hierarchy
            Built coarse-to-fine levels with valid transfers.
        config
            Smoothing and coarse-solve options; default ``VCycleConfig()``.
        """
        self.h = hierarchy
        self.cfg = config or VCycleConfig()


    def _as_tuple(self, norms):
        """Allow either a single norm string or a list/tuple of norm strings."""
        if norms is None:
            return tuple()
        if isinstance(norms, str):
            return (norms,)
        return tuple(norms)


    def _zero_fixed(self, level, vec):
        """Zero fixed DOFs of a vector and return the same vector."""
        vec.FV().NumPy()[level.fixed_ids] = 0.0
        return vec


    def _copy_vec(self, vec):
        """Copy an NGSolve vector."""
        out = vec.CreateVector()
        out.data = vec
        return out


    def _exact_error_vec(self, level, x):
        """Return e = u_exact - x on this level."""
        if getattr(level, "_gfu_exact", None) is not None:
            exact_vec = level._gfu_exact.vec
        elif getattr(level, "u_exact", None) is not None:
            tmp = GridFunction(level.fes)
            tmp.Set(level.u_exact)
            exact_vec = tmp.vec
        else:
            raise ValueError(
                "Exact-error norm requested, but this level has no u_exact."
            )

        e = x.CreateVector()
        e.FV().NumPy()[:] = exact_vec.FV().NumPy() - x.FV().NumPy()
        self._zero_fixed(level, e)
        return e


    def record_norms(
        self,
        level,
        b,
        x,
        norms,
        *,
        r_after=None,
        r_before=None,
        dx=None,
    ):
        """Evaluate diagnostics at the current state ``(b, x)``.

        Each name in ``norms`` selects one branch:

        * ``"l2"`` — ``||r_after||_2`` with ``r_after = b - A x``.
        * ``"energy"`` / ``"A"`` — ``||u_exact - x||_A`` (needs ``level._gfu_exact``).
        * ``"update_dual"``, etc. — ``sqrt(r_before^T dx)`` (needs both vectors).

        Parameters
        ----------
        level
            Current level.

        b, x
            RHS and current iterate.

        norms
            Norm names to record.

        r_after
            Residual at the state being measured, ``b - A x``. Required for L2
            keys unless recomputed from ``(b, x)``. For a single snapshot with
            no prior MG step (e.g. ``solve`` cycle 0), pass the current
            residual here — the name means “residual at this state,” not “after
            a V-cycle.”

        r_before
            Residual before a local update: ``b - A x_before``. Required for
            dual keys together with ``dx``.

        dx
            Update ``x_after - x_before``. Required for dual keys.

        Returns
        -------
        dict
            Dictionary mapping norm name -> scalar value.
        """
        norms = self._as_tuple(norms)
        out = {}

        need_residual = any(norm in _L2_NORM_KEYS for norm in norms)

        if r_after is None and need_residual:
            r_after = level.residual(b, x)
            self._zero_fixed(level, r_after)

        for norm in norms:
            # ------------------------------------------------------------
            # Residual L2 norm: ||r||_2
            # ------------------------------------------------------------
            if norm in _L2_NORM_KEYS:
                v = self._copy_vec(r_after)
                self._zero_fixed(level, v)
                out[norm] = level.vector_norm(v)

            # ------------------------------------------------------------
            # Exact/manufactured energy error: sqrt(e^T A e)
            # where e = u_exact - x.
            # ------------------------------------------------------------
            elif norm in _ENERGY_ERROR_NORM_KEYS:
                e = self._exact_error_vec(level, x)
                out[norm] = level.energy_norm(e)

            # ------------------------------------------------------------
            # Approximate dual/update norm: sqrt(r_before^T dx)
            # where dx = x_after - x_before.
            # ------------------------------------------------------------
            elif norm in _DUAL_NORM_KEYS:
                if r_before is None or dx is None:
                    raise ValueError(
                        f"{norm!r} requires r_before and dx. "
                        "Use r_before = b - A*x_before and "
                        "dx = x_after - x_before."
                    )

                rb = self._copy_vec(r_before)
                dz = self._copy_vec(dx)
                self._zero_fixed(level, rb)
                self._zero_fixed(level, dz)

                quad = float(InnerProduct(rb, dz))

                if quad < 0:
                    warnings.warn(
                        f"{norm}: r_before^T dx = {quad:.6e} < 0. "
                        "This quantity is only a norm if the update behaves like "
                        "a symmetric positive definite approximate inverse.",
                        stacklevel=2,
                    )

                out[norm] = float(np.sqrt(max(quad, 0.0)))

            elif norm in _MASS_NORM_KEYS:
                raise NotImplementedError("Mass norm not implemented")

            else:
                raise ValueError(f"Unknown norm/diagnostic {norm!r}")

        return out


    def _smooth_record(
        self,
        level,
        b,
        x,
        nsweeps,
        norms,
        *,
        verbose=False,
        backward: bool = False,
        ):
        """Run one smoothing block (``nsweeps``) and record norms once after it."""
        norms = self._as_tuple(norms)
        out = {norm: [] for norm in norms}

        need_dual = any(norm in _DUAL_NORM_KEYS for norm in norms)


        if need_dual:
            r_before = level.residual(b, x)
            self._zero_fixed(level, r_before)

            x_before = self._copy_vec(x)
        else:
            r_before = None
            x_before = None

        level.smooth(
            b,
            x,
            kind=self.cfg.smoother,
            nsweeps=nsweeps,
            omega=self.cfg.omega,
            verbose=verbose,
            backward=backward,
        )

        r_after = level.residual(b, x)
        self._zero_fixed(level, r_after)

        if need_dual:
            dx = x.CreateVector()
            dx.data = x
            dx.data -= x_before
            self._zero_fixed(level, dx)
        else:
            dx = None

        vals = self.record_norms(
            level,
            b,
            x,
            norms,
            r_after=r_after,
            r_before=r_before,
            dx=dx,
        )

        for norm in norms:
            out[norm].append(vals[norm])

        return out

    def v_cycle(
        self,
        idx: int,
        b,
        x,
        *,
        verbose: bool = False,
        rec_cycle: Optional[dict] = None,
        rec: Optional[dict] = None,
        norms: "NormKind | object" | list["NormKind | object"] = "l2",
        debug: bool = False,
    ) -> None:
        """One V-cycle: approximately solve A_idx x = b in place.

        This version supports multiple diagnostics.

        Parameters
        ----------
        idx
            Level index, with 0 = coarsest.

        b, x
            RHS and iterate/correction. The vector x is updated in place.

        rec_cycle
            If not None, stores one diagnostic dictionary per level:
                rec_cycle[idx] = {"l2": value, "update_dual": value, ...}

        rec
            If not None, stores smoothing-block diagnostics:
                rec[idx] = {
                    "down": {"l2": [...], "update_dual": [...]},
                    "up":   {"l2": [...], "update_dual": [...]},
                }

        norms
            One norm name or a list/tuple of norm names.

        debug
            Print dimension information.
        """
        norms = self._as_tuple(norms)
        level = self.h.levels[idx]
        pad = "  " * (self.h.finest_idx - idx)

        # We need these only if we want the update-dual diagnostic
        # for the whole V-cycle on this level:
        #
        #     sqrt(r_before^T dx)
        #
        # where dx = x_after - x_before.
        need_cycle_dual = (
            rec_cycle is not None
            and any(norm in _DUAL_NORM_KEYS for norm in norms)
        )

        if need_cycle_dual:
            r_cycle_before = level.residual(b, x)
            self._zero_fixed(level, r_cycle_before)

            x_cycle_before = self._copy_vec(x)
        else:
            r_cycle_before = None
            x_cycle_before = None

        # ============================================================
        # Coarsest level
        # ============================================================
        if idx == self.h.coarsest_idx:
            if debug:
                print(
                    f"{pad}[lvl {idx}] coarsest: "
                    f"x={len(x)}, b={len(b)}, "
                    f"A={level.a.mat.height}x{level.a.mat.width}, "
                    f"direct={self.cfg.coarse_direct}"
                )

            # ------------------------------------------------------------
            # Coarsest level: direct solve
            # ------------------------------------------------------------
            if self.cfg.coarse_direct:
                need_dual = (
                    (rec is not None or rec_cycle is not None)
                    and any(norm in _DUAL_NORM_KEYS for norm in norms)
                )

                if need_dual:
                    r_before = level.residual(b, x)
                    self._zero_fixed(level, r_before)

                    x_before = self._copy_vec(x)
                else:
                    r_before = None
                    x_before = None

                # Direct coarse solve updates x in place.
                level.coarse_solve(b, x)

                if rec is not None or rec_cycle is not None:
                    r_after = level.residual(b, x)
                    self._zero_fixed(level, r_after)

                    if need_dual:
                        dx = x.CreateVector()
                        dx.data = x
                        dx.data -= x_before
                        self._zero_fixed(level, dx)
                    else:
                        dx = None

                    vals = self.record_norms(
                        level,
                        b,
                        x,
                        norms,
                        r_after=r_after,
                        r_before=r_before,
                        dx=dx,
                    )

                    if rec_cycle is not None:
                        rec_cycle[idx] = vals

                    if rec is not None:
                        # Match your smoothing-block convention:
                        # record once after the coarse update block.
                        rec[idx] = {
                            "down": {norm: [vals[norm]] for norm in norms},
                            "up": {norm: [] for norm in norms},
                        }

            # ------------------------------------------------------------
            # Coarsest level: smoothing instead of direct solve
            # ------------------------------------------------------------
            else:
                if rec is not None:
                    down = self._smooth_record(
                        level,
                        b,
                        x,
                        self.cfg.coarse_sweeps,
                        norms,
                        verbose=verbose,
                        backward=False,
                    )

                    rec[idx] = {
                        "down": down,
                        "up": {norm: [] for norm in norms},
                    }
                else:
                    level.smooth(
                        b,
                        x,
                        kind=self.cfg.smoother,
                        nsweeps=self.cfg.coarse_sweeps,
                        omega=self.cfg.omega,
                        verbose=verbose,
                        backward=False,
                    )

                if rec_cycle is not None:
                    r_after = level.residual(b, x)
                    self._zero_fixed(level, r_after)

                    if need_cycle_dual:
                        dx = x.CreateVector()
                        dx.data = x
                        dx.data -= x_cycle_before
                        self._zero_fixed(level, dx)
                    else:
                        dx = None

                    vals = self.record_norms(
                        level,
                        b,
                        x,
                        norms,
                        r_after=r_after,
                        r_before=r_cycle_before,
                        dx=dx,
                    )

                    rec_cycle[idx] = vals

            return
        # ============================================================
        # Non-coarsest levels
        # ============================================================
        P = level.P
        PT = level.PT
        if P is None or PT is None:
            raise RuntimeError(f"Level {idx} is missing transfer operators P/PT.")

        if debug:
            print(
                f"{pad}[lvl {idx}] down: "
                f"x={len(x)}, b={len(b)}, "
                f"A={level.a.mat.height}x{level.a.mat.width}, "
                f"PT={PT.height}x{PT.width}, "
                f"P={P.height}x{P.width}"
            )

        # ------------------------------------------------------------
        # 1. Pre-smoothing
        # ------------------------------------------------------------
        down = None
        if rec is not None:
            down = self._smooth_record(
                level,
                b,
                x,
                self.cfg.pre_sweeps,
                norms,
                verbose=verbose,
                backward=False,
            )
        else:
            level.smooth(
                b,
                x,
                kind=self.cfg.smoother,
                nsweeps=self.cfg.pre_sweeps,
                omega=self.cfg.omega,
                verbose=verbose,
                backward=False,
            )

        # ------------------------------------------------------------
        # 2. Fine residual after pre-smoothing
        # ------------------------------------------------------------
        r = level.residual(b, x)
        self._zero_fixed(level, r)

        # ------------------------------------------------------------
        # 3. Restrict residual to coarse level
        # ------------------------------------------------------------
        r_c = PT.CreateColVector()
        r_c.data = PT * r

        if debug:
            print(
                f"{pad}[lvl {idx}] restrict: "
                f"r={len(r)} --PT({PT.height}x{PT.width})--> "
                f"r_c={len(r_c)}  (to lvl {idx - 1})"
            )

        # ------------------------------------------------------------
        # 4. Coarse-grid correction solve
        # ------------------------------------------------------------
        e_c = r_c.CreateVector()
        e_c.FV().NumPy()[:] = 0.0

        self.v_cycle(
            idx - 1,
            r_c,
            e_c,
            verbose=verbose,
            rec_cycle=rec_cycle,
            rec=rec,
            norms=norms,
            debug=debug,
        )

        # ------------------------------------------------------------
        # 5. Prolong correction and update x
        # ------------------------------------------------------------
        e_f = P.CreateColVector()
        e_f.data = P * e_c

        # This is a correction vector, so fixed/Dirichlet entries should be zero.
        self._zero_fixed(level, e_f)

        x.data += e_f

        if debug:
            print(
                f"{pad}[lvl {idx}] prolong: "
                f"e_c={len(e_c)} --P({P.height}x{P.width})--> "
                f"e_f={len(e_f)}  (back to lvl {idx})"
            )

        # ------------------------------------------------------------
        # 6. Post-smoothing
        # ------------------------------------------------------------
        if rec is not None:
            up = self._smooth_record(
                level,
                b,
                x,
                self.cfg.post_sweeps,
                norms,
                verbose=verbose,
                backward=True,
            )
            rec[idx] = {"down": down, "up": up}
        else:
            level.smooth(
                b,
                x,
                kind=self.cfg.smoother,
                nsweeps=self.cfg.post_sweeps,
                omega=self.cfg.omega,
                verbose=verbose,
                backward=True,
            )

        # ------------------------------------------------------------
        # 7. Record one diagnostic dictionary for the whole V-cycle
        #    on this level.
        # ------------------------------------------------------------
        if rec_cycle is not None:
            r_after = level.residual(b, x)
            self._zero_fixed(level, r_after)

            if need_cycle_dual:
                dx = x.CreateVector()
                dx.data = x
                dx.data -= x_cycle_before
                self._zero_fixed(level, dx)
            else:
                dx = None

            vals = self.record_norms(
                level,
                b,
                x,
                norms,
                r_after=r_after,
                r_before=r_cycle_before,
                dx=dx,
            )

            rec_cycle[idx] = vals

    @overload
    def solve(
        self,
        *,
        max_cycles: int = ...,
        tol: float = ...,
        verbose: bool = ...,
        record_levels: Literal[False] = ...,
        norms=...,
        stop_norm: str = ...,
        debug: bool = ...,
    ) -> tuple[dict[str, list], list[list[dict]]]: ...

    @overload
    def solve(
        self,
        *,
        max_cycles: int = ...,
        tol: float = ...,
        verbose: bool = ...,
        record_levels: Literal[True],
        norms=...,
        stop_norm: str = ...,
        debug: bool = ...,
    ) -> tuple[dict[str, list], list[list[dict]], list[list[dict]]]: ...

    def solve(
        self,
        *,
        max_cycles: int = 20,
        tol: float = 1e-10,
        verbose: bool = False,
        record_levels: bool = False,
        norms=("l2",),
        stop_norm: str = "l2",
        debug: bool = False,
    ) -> (
        tuple[dict[str, list], list[list[dict]]]
        | tuple[dict[str, list], list[list[dict]], list[list[dict]]]
    ):
        """Repeated V-cycles on the finest-level system until tolerance or cap.

        Parameters
        ----------
        max_cycles
            Maximum number of V-cycles.

        tol
            Relative stopping tolerance:

                delta <= tol * delta_0

            where delta is the current scalar stopping norm and delta_0 is
            the initial scalar stopping norm.

        verbose
            Print initial and per-cycle finest-level diagnostics.

        record_levels
            If True, also return detailed per-level/per-step diagnostics.

        norms
            Finest-level diagnostics appended after each V-cycle. Use ``"l2"``
            (or aliases) for ``||r||_2``; ``"energy"``/``"A"`` for
            ``||u_exact - x||_A`` (needs ``u_exact`` on the finest level);
            ``"update_dual"`` for ``sqrt(r_before^T dx)`` over the full cycle
            on each level (not for stopping).

            Examples
            --------
            norms=("l2",)

            norms=("l2", "update_dual")

        stop_norm
            Which entry in ``norms`` controls ``tol``. Must be an L2 residual
            key (e.g. ``"l2"``). Energy and dual keys cannot be used for stopping.

        debug
            Dimension trace for the first cycle only.

        Returns
        -------
        hist
            ``hist[norm]`` lists one scalar per completed V-cycle (not the
            initial value printed when ``verbose=True`` at cycle 0).

        level_hist
            ``level_hist[level_idx][cycle]`` is the same norm dict recorded in
            ``v_cycle`` for that level. Energy keys on the finest level are
            filled after the cycle (they are not computed on coarse levels).

        sweep_hist
            Only if ``record_levels=True``: per-level lists of
            ``{"down": ..., "up": ...}`` smoothing-block histories.

        Notes
        -----
        Initial stopping scale ``delta_0`` uses a snapshot ``(b, x)`` before the
        first cycle. Relative stop: ``delta <= tol * delta_0`` with ``delta``
        the finest ``stop_norm`` after each cycle.
        """
        
        fine = self.h.finest
        x = fine.gfu.vec
        fine.enforce_dirichlet(x)
        b = fine.f.vec
        x_0 = self._copy_vec(fine.gfu.vec)
        b_0 = self._copy_vec(fine.f.vec)

        norms = self._as_tuple(norms)


        all_norms = self._as_tuple(norms)
        if stop_norm not in all_norms:
            all_norms = (stop_norm,) + all_norms

        energy_norms = tuple(
            n for n in all_norms
            if n in _ENERGY_ERROR_NORM_KEYS
        )

        vcycle_norms = tuple(
            n for n in all_norms
            if n not in _ENERGY_ERROR_NORM_KEYS
        )


        if stop_norm not in _L2_NORM_KEYS:
            raise ValueError(
                "Use an L2 residual norm for stopping, e.g. stop_norm='l2'. "
                "Do not use energy or update_dual as the stopping norm."
            )

        if energy_norms and fine._gfu_exact is None and fine.u_exact is None:
            raise ValueError(
                "norms 'energy'/'A' in solve/record_norms measure ||u_exact-x||_A "
                "and require u_exact on the finest level (_gfu_exact)."
            )


        # ------------------------------------------------------------
        # Histories
        # ------------------------------------------------------------
        hist = {norm: [] for norm in all_norms}

        level_hist: list[list[dict]] = [
            [] for _ in range(self.h.nlevels)
        ]

        sweep_hist: "list[list[dict]] | None" = (
            [[] for _ in range(self.h.nlevels)] if record_levels else None
        )

        # ------------------------------------------------------------
        # Initial residual vector and initial scalar stopping norm
        # ------------------------------------------------------------
        r_0 = fine.residual(b_0, x_0)
        self._zero_fixed(fine, r_0)
        init_vals = self.record_norms(
            fine,
            b_0,
            x_0,
            (stop_norm,),
            r_after=r_0,
        )

        delta_0 = init_vals[stop_norm]

        if verbose:
            print(f"cycle {0:3d}  {stop_norm} = {delta_0:.6e}")

        # ------------------------------------------------------------
        # Main V-cycle loop
        # ------------------------------------------------------------
        for cyc in range(1, max_cycles + 1):
            rec_cycle: dict[int, dict] = {}
            rec_sweeps = {} if record_levels else None

            if debug and cyc == 1:
                print(f"--- v_cycle dimension trace (cycle {cyc}) ---")

            self.v_cycle(
                self.h.finest_idx,
                b,
                x,
                rec_cycle=rec_cycle,
                rec=rec_sweeps,
                norms=vcycle_norms,
                debug=(debug and cyc == 1),
            )

            fine.enforce_dirichlet(x)

            # Finest-level diagnostics after this full V-cycle.
            vals = dict(rec_cycle[self.h.finest_idx])

            if energy_norms:
                energy_vals = self.record_norms(
                    fine,
                    b,
                    x,
                    energy_norms,
                )
                vals.update(energy_vals)

            for norm in all_norms:
                hist[norm].append(vals[norm])

        
            for idx in range(self.h.nlevels):
                entry = dict(rec_cycle.get(idx, {}))
                if idx == self.h.finest_idx:
                    entry.update({n: vals[n] for n in energy_norms})
                level_hist[idx].append(entry)

            if sweep_hist is not None:
                for idx in range(self.h.nlevels):
                    sweep_hist[idx].append(rec_sweeps.get(idx))

            delta = vals[stop_norm]

            if verbose:
                prev_delta = delta_0 if cyc == 1 else hist[stop_norm][-2]
                rate = delta / prev_delta
                print(
                    f"cycle {cyc:3d}  {stop_norm} = {delta:.6e}  "
                    f"(rate {rate:.3f})"
                )

            if tol > 0 and delta <= tol * delta_0:
                break

        if sweep_hist is not None:
            return hist, level_hist, sweep_hist

        return hist, level_hist
