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
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
import scipy.sparse as sp

import ngsolve as ng
from ngsolve import BilinearForm, GridFunction, H1, InnerProduct, LinearForm, grad, dx

# Make the project's ``src`` helpers importable regardless of CWD.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from NGSolve_utils import (  # noqa: E402
    apply_dirichlet,
    bilinear_form_to_csr,
    boundary_dof_ids,
    get_free_fixed_ids,
    vector_norm as _vector_norm,
)
from preconditioners import gauss_seidel_sweeps  

FormSetupFn = Callable[[object], "tuple[BilinearForm, LinearForm]"]
SmootherKind = Literal["gs", "native"]
# "l2": ||v||_2 on free DOFs.
# "A" / "energy": sqrt(v^T A v) — energy norm for errors e; for residuals r this is r^T A r, not ||r||_{A^{-1}}.
NormKind = Literal["l2", "euclidean", "A", "energy", "M", "mass"]


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
    """

    mesh: ng.Mesh
    fes: object
    a: BilinearForm
    f: LinearForm
    gfu: GridFunction
    free_ids: np.ndarray
    fixed_ids: np.ndarray
    P: Optional[object] = None   # coarse -> fine, into this level
    PT: Optional[object] = None  # fine -> coarse, out of this level

    dirichlet_value: "float | np.ndarray | ng.CoefficientFunction | dict" = 0.0
    dirichlet: str = ""          # boundary pattern used to build the FE space

    _A_csr: Optional[sp.csr_matrix] = field(default=None, repr=False, compare=False)
    _smoother: object = field(default=None, repr=False, compare=False)
    _boundary_ids: dict = field(default_factory=dict, repr=False, compare=False)

    u_exact: GridFunction | None = None
    _gfu_exact: ng.GridFunction | None = None



    # gfu_exact = ng.GridFunction(level.fes)
    # gfu_exact.Set(u_exact)

    # err_vec = level.gfu.vec.CreateVector()
    # err_vec.data = level.gfu.vec - gfu_exact.vec
    # err_vec.FV().NumPy()[level.fixed_ids] = 0.0
    # return level.vector_norm(err_vec, norm="energy")


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
                   gfu=None, dirichlet_value=0.0, dirichlet="",
                   built_P: bool = True, u_exact: GridFunction | None = None
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
            Exact solution for error calculation.
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
    def smoother(self):
        """NGSolve symmetric Gauss-Seidel preconditioner on free DOFs (cached).

        Notes
        -----
        Built with ``CreateSmoother(FreeDofs(), GS=True)``. Native sweeps use
        ``SmoothBack`` when available; otherwise the same ``smoother * residual``
        update for forward and backward. Invalid after reassembly; call ``refresh()``.
        """
        if self._smoother is None:
            self._smoother = self.a.mat.CreateSmoother(self.fes.FreeDofs(), GS=True)
        return self._smoother

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

    def error(self, u_exact, x):
        """Compute error ``u_exact - x``."""
        e = x.CreateVector()
        e.data = u_exact.vec - x.vec
        return e

    def vector_norm(self, vec, *, norm: "NormKind | object" = "l2") -> float:
        """Norm of a coefficient vector on this level.

        Parameters
        ----------
        r
            Vector to measure (zero on fixed DOFs for ``"A"`` / ``"energy"``).
        norm
            ``"l2"`` — Euclidean norm on ``free_ids``.
            ``"A"`` / ``"energy"`` — ``sqrt(vec^T A vec)`` with this level's ``a.mat``.
            Or any NGSolve operator ``B`` for ``sqrt(r^T B r)``.

        Returns
        -------
        float

        Notes
        -----
        For error ``e``, ``norm="energy"`` is ``||e||_A``. For residual ``r``,
        the energy error norm is ``sqrt(vec^T A^{-1} vec)`` (not implemented here;
        use ``"l2"`` for standard residual monitoring, or pass an inverse operator).
        ``"mass"`` / ``"M"`` is not implemented yet.
        """
        if isinstance(norm, str):
            key = norm.lower()
            if key in ("l2", "euclidean", "2"):
                return _vector_norm(vec, free_ids=self.free_ids)
            if key in ("a", "energy"):
                op = self.a.mat
            elif key in ("m", "mass"):
                raise NotImplementedError(
                    "Mass-matrix ('M'/'mass') norm is not implemented yet; the "
                    "mass matrix will be provided via the form setup. For now use "
                    "'l2', 'A', or pass a matrix/operator directly."
                )
            else:
                raise ValueError(   
                    f"Unknown norm {norm!r}; use 'l2', 'A', or pass an operator."
                )
        else:
            op = norm  # assume a BaseMatrix-like operator with `op * vec`

        return _vector_norm(vec, op)

    def residual_norm(self, b=None, x=None, *, norm: "NormKind | object" = "l2") -> float:
        """Residual norm ``||b - A x||`` on free DOFs.

        Parameters
        ----------
        b, x
            Defaults to ``f.vec`` and ``gfu.vec``.
        norm
            ``"l2"`` (default) — standard relative residual test for ``solve``.
            ``"A"`` / ``"energy"`` — ``sqrt(r^T A r)``, not ``||r||_{A^{-1}}``.

        Returns
        -------
        float
        """
        b = self.f.vec if b is None else b
        x = self.gfu.vec if x is None else x
        r = self.residual(b, x)
        r.FV().NumPy()[self.fixed_ids] = 0.0
        return self.vector_norm(r, norm=norm)

    def error_norm(self, u_exact, x, *, norm: "NormKind | object" = "energy") -> float:
        """Error norm ``||u_exact - x||`` on free DOFs."""
        u_exact = self.u_exact if u_exact is None else u_exact
        x = self.gfu.vec if x is None else x
        e = self.error(u_exact, x)
        e.FV().NumPy()[self.fixed_ids] = 0.0
        return self.vector_norm(e, norm=norm)

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
        x0 = x.CreateVector()
        for sweep in range(1, nsweeps + 1):
            if backward and hasattr(sm, "SmoothBack"):
                x0.data = x
                sm.SmoothBack(x, b)
                if omega != 1.0:
                    x.data = x0.data + omega * (x.data - x0.data)
            else:
                r = b - self.a.mat * x
                x.data += omega * (sm * r)
            if verbose:
                r = b - self.a.mat * x
                rn = float(np.linalg.norm(r.FV().NumPy()[self.free_ids]))
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
    dirichlet_value: "float | np.ndarray | ng.CoefficientFunction | dict" = 0.0,
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

    # -- core recursion -----------------------------------------------------
    def _smooth_record(self, level, b, x, nsweeps, norm, *, verbose=False,
                       backward: bool = False):
        """Run single-sweep smooths and record residual norms.

        Parameters
        ----------
        level, b, x
            Level and vectors for ``smooth``.
        nsweeps
            Number of sweeps to record.
        norm
            Norm for each recorded residual.
        backward
            Forward or backward GS.

        Returns
        -------
        list[float]
            Residual norm after each sweep.
        """
        out: list[float] = []
        for _ in range(nsweeps):
            level.smooth(b, x, kind=self.cfg.smoother, nsweeps=1,
                         omega=self.cfg.omega, verbose=verbose, backward=backward)
            if norm == "energy" and level.u_exact is not None:
                v = level.error(level.u_exact, x)
                v.FV().NumPy()[level.fixed_ids] = 0.0
            else:
                v = level.residual(b, x)
                v.FV().NumPy()[level.fixed_ids] = 0.0

            out.append(level.vector_norm(v, norm=norm))
        return out

    def v_cycle(self, idx: int, b, x, *, verbose: bool = False,
                rec_cycle: Optional[dict] = None,
                rec: Optional[dict] = None,
                norm: "NormKind | object" = "l2",
                debug: bool = False) -> None:
        """One V-cycle: approximate ``A_idx x = b`` in place.

        Parameters
        ----------
        idx
            Level index (0 = coarsest).
        b, x
            RHS and iterate/correction (``x`` updated in place).
        verbose
            Per-sweep smoother output.
        rec_cycle
            If set, filled with one residual norm per level per cycle.
        rec
            If set, per-level ``{"down": [...], "up": [...]}`` sweep norms.
        norm
            Residual norm type for recording.
        debug
            Print vector and operator dimensions.

        Notes
        -----
        Does not modify ``f.vec``. Coarsest: direct or iterative solve.
        Finer levels: pre-smooth, restrict, recurse, prolong, post-smooth.
        """
        level = self.h.levels[idx]
        pad = "  " * (self.h.finest_idx - idx)  # indent by descent depth

        # If at coarsest level, solve the coarse problem directly or with smoothing.
        if idx == self.h.coarsest_idx:
            if debug:
                print(f"{pad}[lvl {idx}] coarsest: "
                      f"x={len(x)}, b={len(b)}, A={level.a.mat.height}x{level.a.mat.width}, "
                      f"direct={self.cfg.coarse_direct}")

            if self.cfg.coarse_direct:
                before = None
                if rec is not None or rec_cycle is not None:
                    if norm == "energy" and level.u_exact is not None:
                        before = level.error_norm(level.u_exact, x, norm=norm)
                    else:
                        before = level.residual_norm(b, x, norm=norm)
                    if rec_cycle is not None:
                        rec_cycle[idx] = before
                level.coarse_solve(b, x)
                if rec is not None:
                    if norm == "energy" and level.u_exact is not None:
                        after = level.error_norm(level.u_exact, x, norm=norm)
                    else:
                        after = level.residual_norm(b, x, norm=norm)
                    rec[idx] = {"down": [before], "up": [after]}
            elif rec is not None:
                down = self._smooth_record(level, b, x, self.cfg.coarse_sweeps,
                                           norm, verbose=verbose)
                rec[idx] = {"down": down, "up": []}
                if rec_cycle is not None:
                    rec_cycle[idx] = down[-1]
            else:
                level.smooth(b, x, kind=self.cfg.smoother,
                             nsweeps=self.cfg.coarse_sweeps,
                             omega=self.cfg.omega, verbose=verbose)
                if rec_cycle is not None:
                    if norm == "energy" and level.u_exact is not None:
                        rec_cycle[idx] = level.error_norm(level.u_exact, x, norm=norm)
                    else:
                        rec_cycle[idx] = level.residual_norm(b, x, norm=norm)
            return

        if debug:
            print(f"{pad}[lvl {idx}] down: x={len(x)}, b={len(b)}, "
                  f"A={level.a.mat.height}x{level.a.mat.width}, "
                  f"PT={level.PT.height}x{level.PT.width}, P={level.P.height}x{level.P.width}")

        # pre-smoothing (descent)
        if rec is not None:
            down = self._smooth_record(level, b, x, self.cfg.pre_sweeps,
                                       norm, verbose=verbose)
        else:
            level.smooth(b, x, kind=self.cfg.smoother,
                         nsweeps=self.cfg.pre_sweeps, omega=self.cfg.omega, verbose=verbose)

        r = level.residual(b, x)
        r.FV().NumPy()[level.fixed_ids] = 0.0
        if rec_cycle is not None:
            if norm == "energy" and level.u_exact is not None:
                rec_cycle[idx] = level.error_norm(level.u_exact, x, norm=norm)
            else:
                rec_cycle[idx] = level.vector_norm(r, norm=norm)

        r_c = level.PT.CreateColVector()
        r_c.data = level.PT * r

        if debug:
            print(f"{pad}[lvl {idx}] restrict: r={len(r)} --PT({level.PT.height}x"
                  f"{level.PT.width})--> r_c={len(r_c)}  (to lvl {idx - 1})")

        e_c = r_c.CreateVector()
        e_c.FV().NumPy()[:] = 0.0
        self.v_cycle(idx - 1, r_c, e_c, verbose=verbose, rec_cycle=rec_cycle,
                     rec=rec, norm=norm, debug=debug)

        e_f = level.P.CreateColVector()
        e_f.data = level.P * e_c
        x.data += e_f

        if debug:
            print(f"{pad}[lvl {idx}] prolong: e_c={len(e_c)} --P({level.P.height}x"
                  f"{level.P.width})--> e_f={len(e_f)}  (back to lvl {idx})")

        # post-smoothing (ascent): backward GS on the way up (NGSolve MG pattern)
        if rec is not None:
            up = self._smooth_record(level, b, x, self.cfg.post_sweeps,
                                     norm, verbose=verbose, backward=True)
            rec[idx] = {"down": down, "up": up}
        else:
            level.smooth(b, x, kind=self.cfg.smoother,
                         nsweeps=self.cfg.post_sweeps, omega=self.cfg.omega,
                         verbose=verbose, backward=True)

    # -- drivers ------------------------------------------------------------
    def solve(self, *, max_cycles: int = 20, tol: float = 1e-10,
              verbose: bool = False, record_levels: bool = False,
              norm: "NormKind | object" = "l2", debug: bool = False
              ):
        """Repeated V-cycles on the finest-level system until tolerance or cap.

        Parameters
        ----------
        max_cycles
            Maximum V-cycles.
        tol
            Relative stop test: ``||r|| <= tol * ||r0||`` on the finest level.
            Use ``0`` to run exactly ``max_cycles`` cycles.
        verbose
            Print initial and per-cycle finest-level residuals.
        record_levels
            If True, also return per-sweep norms on all levels.
        norm
            ``"l2"`` (recommended for stopping), ``"A"`` / ``"energy"``
            (``sqrt(r^T A r)``), or a custom operator.
        debug
            Dimension trace for the first cycle only.
        Returns
        -------
        hist
            Finest-level residual after each cycle.
        level_hist
            ``level_hist[level_idx][cycle]`` snapshot per level per cycle.
        sweep_hist
            Only if ``record_levels=True``: per-sweep down/up norms per level.

        Notes
        -----
        Uses finest ``f.vec`` and ``gfu.vec``. Enforces Dirichlet on ``x`` each
        cycle. Set initial guess and BCs before calling.
        """
        fine = self.h.finest
        x = fine.gfu.vec
        b = fine.f.vec
        fine.enforce_dirichlet(x)

        nlabel = norm if isinstance(norm, str) else "B"
        hist: list[float] = []
        level_hist: list[list[float]] = [[] for _ in range(self.h.nlevels)]
        sweep_hist: "list[list[dict]] | None" = (
            [[] for _ in range(self.h.nlevels)] if record_levels else None
        )
        if norm == "energy" and fine.u_exact is not None:
            r0 = fine.error_norm(fine.u_exact, x, norm=norm)
        else:
            r0 = fine.residual_norm(b, x, norm=norm)
        if verbose:
            print(f"cycle {0:3d}  ||r||_{nlabel} = {r0:.6e}")
        for cyc in range(1, max_cycles + 1):
            rec_cycle: dict[int, float] = {}
            rec_sweeps = {} if record_levels else None
            if debug and cyc == 1:
                print(f"--- v_cycle dimension trace (cycle {cyc}) ---")
            # only trace the first cycle to avoid flooding the console
            self.v_cycle(self.h.finest_idx, b, x, rec_cycle=rec_cycle,
                         rec=rec_sweeps, norm=norm, debug=debug and cyc == 1)
            for idx in range(self.h.nlevels):
                level_hist[idx].append(rec_cycle.get(idx, float("nan")))
            if sweep_hist is not None:
                for idx in range(self.h.nlevels):
                    sweep_hist[idx].append(rec_sweeps.get(idx))
            fine.enforce_dirichlet(x)
            if norm == "energy" and fine.u_exact is not None:
                rn = fine.error_norm(fine.u_exact, x, norm=norm)
            else:
                rn = fine.residual_norm(b, x, norm=norm)
            hist.append(rn)
            if verbose:
                rate = rn / hist[-2] if len(hist) > 1 else rn / r0
                print(f"cycle {cyc:3d}  ||r||_{nlabel} = {rn:.6e}  (rate {rate:.3f})")
            if rn <= tol * r0:
                break
        if record_levels:
            return hist, level_hist, sweep_hist
        return hist, level_hist

    # def apply(self, rhs, *, x=None):
    #     """Preconditioner action ``z = M^{-1} rhs`` (one V-cycle from zero).

    #     Does not touch any level's load vector. Returns the correction ``z`` as
    #     a fresh NGSolve vector on the finest level.
    #     """
    #     fine = self.h.finest
    #     z = fine.f.vec.CreateVector() if x is None else x
    #     z.FV().NumPy()[:] = 0.0
    #     self.v_cycle(self.h.finest_idx, rhs, z)
    #     return z

    # def pcg(self, *, max_iter: int = 100, tol: float = 1e-10,
    #         verbose: bool = False) -> list[float]:
    #     """Preconditioned CG on the finest problem with this MG as preconditioner.

    #     Assumes homogeneous Dirichlet data (correction-scheme friendly). Returns
    #     the history of preconditioned-residual norms ``sqrt(r . z)``.
    #     """
    #     fine = self.h.finest
    #     A = fine.a.mat
    #     x = fine.gfu.vec
    #     fine.enforce_dirichlet(x)

    #     r = fine.residual(fine.f.vec, x)
    #     r.FV().NumPy()[fine.fixed_ids] = 0.0
    #     z = self.apply(r)
    #     p = z.CreateVector()
    #     p.data = z
    #     rz = float(ng.InnerProduct(r, z))

    #     hist: list[float] = []
    #     for it in range(1, max_iter + 1):
    #         Ap = p.CreateVector()
    #         Ap.data = A * p
    #         Ap.FV().NumPy()[fine.fixed_ids] = 0.0
    #         alpha = rz / float(ng.InnerProduct(p, Ap))
    #         x.data += alpha * p
    #         r.data -= alpha * Ap
    #         r.FV().NumPy()[fine.fixed_ids] = 0.0

    #         rn = float(np.linalg.norm(r.FV().NumPy()[fine.free_ids]))
    #         hist.append(rn)
    #         if verbose:
    #             print(f"pcg {it:3d}  ||r_free|| = {rn:.6e}")
    #         if rn <= tol:
    #             break

    #         z = self.apply(r)
    #         rz_new = float(ng.InnerProduct(r, z))
    #         beta = rz_new / rz
    #         p.data = z + beta * p
    #         rz = rz_new
    #     return hist
