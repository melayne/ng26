"""Two-/multi-level multigrid tools for NGSolve (clean rewrite).

This is a from-scratch reimplementation of ``src/multigrid_tools.py`` with the
same educational goal but a simpler, correct, and faster structure.

Key design choices that differ from the original
-------------------------------------------------
* **One level type.** ``Level`` merges the old ``LevelData`` + ``FELevel``
  (which duplicated each other). Each level caches its scipy CSR export and a
  native NGSolve smoother so repeated sweeps are cheap.
* **A textbook V-cycle on explicit vectors.** ``v_cycle(idx, b, x)`` solves
  ``A_idx x ~= b`` for a *given* right-hand side ``b`` and iterate ``x``. The
  cycle never overwrites a level's real load vector ``f``, so there is no
  save/restore dance. The preconditioner action is simply one V-cycle from a
  zero initial guess.
* **Correct smoother dispatch.** The original fell through to an unconditional
  ``raise`` after a valid smoother call, so the V-cycle could never run.
* **Pure correction scheme.** Smoothers and the coarse solve only touch free
  DOFs, so Dirichlet data lives in the finest iterate's fixed entries and is
  preserved automatically. Coarse corrections have homogeneous BCs.

Quick start
-----------
::

    from netgen.geom2d import unit_square
    import ngsolve as ng
    from multigrid_tools_2 import build_form_setup, build_hierarchy, MultigridSolver

    def poisson(a, u, v):
        a += ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
    def rhs(f, u, v):
        f += 1.0 * v * ng.dx

    setup = build_form_setup(bilinear=poisson, linear=rhs)
    coarse = ng.Mesh(unit_square.GenerateMesh(maxh=0.3))
    hierarchy = build_hierarchy(coarse, setup, n_refines=3, dirichlet="left|right|top|bottom")

    solver = MultigridSolver(hierarchy)
    res = solver.solve(max_cycles=20, tol=1e-10, verbose=True)
    u = hierarchy.finest.gfu          # solution GridFunction
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
)
from preconditioners import gauss_seidel_sweeps  # noqa: E402

FormSetupFn = Callable[[object], "tuple[BilinearForm, LinearForm]"]
SmootherKind = Literal["gs", "native"]


# ---------------------------------------------------------------------------
# Form factory
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
    """One mesh level: space, assembled forms, DOF layout, and grid transfers.

    ``P`` maps this level's *coarser* neighbour up to this level (coarse -> fine)
    and ``PT`` maps this level down to the coarser neighbour (fine -> coarse).
    Both are ``None`` on the coarsest level.
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
    # Either a single value (scalar / (nfixed,) array / CoefficientFunction)
    # applied to all Dirichlet DOFs, or a dict {boundary_name: value_or_CF}.
    dirichlet_value: "float | np.ndarray | ng.CoefficientFunction | dict" = 0.0
    dirichlet: str = ""          # boundary pattern used to build the FE space

    _A_csr: Optional[sp.csr_matrix] = field(default=None, repr=False, compare=False)
    _smoother: object = field(default=None, repr=False, compare=False)
    _boundary_ids: dict = field(default_factory=dict, repr=False, compare=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_forms(cls, mesh, fes, a, f, *, P=None, PT=None,
                   gfu=None, dirichlet_value=0.0, dirichlet="") -> "Level":
        if gfu is None:
            gfu = GridFunction(fes)
        free_ids, fixed_ids = get_free_fixed_ids(fes)
        return cls(mesh=mesh, fes=fes, a=a, f=f, gfu=gfu,
                   free_ids=free_ids, fixed_ids=fixed_ids,
                   P=P, PT=PT, dirichlet_value=dirichlet_value, dirichlet=dirichlet)

    # -- cached operators ---------------------------------------------------
    @property
    def ndof(self) -> int:
        return self.fes.ndof

    @property
    def A_csr(self) -> sp.csr_matrix:
        """scipy CSR export of the assembled stiffness matrix (cached).

        The cache assumes ``a`` is assembled once and not changed afterward
        (the invariant ``build_hierarchy`` establishes). If you reassemble
        ``a`` (changed coefficients, deformed mesh, new BCs), call
        :meth:`refresh` first, otherwise this returns the stale matrix.
        """
        if self._A_csr is None:
            self._A_csr = bilinear_form_to_csr(self.a)
        return self._A_csr

    @property
    def smoother(self):
        """Native NGSolve Gauss-Seidel smoother over free DOFs (cached).

        Like :attr:`A_csr`, this is built once from ``a.mat`` and assumes the
        matrix is fixed. Call :meth:`refresh` after any reassembly.
        """
        if self._smoother is None:
            self._smoother = self.a.mat.CreateSmoother(self.fes.FreeDofs(), GS=True)
        return self._smoother

    def refresh(self) -> None:
        """Drop cached operators so they are rebuilt from the current ``a``.

        Call this after reassembling ``a`` (e.g. ``a.Assemble()`` following a
        coefficient/mesh/BC change). NGSolve updates ``a.mat`` in place, so the
        staleness cannot be detected automatically.
        """
        self._A_csr = None
        self._smoother = None

    # -- DOF / vector helpers ----------------------------------------------
    def gfu_np(self) -> np.ndarray:
        return self.gfu.vec.FV().NumPy()

    def dirichlet_ids(self, name: str) -> np.ndarray:
        """Fixed-DOF indices on the named boundary (cached, regex allowed).

        Computed as the geometric boundary DOFs intersected with this level's
        Dirichlet (fixed) DOFs, so it never touches free DOFs.
        """
        if name not in self._boundary_ids:
            geom = boundary_dof_ids(self.fes, name)
            self._boundary_ids[name] = np.intersect1d(geom, self.fixed_ids)
        return self._boundary_ids[name]

    def enforce_dirichlet(self, x=None, values=None) -> None:
        """Pin this level's fixed DOFs in ``x`` (defaults to ``gfu``).

        ``values`` may be:

        * a **dict** ``{boundary_name: value}`` assigning a constant or a
          CoefficientFunction to each named boundary, e.g.
          ``{"left": 1.0, "right": 0.0, "top": x}``. Fixed DOFs are reset to 0
          first, then each boundary's value is written into its own DOFs.
        * a **single value**: a scalar, a 1-D array of shape
          ``(len(self.fixed_ids),)``, or a CoefficientFunction, applied to all
          Dirichlet DOFs at once.

        Only fixed DOFs are written, so interior entries of ``x`` are preserved.
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
        """Project a boundary CF and copy only ``ids`` into ``target``.

        Uses a scratch GridFunction so the boundary L2-projection does not
        clobber interior entries of ``target`` (``GridFunction.Set`` zeroes the
        whole vector before projecting).
        """
        tmp = GridFunction(self.fes)
        pattern = region_name if region_name else ".*"
        tmp.Set(cf, definedon=self.mesh.Boundaries(pattern))
        target.FV().NumPy()[ids] = tmp.vec.FV().NumPy()[ids]

    def set_initial_guess(self, cf, *, enforce_bc: bool = True) -> None:
        """Interpolate ``cf`` into ``gfu`` and (by default) pin Dirichlet DOFs.

        When ``enforce_bc`` is True, the fixed DOFs are overwritten with this
        level's ``dirichlet_value`` (which may be nonzero), not forced to zero.
        If there are no Dirichlet DOFs this is a harmless no-op.
        """
        self.gfu.Set(cf)
        if enforce_bc:
            self.enforce_dirichlet()

    def residual(self, b, x):
        """Return ``b - A x`` as a fresh NGSolve vector."""
        r = x.CreateVector()
        r.data = b - self.a.mat * x
        return r

    def free_residual_norm(self, b=None, x=None) -> float:
        b = self.f.vec if b is None else b
        x = self.gfu.vec if x is None else x
        r = self.residual(b, x).FV().NumPy()
        return float(np.linalg.norm(r[self.free_ids]))

    # -- solves / smoothing -------------------------------------------------
    def coarse_solve(self, b, x) -> None:
        """Exact solve ``A x = b`` on free DOFs (fixed DOFs of ``x`` untouched)."""
        x.data = self.a.mat.Inverse(self.fes.FreeDofs()) * b

    def smooth(self, b, x, *, kind: SmootherKind = "native",
               nsweeps: int = 2, omega: float = 1.0, verbose: bool = False) -> None:
        """Relax ``A x = b`` in place on free DOFs using the chosen smoother."""
        if nsweeps <= 0:
            return
        if kind == "native":
            self._smooth_native(b, x, nsweeps=nsweeps, omega=omega, verbose=verbose)
        elif kind == "gs":
            self._smooth_scipy_gs(b, x, nsweeps=nsweeps, omega=omega, verbose=verbose)
        else:
            raise ValueError(f"Unknown smoother kind: {kind!r} (use 'native' or 'gs').")

    def _smooth_native(self, b, x, *, nsweeps, omega, verbose) -> None:
        r = x.CreateVector()
        for sweep in range(1, nsweeps + 1):
            r.data = b - self.a.mat * x
            x.data += omega * (self.smoother * r)
            if verbose:
                rn = float(np.linalg.norm(r.FV().NumPy()[self.free_ids]))
                print(f"    [native] sweep {sweep:3d}  ||r_free|| = {rn:.6e}")

    def _smooth_scipy_gs(self, b, x, *, nsweeps, omega, verbose) -> None:
        b_np = b.FV().NumPy()
        x_np = x.FV().NumPy().copy()
        x_np, _ = gauss_seidel_sweeps(
            self.A_csr, b_np, x_np, self.free_ids,
            nsweeps=nsweeps, omega=omega, verbose=verbose,
        )
        x.FV().NumPy()[:] = x_np


# ---------------------------------------------------------------------------
# Multigrid hierarchy
# ---------------------------------------------------------------------------
@dataclass
class MultigridHierarchy:
    """Ordered list of :class:`Level` objects, coarse (index 0) -> fine (-1)."""

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
    """Refine ``coarse_mesh`` ``n_refines`` times and assemble one level per mesh.

    Produces ``n_refines + 1`` levels ordered coarse -> fine. Each non-coarsest
    level gets the prolongation ``P`` (coarse -> fine) and its transpose ``PT``.
    """
    if n_refines < 1:
        raise ValueError("n_refines must be >= 1.")

    working = ng.Mesh(coarse_mesh.ngmesh.Copy())
    levels: list[Level] = []
    pending_P = None

    if verbose:
        print(f"  {'lev':>3}  {'ndof':>7}  {'A':>13}  {'P(c->f)':>12}  {'nfree':>7}  {'nfixed':>6}")
        print(f"  {'---':>3}  {'-------':>7}  {'-------------':>13}  {'------------':>12}  {'-------':>7}  {'------':>6}")

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

        if verbose:
            a_shape = f"{a.mat.height}x{a.mat.width}"
            p_shape = f"{pending_P.height}x{pending_P.width}" if pending_P is not None else "-"
            tag = "  (coarse)" if lev == 0 else ("  (fine)" if lev == n_refines else "")
            print(f"  {lev:>3}  {fes.ndof:>7}  {a_shape:>13}  {p_shape:>12}  "
                  f"{len(level.free_ids):>7}  {len(level.fixed_ids):>6}{tag}")

        if lev < n_refines:
            working.Refine()
            fes_next = H1(working, order=order, dirichlet=dirichlet)
            pending_P = fes_next.Prolongation().CreateMatrix(working.levels - 1)

    return MultigridHierarchy(levels)


# Backwards-compatible alias for the name used in the test script.
build_multilevel_data = build_hierarchy


# ---------------------------------------------------------------------------
# V-cycle solver
# ---------------------------------------------------------------------------
@dataclass
class VCycleConfig:
    """Smoothing/coarse-solve configuration shared across levels."""

    smoother: SmootherKind = "native"
    pre_sweeps: int = 2
    post_sweeps: int = 2
    omega: float = 1.0
    coarse_direct: bool = True
    coarse_sweeps: int = 20  # used only when coarse_direct is False


class MultigridSolver:
    """Recursive V-cycle solver / preconditioner over a :class:`MultigridHierarchy`."""

    def __init__(self, hierarchy: MultigridHierarchy,
                 config: Optional[VCycleConfig] = None) -> None:
        self.h = hierarchy
        self.cfg = config or VCycleConfig()

    # -- core recursion -----------------------------------------------------
    def v_cycle(self, idx: int, b, x, *, verbose: bool = False) -> None:
        """Approximately solve ``A_idx x = b`` in place (one V-cycle)."""
        level = self.h.levels[idx]

        if idx == self.h.coarsest_idx:
            if self.cfg.coarse_direct:
                level.coarse_solve(b, x)
            else:
                level.smooth(b, x, kind=self.cfg.smoother,
                             nsweeps=self.cfg.coarse_sweeps,
                             omega=self.cfg.omega, verbose=verbose)
            return

        level.smooth(b, x, kind=self.cfg.smoother,
                     nsweeps=self.cfg.pre_sweeps, omega=self.cfg.omega, verbose=verbose)

        r = level.residual(b, x)
        r.FV().NumPy()[level.fixed_ids] = 0.0

        coarse = self.h.levels[idx - 1]
        r_c = level.PT.CreateColVector()
        r_c.data = level.PT * r

        e_c = r_c.CreateVector()
        e_c.FV().NumPy()[:] = 0.0
        self.v_cycle(idx - 1, r_c, e_c, verbose=verbose)

        e_f = level.P.CreateColVector()
        e_f.data = level.P * e_c
        x.data += e_f

        level.smooth(b, x, kind=self.cfg.smoother,
                     nsweeps=self.cfg.post_sweeps, omega=self.cfg.omega, verbose=verbose)

    # -- drivers ------------------------------------------------------------
    def solve(self, *, max_cycles: int = 20, tol: float = 1e-10,
              verbose: bool = False) -> list[float]:
        """Iterate V-cycles on the finest level's real problem ``A x = f``.

        Dirichlet data is taken from ``finest.gfu`` (set it via
        ``finest.set_initial_guess`` or ``finest.enforce_dirichlet`` first).
        Returns the history of free-residual norms.
        """
        fine = self.h.finest
        x = fine.gfu.vec
        b = fine.f.vec
        fine.enforce_dirichlet(x)

        hist: list[float] = []
        r0 = fine.free_residual_norm(b, x)
        if verbose:
            print(f"cycle {0:3d}  ||r_free|| = {r0:.6e}")
        for cyc in range(1, max_cycles + 1):
            self.v_cycle(self.h.finest_idx, b, x)
            fine.enforce_dirichlet(x)
            rn = fine.free_residual_norm(b, x)
            hist.append(rn)
            if verbose:
                rate = rn / hist[-2] if len(hist) > 1 else rn / r0
                print(f"cycle {cyc:3d}  ||r_free|| = {rn:.6e}  (rate {rate:.3f})")
            if rn <= tol:
                break
        return hist

    def apply(self, rhs, *, x=None):
        """Preconditioner action ``z = M^{-1} rhs`` (one V-cycle from zero).

        Does not touch any level's load vector. Returns the correction ``z`` as
        a fresh NGSolve vector on the finest level.
        """
        fine = self.h.finest
        z = fine.f.vec.CreateVector() if x is None else x
        z.FV().NumPy()[:] = 0.0
        self.v_cycle(self.h.finest_idx, rhs, z)
        return z

    def pcg(self, *, max_iter: int = 100, tol: float = 1e-10,
            verbose: bool = False) -> list[float]:
        """Preconditioned CG on the finest problem with this MG as preconditioner.

        Assumes homogeneous Dirichlet data (correction-scheme friendly). Returns
        the history of preconditioned-residual norms ``sqrt(r . z)``.
        """
        fine = self.h.finest
        A = fine.a.mat
        x = fine.gfu.vec
        fine.enforce_dirichlet(x)

        r = fine.residual(fine.f.vec, x)
        r.FV().NumPy()[fine.fixed_ids] = 0.0
        z = self.apply(r)
        p = z.CreateVector()
        p.data = z
        rz = float(ng.InnerProduct(r, z))

        hist: list[float] = []
        for it in range(1, max_iter + 1):
            Ap = p.CreateVector()
            Ap.data = A * p
            Ap.FV().NumPy()[fine.fixed_ids] = 0.0
            alpha = rz / float(ng.InnerProduct(p, Ap))
            x.data += alpha * p
            r.data -= alpha * Ap
            r.FV().NumPy()[fine.fixed_ids] = 0.0

            rn = float(np.linalg.norm(r.FV().NumPy()[fine.free_ids]))
            hist.append(rn)
            if verbose:
                print(f"pcg {it:3d}  ||r_free|| = {rn:.6e}")
            if rn <= tol:
                break

            z = self.apply(r)
            rz_new = float(ng.InnerProduct(r, z))
            beta = rz_new / rz
            p.data = z + beta * p
            rz = rz_new
        return hist
