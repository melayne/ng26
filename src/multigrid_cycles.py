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
    # tol is a *relative* reduction factor: stop when ||r|| <= tol * ||r0||.
    res, level_res = solver.solve(max_cycles=20, tol=1e-10, verbose=True)
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
    vector_norm as _vector_norm,
)
from preconditioners import gauss_seidel_sweeps  # noqa: E402

FormSetupFn = Callable[[object], "tuple[BilinearForm, LinearForm]"]
SmootherKind = Literal["gs", "native"]
# Norm for residual measurement: Euclidean, energy (A), or mass/L2 (M).
# A NGSolve matrix/operator B may also be passed for a generic sqrt(r^T B r).
NormKind = Literal["l2", "euclidean", "A", "energy", "M", "mass"]


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
    @staticmethod
    def transfer_available(fes) -> bool:
        """True if ``fes`` can build a prolongation to the immediately coarser level.

        This holds exactly when the underlying mesh has been refined at least
        once (``fes.mesh.levels > 1``). Note that ``fes.Prolongation()`` itself is
        not a usable signal: it always returns a ``Prolongation`` object, and
        ``CreateMatrix`` on an unrefined mesh silently returns a width-0 matrix
        rather than raising.
        """
        return fes.mesh.levels > 1

    @classmethod
    def from_forms(cls, mesh, fes, a, f, *, P=None, PT=None,
                   gfu=None, dirichlet_value=0.0, dirichlet="",
                   built_P: bool = True) -> "Level":
        """Build a :class:`Level` from assembled forms.

        The prolongation ``P`` (coarse->fine) and its transpose ``PT``
        (fine->coarse) are populated automatically: if ``P`` is not given and
        ``built_P`` is true and the space supports it
        (:meth:`transfer_available`), the prolongation to the immediately coarser
        level is built from ``fes.Prolongation()``. ``PT`` is derived as
        ``P.CreateTranspose()`` whenever ``P`` is available and ``PT`` was not
        supplied. Pass ``built_P=False`` (or explicit ``P``/``PT``) to override
        -- e.g. to force a coarsest level on an already-refined mesh.
        """
        if gfu is None:
            gfu = GridFunction(fes)
        free_ids, fixed_ids = get_free_fixed_ids(fes)
        if P is None and built_P and cls.transfer_available(fes):
            P = fes.Prolongation().CreateMatrix(fes.mesh.levels - 1)
        if PT is None and P is not None:
            PT = P.CreateTranspose()
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

    def vector_norm(self, r, *, norm: "NormKind | object" = "energy") -> float:
        """Norm of a vector over this level's free DOFs.

        ``norm`` selects the metric:

        * ``"l2"`` / ``"euclidean"`` -- Euclidean norm ``sqrt(r . r)`` (no mat-vec).
        * ``"A"`` / ``"energy"``     -- energy norm ``sqrt(r^T A r)`` (stiffness).
        * ``"M"`` / ``"mass"``       -- mass / discrete-``L2`` norm ``sqrt(r^T M r)``.
        * any NGSolve matrix/operator ``B`` -- generic ``sqrt(r^T B r)``.

        For the weighted norms ``r`` must already be zero on fixed DOFs (the
        residuals computed here are), so only free DOFs contribute. The weighted
        norms cost one extra matrix-vector product; ``"l2"`` costs none.

        This only resolves ``norm`` to a metric matrix; the actual computation
        is the stateless ``NGSolve_utils.vector_norm(vec, mat)``.
        """
        if isinstance(norm, str):
            key = norm.lower()
            if key in ("l2", "euclidean", "2"):
                return _vector_norm(r, free_ids=self.free_ids)
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
            op = norm  # assume a BaseMatrix-like operator with `op * r`

        return _vector_norm(r, op)

    def residual_norm(self, b=None, x=None, *, norm: "NormKind | object" = "energy") -> float:
        """Norm of the residual ``b - A x`` over free DOFs (see :meth:`vector_norm`)."""
        b = self.f.vec if b is None else b
        x = self.gfu.vec if x is None else x
        r = self.residual(b, x)
        r.FV().NumPy()[self.fixed_ids] = 0.0
        return self.vector_norm(r, norm=norm)

    # -- solves / smoothing -------------------------------------------------
    def coarse_solve(self, b, x) -> None:
        """Exact solve ``A x = b`` on free DOFs (fixed DOFs of ``x`` untouched)."""
        x.data = self.a.mat.Inverse(self.fes.FreeDofs()) * b

    def smooth(self, b, x, *, kind: str = "native",
               nsweeps: int = 2, omega: float = 1.0, verbose: bool = False,
               backward: bool = False) -> None:
        """Relax ``A x = b`` in place on free DOFs using the chosen smoother.

        ``backward=True`` uses a backward Gauss-Seidel sweep (post-smoothing leg).
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
        sm = self.smoother
        x0 = x.CreateVector()
        for sweep in range(1, nsweeps + 1):
            if backward:
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

    def info(self) -> None:
        """Print a compact per-level hierarchy summary table."""
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
    """Refine ``coarse_mesh`` ``n_refines`` times and assemble one level per mesh.

    Produces ``n_refines + 1`` levels ordered coarse -> fine. Each non-coarsest
    level gets the prolongation ``P`` (coarse -> fine) and its transpose ``PT``.
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
    def _smooth_record(self, level, b, x, nsweeps, norm, *, verbose=False,
                       backward: bool = False):
        """Smooth one sweep at a time, recording the free-residual norm after each.

        Returns a list of length ``nsweeps`` (the residual ``norm`` after sweep
        1, 2, ..., nsweeps). Costs one extra residual evaluation per sweep, so it
        is only used when level recording is active.
        """
        out: list[float] = []
        for _ in range(nsweeps):
            level.smooth(b, x, kind=self.cfg.smoother, nsweeps=1,
                         omega=self.cfg.omega, verbose=verbose, backward=backward)
            r = level.residual(b, x)
            r.FV().NumPy()[level.fixed_ids] = 0.0
            out.append(level.vector_norm(r, norm=norm))
        return out

    def v_cycle(self, idx: int, b, x, *, verbose: bool = False,
                rec_cycle: Optional[dict] = None,
                rec: Optional[dict] = None,
                norm: "NormKind | object" = "energy",
                debug: bool = False) -> None:
        """Approximately solve ``A_idx x = b`` in place (one V-cycle).

        If ``rec_cycle`` is a dict, one residual norm per level (in ``norm``) is
        stored as ``rec_cycle[idx]``: for the coarsest level it is measured
        *before* the coarse solve; on every other level it is measured *after
        pre-smoothing* (the residual that gets restricted). This reuses work the
        cycle already performs and is cheap.

        If ``rec`` is a dict, per-sweep norms are stored as
        ``rec[idx] = {"down": [...], "up": [...]}`` (see ``record_levels`` in
        :meth:`solve`). This smooths one sweep at a time and costs extra mat-vecs.

        If ``debug`` is true, prints the current level and the shapes of the
        vectors/operators involved (``x``, ``b``, ``r``, ``PT``, ``r_c``, ``P``),
        indented by depth, to trace the transfer dimensions on both legs.
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
                    before = level.residual_norm(b, x, norm=norm)
                    if rec_cycle is not None:
                        rec_cycle[idx] = before
                level.coarse_solve(b, x)
                if rec is not None:
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
              norm: "NormKind | object" = "energy", debug: bool = False):
        """Iterate V-cycles on the finest level's real problem ``A x = f``.

        Uses ``self.h.finest`` for ``b = f.vec`` and ``x = gfu.vec``. Smoothing
        counts and the coarse solver come from ``self.cfg`` (:class:`VCycleConfig`),
        set when constructing :class:`MultigridSolver`.

        Before calling, set the initial iterate and Dirichlet data on the finest
        level, e.g. ``finest.set_initial_guess(...)`` or
        ``finest.enforce_dirichlet(finest.gfu.vec)``.

        Parameters
        ----------
        max_cycles : int, default 20
            Maximum number of V-cycles to apply. The loop may stop earlier if
            ``tol`` is satisfied.
        tol : float, default 1e-10
            Relative residual tolerance. Stops when the finest-level residual
            (in ``norm``) satisfies ``||r|| <= tol * ||r0||``, where ``||r0||``
            is measured before the first cycle. Use ``tol=0.0`` to always run
            exactly ``max_cycles`` cycles.
        verbose : bool, default False
            If True, print the initial residual and one line per cycle with
            finest-level ``||r||`` and contraction rate.
        record_levels : bool, default False
            If True, also record per-sweep residual norms on each level during
            each cycle (extra mat-vecs; see ``sweep_hist``). If False, only the
            cheap per-level snapshots in ``level_hist`` are recorded.
        norm : str or matrix-like, default ``"energy"``
            Norm for convergence and all recorded residuals. Passed to
            :meth:`Level.residual_norm` / :meth:`Level.vector_norm`: ``"l2"``,
            ``"energy"`` (alias ``"A"``), ``"mass"`` (alias ``"M"``), or a custom
            operator understood by those methods.
        debug : bool, default False
            If True, print a dimension trace for the **first** V-cycle only
            (level indices and vector/operator shapes during restriction and
            prolongation); see :meth:`v_cycle`.

        Returns
        -------
        hist : list[float]
            Finest-level residual norm after each V-cycle (same ``norm``).
        level_hist : list[list[float]]
            One residual norm per level per cycle, measured during the cycle
            (coarsest: before the coarse solve; other levels: after pre-smoothing).
            ``level_hist[idx][c]`` is cycle ``c`` (0-based).
        sweep_hist : list[list[dict]] | None
            Present only when ``record_levels=True``. ``sweep_hist[idx][c]`` is
            ``{"down": [...], "up": [...]}`` of per-sweep norms on level ``idx``
            during cycle ``c`` (see :meth:`v_cycle`).
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
