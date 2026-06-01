"""
Demo: using multigrid_tools_temp.py for
  (1) a single-level smoother (no coarse correction), and
  (2) a single V-cycle (and, for context, a full multi-cycle solve).

Model problem (2D Poisson, homogeneous Dirichlet on all sides):
    -Laplace(u) = f   on the unit square,  u = 0 on the boundary

Exact solution (polynomial, not sinusoidal):
    u = x(1-x) y(1-y) (2x + 3y - xy)
Asymmetric hill with curved level sets — zero on the boundary, not separable.

Initial guess u0 = 0: smooth global error; section (1) shows smoother stall,
section (2) shows V-cycle correction via the hierarchy.

Other smooth polynomial options (swap u_exact / rhs_cf):
  - flat-top:  u = 256 x²(1-x)² y²(1-y)²
  - ridge:     u = 16 x(1-x) y²(1-y)²
  - two peaks: u = 16 x(1-x) y(1-y) [(x-0.35)²+(y-0.5)²][(x-0.65)²+(y-0.5)²]

Run as a script, or step through the `#%%` cells interactively.
"""
#%%
import os
import sys

import ngsolve as ng
from ngsolve import H1, InnerProduct, Mesh, grad, dx, x, y, sin, pi, BaseMatrix, solvers
from netgen.geom2d import unit_square
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from multigrid_cycles import (
    build_form_setup,
    build_hierarchy,
    Level,
    MultigridSolver,
    VCycleConfig,
)

# ---------------------------------------------------------------------------
# Shared problem definition
# ---------------------------------------------------------------------------
DIRICHLET = "left|right|top|bottom"

# u = A(x) B(y) C(x,y),  A=x(1-x), B=y(1-y), C=2x+3y-xy  (zero on boundary)
_A = x * (1 - x)
_B = y * (1 - y)
_C = 2 * x + 3 * y - x * y
u_exact = _A * _B * _C
# -Laplace(u) via product rule (C_xx = C_yy = 0)
_u_xx = (-2) * _B * _C + 2 * (1 - 2 * x) * _B * (2 - y)
_u_yy = (-2) * _A * _C + 2 * _A * (1 - 2 * y) * (3 - x)
rhs_cf = -(_u_xx + _u_yy)

# Zero initial guess: smooth global error, no injected oscillations.
x0_slow = sin(pi*x) * sin(pi*y)
x0_fast = 0.5 * sin(30*pi*x/2) * sin(30*pi*y/2)
x0_initial = x0_slow + x0_fast
x0_cf = x0_initial


def poisson_bilinear(a, u, v):
    a += InnerProduct(grad(u), grad(v)) * dx


def poisson_linear(f, u, v):
    f += rhs_cf * v * dx


poisson_setup = build_form_setup(bilinear=poisson_bilinear, linear=poisson_linear)


def l2_error(level) -> float:
    """L2 norm of (current iterate - exact solution) on a level's mesh."""
    return float(ng.sqrt(ng.Integrate((level.gfu - u_exact) ** 2 * dx, level.mesh)))


def l2_residual(level, b=None, x=None) -> float:
    """Discrete residual norm ``||r||_2`` for ``r = b - A x`` on free DOFs."""
    return level.residual_norm(b, x, norm="l2")


def energy_error(level) -> float:
    """Discrete energy error ``sqrt(e_vec^T A e_vec)`` for ``e = u_h - u_exact``."""
    gfu_exact = ng.GridFunction(level.fes)
    gfu_exact.Set(u_exact)

    err_vec = level.gfu.vec.CreateVector()
    err_vec.data = level.gfu.vec - gfu_exact.vec
    return level.vector_norm(err_vec, norm="energy")


def energy_residual(level, b=None, x=None) -> float:
    """Discrete energy residual norm ``sqrt(r^T A r)`` on free DOFs."""
    return level.residual_norm(b, x, norm="energy")


def plot_energy_residual_vs_cycle(
    hist,
    *,
    r0: float | None = None,
    ax=None,
    title: str | None = None,
    show: bool = True,
):
    """Finest-level energy residual vs cycle (end-of-cycle ``hist`` only)."""
    ncyc = len(hist)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    cycles = np.arange(1, ncyc + 1, dtype=float)
    ax.semilogy(cycles, hist, "o-", ms=7, lw=1.5,
                label=r"$\|r\|_E$ after cycle")

    if r0 is not None:
        ax.semilogy([0.0], [r0], "s", ms=8, label="initial")

    # for k in range(1, ncyc):
    #     ax.axvline(k + 0.5, ls=":", color="0.35", lw=1.2, zorder=1)
    # if ncyc > 1:
    #     ax.plot([], [], ":", color="0.35", label="new V-cycle (down)")

    ax.set_xlim(-0.15, ncyc + 0.15)
    ax.set_xticks(np.arange(0, ncyc + 1))
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\|r\|_E$ (finest)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if show:
        plt.show()
    return ax


# ===========================================================================
# (1) SINGLE LEVEL: just relax A x = f on one grid (no coarse correction).
#    Smooth error (u0=0): residual drops slowly and stalls — MG motivation.
# ===========================================================================
#%%
print("=" * 70)
print("(1) Single-level smoother")
print("=" * 70)

_base = unit_square.GenerateMesh(maxh=0.05)
_working = Mesh(_base.Copy())
mesh_c_parent = Mesh(_working.ngmesh.Copy())   # coarse parent (section 2)
_working.Refine()
mesh = _working
fes = H1(mesh, order=1, dirichlet=DIRICHLET)
a, f = poisson_setup(fes)
a.Assemble()
f.Assemble()

poisson_level = Level.from_forms(mesh, fes, a, f, dirichlet_value=0.0, dirichlet=DIRICHLET)

# Set zero initial guess (enforce_bc pins Dirichlet DOFs to 0).
poisson_level.set_initial_guess(x0_cf, enforce_bc=True)

Draw(u_exact, poisson_level.mesh, "exact solution")

print()
_scene = Draw(
    poisson_level.gfu,
    poisson_level.mesh,
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
        "Misc": {"line_thickness": 0.01},
        "Objects": {"Wireframe": False, "Surface": True},
    },
)

n_sweeps = 10
gs_residual: list[float] = []
gs_l2_err: list[float] = []

# Live residual + L2 error plot.
# Jupyter inline backend ignores plt.ion()/pause(); use display_id updates instead.
# Optional (smoother): put `%matplotlib widget` in a cell above this one.
try:
    from IPython.display import display
    from IPython import get_ipython
    _ipython = get_ipython() is not None
except ImportError:
    display = None
    _ipython = False

fig, (ax_r, ax_e) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
(line_r,) = ax_r.plot([], [], "o-", markersize=7, linewidth=1, label=r"$\|r\|_E$")
(line_e,) = ax_e.plot([], [], "s-", markersize=6, linewidth=1, label=r"$L2(u-u_{exact})$")
ax_r.set_ylabel(r"$\| \, r \, \|_A$")
ax_r.set_title("Gauss-Seidel smoothing (10 sweeps)")
ax_r.set_xlim(0.5, n_sweeps + 0.5)
ax_r.legend(loc="upper right")
ax_r.grid(True, alpha=0.3)
ax_e.set_xlabel("Number of sweeps")
ax_e.set_ylabel(r"$L2 Error$" + "\n" + r"$(u-u_{exact})$")
ax_e.set_xlim(0.5, n_sweeps + 0.5)
ax_e.set_xticks(range(1, n_sweeps + 1))
ax_e.legend(loc="upper right")
ax_e.grid(True, alpha=0.3)
plt.setp(ax_r.get_xticklabels(), visible=False)
fig.tight_layout()

_plot_handle = display(fig, display_id="gs_residual_plot") if _ipython else None
if not _ipython:
    plt.ion()
    fig.show()


def _update_gs_plot(*, xlabels: list[str] | None = None, title: str | None = None) -> None:
    """Refresh the section (1) plot from ``gs_residual`` / ``gs_l2_err``."""
    n = len(gs_residual)
    x_pts = list(range(1, n + 1))
    if xlabels is not None:
        ax_r.set_xlim(0.5, n + 0.5)
        ax_e.set_xlim(0.5, n + 0.5)
        ax_e.set_xticks(x_pts)
        ax_e.set_xticklabels(xlabels, rotation=12, ha="right")
    if title is not None:
        ax_r.set_title(title)
    line_r.set_data(x_pts, gs_residual)
    line_e.set_data(x_pts, gs_l2_err)
    ax_r.relim()
    ax_r.autoscale_view(scalex=False)
    ax_e.relim()
    ax_e.autoscale_view(scalex=False)
    if _plot_handle is not None:
        _plot_handle.update(fig)
    else:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.5)


for sweep in range(1, n_sweeps + 1):
    poisson_level.smooth(poisson_level.f.vec, poisson_level.gfu.vec,
                         kind="native", nsweeps=1, omega=1.0)
    _scene.Redraw()
    gs_residual.append(poisson_level.residual_norm(norm="energy"))
    gs_l2_err.append(l2_error(poisson_level))
    _update_gs_plot()
    if _plot_handle is not None:
        time.sleep(0.5)

if not _ipython:
    plt.ioff()
    plt.show()

# ===========================================================================
# (2) Coarse Grid Correction
# ===========================================================================

# Coarse mesh must be the *parent* of poisson_level.mesh (same base as section 1).
mesh_c = mesh_c_parent
fes_c = H1(mesh_c, order=1, dirichlet=DIRICHLET)
a_c, f_c = poisson_setup(fes_c)
a_c.Assemble()
f_c.Assemble()

poisson_level_c = Level.from_forms(
    mesh_c, fes_c, a_c, f_c,
    dirichlet_value=0.0, dirichlet=DIRICHLET,
    built_P=False,
)

b = poisson_level.f.vec
x = poisson_level.gfu.vec
poisson_level.enforce_dirichlet(x)

# P / PT from the fine space (coarse -> fine / fine -> coarse).
P = poisson_level.fes.Prolongation().CreateMatrix(poisson_level.mesh.levels - 1)
PT = P.CreateTranspose()

print(f"  fine ndof={poisson_level.ndof}, coarse ndof={poisson_level_c.ndof}")
print(f"  P shape {P.height}x{P.width}")
print(f"  ||r||_E (after section 1 pre-smooth): "
      f"{poisson_level.residual_norm(b, x, norm='energy'):.6e}")

# restrict: r_c = P^T r
r = poisson_level.residual(b, x)
r.FV().NumPy()[poisson_level.fixed_ids] = 0.0
r_c = PT.CreateColVector()
r_c.data = PT * r

# coarse solve: A_c e_c = r_c
e_c = r_c.CreateVector()
e_c.FV().NumPy()[:] = 0.0
poisson_level_c.coarse_solve(r_c, e_c)

# prolong + add: x += P e_c
e_f = P.CreateColVector()
e_f.data = P * e_c
x.data += e_f
poisson_level.enforce_dirichlet(x)

r_cgc = poisson_level.residual_norm(b, x, norm="energy")
err_cgc = l2_error(poisson_level)
print(f"  ||r||_E after coarse correction:     {r_cgc:.6e}")
print(f"  L2 error:                            {err_cgc:.6e}")

gs_residual.append(r_cgc)
gs_l2_err.append(err_cgc)
_cgc_labels = [str(i) for i in range(1, n_sweeps + 1)] + ["CGC"]
_update_gs_plot(xlabels=_cgc_labels, title="Gauss-Seidel + coarse correction")

_scene.Redraw()
plt.close(fig)

#%%
# ===========================================================================
# (2) SINGLE V-CYCLE: build a hierarchy, then run exactly one cycle.
# ===========================================================================

print()
print("=" * 70)
print("(2) Single V-cycle on a 3-level hierarchy")
print("=" * 70)

mesh_c = Mesh(unit_square.GenerateMesh(maxh=0.05))
hierarchy = build_hierarchy(
    mesh_c,
    poisson_setup,
    n_refines=2,                 # -> 3 levels (coarse, mid, fine)
    order=1,
    dirichlet=DIRICHLET,
    dirichlet_value=0.0,
    verbose=True,
)

cfg = VCycleConfig(
    smoother="native",
    pre_sweeps=1,
    post_sweeps=1,
    omega=1.0,
    coarse_direct=True,
)
solver = MultigridSolver(hierarchy, cfg)

fine = hierarchy.finest
fine.set_initial_guess(x0_cf, enforce_bc=True)
print(f"\n  before cycle  ||r_free||_A = {fine.residual_norm():.6e}"
      f"   L2 err = {l2_error(fine):.6e}")

# --- Option A: the driver, capped to a single cycle (sets BC, returns history)
r0 = fine.residual_norm()
hist, level_res = solver.solve(max_cycles = 25, tol=1e-10, verbose=False, record_levels=False)
num_cycles = len(hist)
print(f"  after {num_cycles} cycles $||r_free||_A = {hist[-1]:.6e}"
      f"   L2 err = {l2_error(fine):.6e}")

plot_energy_residual_vs_cycle(hist, 
                              r0=r0, 
                              title = f"Single V-cycle on a 3-level hierarchy \n tol = 1e-10"
                            )

#%%

# ===========================================================================
# Num cycles vs ndof for a fixed tolerance
# 
# ===========================================================================

print()
print("=" * 70)
print("Num cycles vs ndof for a fixed tolerance")
print("=" * 70)

TOL = 1e-10
study_cfg = VCycleConfig(
    smoother="native", pre_sweeps=1, post_sweeps=1, omega=1.0, coarse_direct=True,
)
# Smooth x0 so ||r0||_A is comparable across refinements (HF x0_cf skews cycle counts).
x0_study = 0.0

mesh_c = Mesh(unit_square.GenerateMesh(maxh=0.1))
results: list[tuple[int, int, float, float, bool]] = []

print(f"  {'n_ref':>5}  {'ndof':>8}  {'cycles':>6}  {'time(s)':>9}  "
      f"{'||r0||':>10}  {'||r_fin||':>10}  {'tol||r0||':>10}  ok")
for n_refines in [1, 2, 3, 4, 5]:
    hierarchy = build_hierarchy(
        mesh_c, poisson_setup, n_refines=n_refines,
        order=1, dirichlet=DIRICHLET, dirichlet_value=0.0,
    )
    fine = hierarchy.finest
    solver = MultigridSolver(hierarchy, study_cfg)
    fine.set_initial_guess(x0_study, enforce_bc=True)
    r0 = fine.residual_norm(norm="energy")
    t0 = time.perf_counter()
    hist, _ = solver.solve(max_cycles=100, tol=TOL, norm="energy")
    elapsed = time.perf_counter() - t0
    r_fin = hist[-1]
    ok = r_fin <= TOL * r0
    results.append((fine.ndof, len(hist), r0, r_fin, ok))
    print(f"  {n_refines:5d}  {fine.ndof:8d}  {len(hist):6d}  {elapsed:9.3f}  "
          f"{r0:10.3e}  {r_fin:10.3e}  {TOL * r0:10.3e}  {ok}")

ndofs, n_cycles, *_ = zip(*results)
plt.figure(figsize=(7, 4))
plt.semilogx(ndofs, n_cycles, "o-")
plt.xlabel("fine ndof")
plt.ylabel("V-cycles to reach tol")
plt.title(fr"fixed tol = {TOL:g}, $\|r\|_A$ norm, $x_0=0$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




#%%
from ngsolve import BaseMatrix, solvers  # type: ignore[import-untyped]

class VCyclePreconditioner(BaseMatrix):
    """y = M^{-1} x  via one V-cycle on A z = x, z0 = 0."""
    def __init__(self, mg_solver, level):
        super().__init__()
        self.solver = mg_solver
        self.level = level
        self.idx = mg_solver.h.finest_idx

    def IsComplex(self):
        return False

    def Shape(self):
        n = len(self.level.gfu.vec)
        return (n, n)

    def CreateVector(self, col):
        return self.level.gfu.vec.CreateVector()

    def Mult(self, x, y):
        y.FV().NumPy()[:] = 0.0
        self.solver.v_cycle(self.idx, x, y)
        self.level.enforce_dirichlet(y)
        y.FV().NumPy()[self.level.fixed_ids] = 0.0

# --- after you built hierarchy + MultigridSolver(hierarchy, study_cfg) ---
fine = hierarchy.finest
a, f = fine.a, fine.f   # BilinearForm / LinearForm on finest

pre = VCyclePreconditioner(solver, fine)
gfu = fine.gfu
gfu.vec[:] = 0.0
fine.set_initial_guess(x0_study, enforce_bc=True)  # or leave 0

inv = solvers.CGSolver(mat=a.mat, pre=pre, maxiter=200, tol=1e-10, printrates=True)
gfu.vec.data = inv * f.vec

n_cg = inv.iterations
print("PCG (NGSolve CG + your V-cycle):", n_cg, "iterations")








# #%%
# # ===========================================================================
# # (3) Per-level residual each cycle (default) + optional per-sweep detail.
# # ===========================================================================

# print()
# print("=" * 70)
# print("(3) Per-level residual each cycle (default recording)")
# print("=" * 70)

# fine.set_initial_guess(x0_cf, enforce_bc=True)
# r0_energy = fine.residual_norm(norm="energy")
# hist, level_hist = solver.solve(max_cycles=8, tol=1e-12, norm="energy")

# ncyc = len(hist)
# plot_energy_residual_vs_cycle(
#     hist, r0=r0_energy, title="Finest energy residual vs cycle",
# )

# header = "  cycle  " + "".join(f"level {i:>2}     " for i in range(hierarchy.nlevels))
# print(header)
# for c in range(ncyc):
#     row = "".join(f"{level_hist[i][c]:.4e}  " for i in range(hierarchy.nlevels))
#     print(f"  {c + 1:>4}   {row}")
# print("\n  hist[c] = finest residual after cycle c+1.")
# print("  level_hist[idx][c] = residual on level idx during cycle c+1")
# print("  (coarsest before direct solve; others after pre-smoothing).")

# print()
# print("=" * 70)
# print("(3b) Per-sweep detail (record_levels=True)")
# print("=" * 70)

# fine.set_initial_guess(x0_cf, enforce_bc=True)
# hist, level_hist, sweep_hist = solver.solve(
#     max_cycles=8, tol=1e-12, record_levels=True, norm="energy",
# )

# ncyc = len(hist)

# # Per-level residual after pre-smooth / before coarse solve (one point per cycle).
# L = np.asarray(level_hist).T
# plt.figure(figsize=(7, 4))
# plt.semilogy(np.arange(1, ncyc + 1), L, "o-")
# plt.xlabel("cycle")
# plt.ylabel(r"$\|r\|_E$")
# plt.legend([f"level {i}" for i in range(L.shape[1])])
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


# def _fmt(seq):
#     return "[" + ", ".join(f"{v:.2e}" for v in seq) + "]"


# for c in range(ncyc):
#     print(f"\n  cycle {c + 1}:")
#     for idx in range(hierarchy.nlevels - 1, -1, -1):  # finest -> coarsest
#         rec = sweep_hist[idx][c]
#         tag = "(fine)" if idx == hierarchy.finest_idx else ("(coarse)" if idx == 0 else "")
#         print(f"    level {idx} {tag:>8}  down={_fmt(rec['down'])}  up={_fmt(rec['up'])}")

# print("\n  down[k] = residual after pre-smooth sweep k (descent);")
# print("  up[k]   = residual after post-smooth sweep k (ascent).")
# print("  coarsest: down=[before direct solve], up=[after direct solve ~ 0].")

# # %%

# import os
# import sys

# import ngsolve as ng
# from ngsolve import H1, InnerProduct, Mesh, grad, dx, x, y, sin, pi
# from netgen.geom2d import unit_square

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# from multigrid_cycles import (
#     build_form_setup,
#     build_hierarchy,
#     Level,
#     MultigridSolver,
#     VCycleConfig,
# )

# # ---------------------------------------------------------------------------
# # Shared problem definition
# # ---------------------------------------------------------------------------
# DIRICHLET = "left|right|top|bottom"
# u_exact = sin(pi * x) * sin(pi * y)
# rhs_cf = 2 * pi * pi * sin(pi * x) * sin(pi * y)   # = -Laplace(u_exact)

# # An initial guess that mixes a smooth mode with a rough (high-frequency) mode,
# # so the effect of smoothing vs. a full V-cycle is visible.
# x0_cf = sin(pi * x) * sin(pi * y) + 0.3 * sin(6 * pi * x) * sin(6 * pi * y)


# def poisson_bilinear(a, u, v):
#     a += InnerProduct(grad(u), grad(v)) * dx


# def poisson_linear(f, u, v):
#     f += rhs_cf * v * dx


# poisson_setup = build_form_setup(bilinear=poisson_bilinear, linear=poisson_linear)


# def l2_error(level) -> float:
#     """L2 norm of (current iterate - exact solution) on a level's mesh."""
#     return float(ng.sqrt(ng.Integrate((level.gfu - u_exact) ** 2 * dx, level.mesh)))


# # ===========================================================================
# # (2) SINGLE V-CYCLE: build a hierarchy, then run exactly one cycle.
# # ===========================================================================
# print()
# print("=" * 70)
# print("(2) Single V-cycle on a 3-level hierarchy")
# print("=" * 70)

# mesh_c = Mesh(unit_square.GenerateMesh(maxh=0.05))
# hierarchy = build_hierarchy(
#     mesh_c,
#     poisson_setup,
#     n_refines=2,                 # -> 3 levels (coarse, mid, fine)
#     order=1,
#     dirichlet=DIRICHLET,
#     dirichlet_value=0.0,
#     verbose=True,
# )
# # %%

# %%
