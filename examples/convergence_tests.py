"""
Demo: using ``multigrid_cycles`` for
  (1) a single-level smoother (no coarse correction), and
  (2) V-cycles / tolerance studies on a built hierarchy.

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
from matplotlib.colors import Colormap
import matplotlib as mpl
from cycler import cycler
import time
import numpy as np
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.multigrid_cycles import (
    build_form_setup,
    build_hierarchy,
    Level,
    MultigridSolver,
    VCycleConfig,
)

# ---------------------------------------------------------------------------
# Get and set colors for plotting
# ---------------------------------------------------------------------------
cmap = plt.get_cmap("jet")
cmap_nums = [tuple(cmap(x)[:3]) for x in np.linspace(0, 1)]
plt.rcParams["image.cmap"] = "jet"
# .colors is the full LUT (hundreds of nearly identical dark entries) — sample
# evenly for distinct line/marker colors in multi-series plots.
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[cmap(x) for x in np.linspace(0, 1, 10, endpoint=False)]
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


def l2_error(u_exact, level) -> float:
    """L2 norm of (current iterate - exact solution) on a level's mesh."""
    return float(ng.sqrt(ng.Integrate((level.gfu - u_exact) ** 2 * dx, level.mesh)))


def l2_residual(level, b=None, x=None) -> float:
    """Discrete residual norm ``||r||_2`` for ``r = b - A x`` on free DOFs."""
    return level.residual_norm(b, x)


def energy_error(u_exact, level) -> float:
    """Energy error ``||e||_A = sqrt(e^T A e)`` for ``e = u_h - u_exact``."""
    return level.error_norm(u_exact=u_exact)


def hist_series(hist, key: str = "l2") -> list[float]:
    """One finest-level history from ``MultigridSolver.solve`` (``hist`` dict)."""
    if isinstance(hist, dict):
        return list(hist[key])
    return list(hist)


def plot_hist_vs_cycle(
    hist,
    *,
    key: str = "l2",
    r0: float | None = None,
    ax=None,
    title: str | None = None,
    ylabel: str | None = None,
    show: bool = True,
):
    """Plot ``hist[key]`` after each V-cycle (cycle index starts at 1)."""
    series = hist_series(hist, key)
    if ylabel is None:
        if key in ("l2", "euclidean", "2"):
            ylabel = r"$\|r\|_2$ (finest)"
        elif key in ("energy", "A"):
            ylabel = r"$\|u_{\mathrm{exact}} - u_h\|_A$ (finest)"
        elif key in ("update_dual", "dual", "mg-dual", "preconditioned"):
            ylabel = r"$\sqrt{r_{\mathrm{before}}^T \Delta x}$ (finest)"
        else:
            ylabel = f"{key} (finest)"

    ncyc = len(series)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    cycles = np.arange(1, ncyc + 1, dtype=float)
    ax.semilogy(cycles, series, "o-", ms=7, lw=1.5, label="after cycle")

    if r0 is not None:
        ax.semilogy([0.0], [r0], "s", ms=8, label="initial")

    ax.set_xlim(-0.15, ncyc + 0.15)
    ax.set_xticks(np.arange(0, ncyc + 1))
    ax.set_xlabel("cycle")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if show:
        plt.show()
    return ax


def plot_residual_vs_cycle(hist, **kwargs):
    """L2 finest residual history (alias for ``plot_hist_vs_cycle(..., key='l2')``)."""
    kwargs.setdefault("key", "l2")
    return plot_hist_vs_cycle(hist, **kwargs)


poisson_setup = build_form_setup(bilinear=poisson_bilinear, linear=poisson_linear)



#%%

# ===========================================================================
# (1) SINGLE LEVEL: just relax A x = f on one grid (no coarse correction).
#    Smooth error (u0=0): residual drops slowly and stalls — MG motivation.
# ===========================================================================
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

Draw(u_exact, poisson_level.mesh, "exact solution", colors=cmap_nums)

print()
_scene = Draw(
    poisson_level.gfu,
    poisson_level.mesh,
    "Gauss-Seidel smoothing",
    deformation=True,
    colors=cmap_nums,
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
gs_energy_err: list[float] = []
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

fig, (ax_eA, ax_l2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
(line_eA,) = ax_eA.plot([], [], "o-", markersize=7, linewidth=1, label=r"$\|e\|_A$")
(line_l2,) = ax_l2.plot([], [], "s-", markersize=6, linewidth=1, label=r"$L2(u-u_{exact})$")
ax_eA.set_ylabel(r"$\|e\|_A$")
ax_eA.set_title("Gauss-Seidel smoothing (10 sweeps)")
ax_eA.set_xlim(0.5, n_sweeps + 0.5)
ax_eA.legend(loc="upper right")
ax_eA.grid(True, alpha=0.3)
ax_l2.set_xlabel("Number of sweeps")
ax_l2.set_ylabel(r"$\|e\|_2$" + "\n" + r"$(u-u_{exact})$")
ax_l2.set_xlim(0.5, n_sweeps + 0.5)
ax_l2.set_xticks(range(1, n_sweeps + 1))
ax_l2.legend(loc="upper right")
ax_l2.grid(True, alpha=0.3)
plt.setp(ax_eA.get_xticklabels(), visible=False)
fig.tight_layout()

_plot_handle = display(fig, display_id="gs_smoothing_plot") if _ipython else None
if not _ipython:
    plt.ion()
    fig.show()


def _update_gs_plot(*, xlabels: list[str] | None = None, title: str | None = None) -> None:
    """Refresh the section (1) plot from ``gs_energy_err`` / ``gs_l2_err``."""
    n = len(gs_energy_err)
    x_pts = list(range(1, n + 1))
    if xlabels is not None:
        ax_eA.set_xlim(0.5, n + 0.5)
        ax_l2.set_xlim(0.5, n + 0.5)
        ax_l2.set_xticks(x_pts)
        ax_l2.set_xticklabels(xlabels, rotation=12, ha="right")
    if title is not None:
        ax_eA.set_title(title)
    line_eA.set_data(x_pts, gs_energy_err)
    line_l2.set_data(x_pts, gs_l2_err)
    ax_eA.relim()
    ax_eA.autoscale_view(scalex=False)
    ax_l2.relim()
    ax_l2.autoscale_view(scalex=False)
    if _plot_handle is not None:
        _plot_handle.update(fig)
    else:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.5)


for sweep in range(1, n_sweeps + 1):
    poisson_level.smooth(poisson_level.f.vec, poisson_level.gfu.vec,
                         kind="native", nsweeps=1, omega=1.0)
    time.sleep(1)
    _scene.Redraw()
    gs_energy_err.append(energy_error(u_exact, poisson_level))
    gs_l2_err.append(l2_error(u_exact, poisson_level))
    _update_gs_plot()
    if _plot_handle is not None:
        time.sleep(0.5)

if not _ipython:
    plt.ioff()
    plt.show()

# ===========================================================================
# (1b) Coarse Grid Correction
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
print(f"  ||e||_E (after section 1 pre-smooth): "
      f"{energy_error(u_exact, poisson_level):.6e}")

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

e_cgc = energy_error(u_exact, poisson_level)
e_l2_cgc = l2_error(u_exact, poisson_level)
print(f"  ||e||_E after coarse correction:     {e_cgc:.6e}")
print(f"  L2 error:                            {e_l2_cgc:.6e}")

gs_energy_err.append(e_cgc)
gs_l2_err.append(e_l2_cgc)
_cgc_labels = [str(i) for i in range(1, n_sweeps + 1)] + ["CGC"]
_update_gs_plot(xlabels=_cgc_labels, title="Gauss-Seidel + coarse correction")

_scene.Redraw()
plt.close(fig)

#%%
# ===========================================================================
# (2) SINGLE V-CYCLE: build a hierarchy, then run one V-cycle.
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
    u_exact=u_exact,
    verbose=True,
)

cfg = VCycleConfig(
    smoother="native",
    pre_sweeps=2,
    post_sweeps=2,
    omega=1.0,
    coarse_direct=True,
)
solver = MultigridSolver(hierarchy, cfg)

fine = hierarchy.finest
fine.set_initial_guess(x0_cf, enforce_bc=True)
print(f"\n  before cycle  ||r_free||_2 = {fine.residual_norm():.6e}"
      f"   L2 err = {l2_error(u_exact, fine):.6e}")

# --- Multi-cycle driver (relative L2 residual stopping)
r0 = fine.residual_norm()
hist, level_hist = solver.solve(
    max_cycles=25,
    tol=1e-8,
    verbose=False,
    record_levels=False,
    norms=("l2", "dual"),
    stop_norm="l2",
)
l2_hist = hist_series(hist, "l2")
dual_hist = hist_series(hist, "dual")
num_cycles = len(l2_hist)
print(f"  after {num_cycles} cycles  ||r_free||_2 = {l2_hist[-1]:.6e}"
      f"   dual = {dual_hist[-1]:.6e}"
      f"   L2 err = {l2_error(u_exact, fine):.6e}")

fig, (ax_l2, ax_dual) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
plot_hist_vs_cycle(
    hist,
    key="l2",
    r0=r0,
    ax=ax_l2,
    title="L2 residual (finest, tol=1e-8)",
    show=False,
)
plot_hist_vs_cycle(
    hist,
    key="dual",
    ax=ax_dual,
    title=r"Update dual $\sqrt{r_{\mathrm{before}}^T \Delta x}$ (full V-cycle, finest)",
    show=False,
)
fig.tight_layout()
plt.show()

#%%

# ===========================================================================
# Num cycles vs ndof for a fixed tolerance
# 
# ===========================================================================

print()
print("=" * 70)
print("Num cycles vs ndof for a fixed tolerance")
print("=" * 70)

TOL = 1e-8

study_cfg = VCycleConfig(
    smoother="native", 
    pre_sweeps=5, 
    post_sweeps=5, 
    omega=1.0, 
    coarse_direct=True,
)

# Zero IC so ||r0||_2 is comparable across refinements (HF x0_cf skews cycle counts).
x0_study = 0.0

mesh_c = Mesh(unit_square.GenerateMesh(maxh=0.1))
results = []  # (ndof, n_cycles, r0, r_fin, dual_fin, ok)

print(f"  {'n_ref':>5}  {'ndof':>8}  {'cycles':>6}  {'time(s)':>9}  "
      f"{'||r0||':>10}  {'||r_fin||':>10}  {'dual_fin':>10}  {'tol||r0||':>10}  ok")

fig_study, (ax_dual_hist, ax_cycles, ax_dual_ndof) = plt.subplots(3, 1, figsize=(8, 10))

for n_refines in [1, 2, 3, 4, 5, 6]:
    hierarchy = build_hierarchy(
        mesh_c, poisson_setup, n_refines=n_refines,
        order=1, dirichlet=DIRICHLET, dirichlet_value=0.0,
        u_exact=u_exact,
    )
    fine = hierarchy.finest
    solver = MultigridSolver(hierarchy, study_cfg)
    fine.set_initial_guess(x0_study, enforce_bc=True)
    r0 = fine.residual_norm()
    t0 = time.perf_counter()
    hist, _ = solver.solve(
        max_cycles=100, tol=TOL, norms=("l2", "dual"), stop_norm="l2",
    )
    elapsed = time.perf_counter() - t0
    l2_hist = hist_series(hist, "l2")
    dual_hist = hist_series(hist, "dual")
    r_fin = l2_hist[-1]
    dual_fin = dual_hist[-1]
    ok = r_fin <= TOL * r0
    results.append((fine.ndof, len(l2_hist), r0, r_fin, dual_fin, ok))

    cycles = np.arange(1, len(dual_hist) + 1, dtype=float)
    ax_dual_hist.semilogy(
        cycles, dual_hist, "o-", ms=5, lw=1.2, label=f"ndof={fine.ndof}",
    )

    print(f"  {n_refines:5d}  {fine.ndof:8d}  {len(l2_hist):6d}  {elapsed:9.3f}  "
          f"{r0:10.3e}  {r_fin:10.3e}  {dual_fin:10.3e}  {TOL * r0:10.3e}  {ok}")

ndofs, n_cycles, _, _, dual_fins, _ = zip(*results)
ndofs_a = np.asarray(ndofs)
n_cycles_a = np.asarray(n_cycles)
dual_fins_a = np.asarray(dual_fins)

ax_dual_hist.set_xlabel("V-cycle")
ax_dual_hist.set_ylabel(r"$\sqrt{r_{\mathrm{before}}^T \Delta x}$ (finest)")
ax_dual_hist.set_title(
    fr"Dual norm per cycle (tol = {TOL:g}, stop $\|r\|_2$, $x_0=0$)"
)
ax_dual_hist.legend(loc="upper right", fontsize=8)
ax_dual_hist.grid(True, alpha=0.3)

ax_cycles.plot(ndofs_a, n_cycles_a, "o-", ms=7, lw=1.5)
ax_cycles.set_ylabel("V-cycles to tol")
ax_cycles.set_title(fr"Stopping on $\|r\|_2$ (tol = {TOL:g}, $x_0=0$)")
ax_cycles.grid(True, alpha=0.3)

ax_dual_ndof.semilogy(ndofs_a, dual_fins_a, "s-", ms=7, lw=1.5, color="C1")
ax_dual_ndof.set_xlabel("fine ndof")
ax_dual_ndof.set_ylabel(r"final $\sqrt{r^T \Delta x}$")
ax_dual_ndof.set_title("Final dual norm at stop (finest)")
ax_dual_ndof.grid(True, alpha=0.3)

fig_study.tight_layout()
plt.show()

#%%
# ===========================================================================
# Mesh independence: fixed finest mesh, varying hierarchy depth
# ===========================================================================

print()
print("=" * 70)
print("V-cycles vs number of levels (fixed finest mesh, fixed tol)")
print("=" * 70)

TOL = 1e-8
study_cfg = VCycleConfig(
    smoother="native", pre_sweeps=2, post_sweeps=2, omega=1.0, coarse_direct=True,
)
# Smooth x0 so ||r0||_A is comparable across refinements (HF x0_cf skews cycle counts).
x0_study = 0.0

results: list[tuple[float, int, int, int, int, float, float, bool]] = []

print(f"  {'h_c':>6}  {'n_ref':>5}  {'nlev':>4}  {'ndof':>8}  {'cycles':>6}  {'time(s)':>9}  "
      f"{'||r0||':>10}  {'||r_fin||':>10}  {'tol||r0||':>10}  ok")

# 0.01/2**2 → h_finest=0.0025 (~16× fewer fine DOFs than 2**4 anchor)
h_finest = 0.01 / (2 ** 2)
for h_coarse in [0.01, 0.02, 0.04, 0.08]:
    mesh_c = Mesh(unit_square.GenerateMesh(maxh=h_coarse))
    n_refines = max(1, int(math.log2(h_coarse / h_finest)))

    hierarchy = build_hierarchy(
        mesh_c, poisson_setup, n_refines=n_refines,
        order=1, dirichlet=DIRICHLET, dirichlet_value=0.0,
        u_exact=u_exact,
    )
    fine = hierarchy.finest
    solver = MultigridSolver(hierarchy, study_cfg)
    fine.set_initial_guess(x0_study, enforce_bc=True)
    r0 = fine.residual_norm()
    t0 = time.perf_counter()
    hist, _ = solver.solve(
        max_cycles=100, tol=TOL, norms=("l2",), stop_norm="l2",
    )
    elapsed = time.perf_counter() - t0
    l2_hist = hist_series(hist, "l2")
    r_fin = l2_hist[-1]
    ok = r_fin <= TOL * r0
    results.append((
        h_coarse, n_refines, hierarchy.nlevels, fine.ndof,
        len(l2_hist), r0, r_fin, ok,
    ))
    print(f"  {h_coarse:6.3f}  {n_refines:5d}  {hierarchy.nlevels:4d}  {fine.ndof:8d}  "
          f"{len(l2_hist):6d}  {elapsed:9.3f}  {r0:10.3e}  {r_fin:10.3e}  {TOL * r0:10.3e}  {ok}")

_, _, n_levels, ndofs, n_cycles, *_ = zip(*results)
ndof_fine = ndofs[0]
if len(set(ndofs)) > 1:
    print(f"  warning: finest ndof varies across runs: {set(ndofs)}")

plt.figure(figsize=(7, 4))
plt.plot(n_levels, n_cycles, "o-", label=f"fine ndof = {ndof_fine}")
plt.xlabel("number of multigrid levels")
plt.ylabel("V-cycles to reach tol")
plt.title(
    fr"fixed finest $h \approx {h_finest:g}$, tol = {TOL:g}, $\|r\|_2$, $x_0=0$"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()





#%%

class VCyclePreconditioner(BaseMatrix):
    """y = M^{-1} x  via one V-cycle on A z = x, z0 = 0."""
    def __init__(self, mg_solver, level):
        super().__init__()
        self.solver = mg_solver
        self.level = level
        self.idx = mg_solver.h.finest_idx

    def Mult(self, x, y):
        y.FV().NumPy()[:] = 0.0
        self.solver.v_cycle(self.idx, x, y)
        y.FV().NumPy()[self.level.fixed_ids] = 0.0

# --- after you built hierarchy + MultigridSolver(hierarchy, study_cfg) ---
fine = hierarchy.finest
a, f = fine.a, fine.f   # BilinearForm / LinearForm on finest

pre = VCyclePreconditioner(solver, fine)
gfu = fine.gfu
gfu.vec[:] = 0.0
fine.set_initial_guess(x0_study, enforce_bc=True)  # or leave 0

inv = solvers.CGSolver(
    mat=a.mat,
    pre=pre,  # pre XOR freedofs — not both
    maxiter=200,
    tol=1e-8,
    printrates=True,
)
gfu.vec.data = inv * f.vec

n_cg = inv.iterations
print("PCG (NGSolve CG + your V-cycle):", n_cg, "iterations")





# %%

fine = hierarchy.finest
a, f = fine.a, fine.f   # BilinearForm / LinearForm on finest

pre = VCyclePreconditioner(solver, fine)
gfu = fine.gfu
gfu.vec[:] = 0.0
fine.set_initial_guess(x0_study, enforce_bc=True)  # or leave 0


inv = solvers.CGSolver(
    mat=a.mat,
    freedofs=fine.fes.FreeDofs(),   # BitArray, not fine.free_ids
    maxiter=2000, tol=1e-8, printrates=True,
)
gfu.vec.data = inv * f.vec

n_cg = inv.iterations
print("PCG (NGSolve CG + your V-cycle):", n_cg, "iterations")



# %%
