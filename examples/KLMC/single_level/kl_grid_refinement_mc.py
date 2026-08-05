"""KL-grid refinement Monte Carlo with a fixed FE mesh.

Fixes one NGSolve mesh and compares Monte Carlo estimates of a scalar QoI
when the discrete KL coefficient grid is refined:

    nx = ny = 16 -> 32 -> 64

PDE / BCs follow a Reddy-style pressure Darcy setup on the unit square:
``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, and natural
no-flow on top/bottom.  The QoI is the right-edge outflow flux
``∫ (-κ ∂_n p) ds``.

The KL eigenproblem is rebuilt on each Cartesian grid (same covariance
kernel), then independent Monte Carlo samples are drawn.  Stabilizing QoI
statistics indicate that the KL grid / VoxelCoefficient interpolation is
fine enough for that mesh.

Run from the project root with

    .venv/bin/python examples/one_level_KLMC/kl_grid_refinement_mc.py
    .venv/bin/python examples/one_level_KLMC/kl_grid_refinement_mc.py --n-samples 50 --no-show
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np
from matplotlib.figure import Figure
from netgen.geom2d import unit_square


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = PROJECT_ROOT / "examples" / "plots"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import (  # noqa: E402
    cartesian_grid_2d,
    exponential_covariance,
    leading_eigenpairs,
    lognormal_transform,
    sample_discrete_kl,
    voxel_coefficient_2d,
)


@dataclass(frozen=True)
class GridMCResult:
    nx: int
    n_modes: int
    retained_variance: float
    qoi_samples: np.ndarray
    wall_time_s: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.qoi_samples))

    @property
    def variance(self) -> float:
        return float(np.var(self.qoi_samples, ddof=1))

    @property
    def stderr(self) -> float:
        n = len(self.qoi_samples)
        return float(np.sqrt(self.variance / n))


def build_fixed_mesh(*, maxh: float) -> tuple[ng.Mesh, ng.H1]:
    """One triangular mesh / H1 space reused for every KL grid.

    Dirichlet only on left/right (pressure drive). Top/bottom are natural
    no-flow boundaries, matching a Reddy-style pressure Darcy setup.
    """
    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = ng.H1(mesh, order=1, dirichlet="left|right")
    return mesh, fes


def quantity_of_interest(solution: ng.GridFunction, mesh: ng.Mesh, kappa) -> float:
    """Outflow flux ``∫_right (-κ ∂_n p) ds``.

    Volume gradients must be wrapped with ``BoundaryFromVolumeCF`` before
    integrating on facets; otherwise NGSolve returns 0.
    """
    normal = ng.specialcf.normal(mesh.dim)
    flux_density = -kappa * ng.InnerProduct(ng.grad(solution), normal)
    return float(
        ng.Integrate(
            ng.BoundaryFromVolumeCF(flux_density),
            mesh.Boundaries("right"),
        )
    )


def solve_diffusion(fes: ng.H1, kappa) -> ng.GridFunction:
    """Solve ``-div(κ ∇p)=0`` with ``p=-1`` on left and ``p=0`` on right.

    Top/bottom have no Dirichlet condition, so the natural condition is
    no-flow (``∂_n p = 0``).  Dirichlet data are set with one ``BoundaryCF``
    call — consecutive ``Set(..., definedon=...)`` calls overwrite earlier
    boundary values.
    """
    mesh = fes.mesh
    u, v = fes.TnT()
    a = ng.BilinearForm(fes, symmetric=True)
    a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
    a.Assemble()

    f = ng.LinearForm(fes)
    f.Assemble()

    solution = ng.GridFunction(fes)
    # One Set for both Dirichlet sides (do not call Set twice).
    solution.Set(
        mesh.BoundaryCF({"left": -1.0, "right": 0.0}),
        definedon=mesh.Boundaries("left|right"),
    )

    residual = solution.vec.CreateVector()
    residual.data = f.vec - a.mat * solution.vec
    solution.vec.data += a.mat.Inverse(fes.FreeDofs()) * residual
    return solution


def prepare_kl_basis(
    nx: int,
    *,
    sigma: float,
    correlation_length: float,
    num_modes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Eigenpairs on an ``nx x nx`` Cartesian KL grid."""
    _, _, points = cartesian_grid_2d(nx, nx)
    covariance = exponential_covariance(
        points,
        sigma=sigma,
        correlation_length=correlation_length,
    )
    n_modes = min(num_modes, points.shape[0])
    eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=n_modes)
    total_variance = float(np.trace(covariance))
    return eigenvalues, eigenvectors, total_variance


def run_mc_for_grid(
    *,
    nx: int,
    mesh: ng.Mesh,
    fes: ng.H1,
    n_samples: int,
    num_modes: int,
    sigma: float,
    correlation_length: float,
    seed: int,
) -> GridMCResult:
    """Monte Carlo over KL realizations on a fixed FE mesh."""
    t0 = time.perf_counter()
    eigenvalues, eigenvectors, total_variance = prepare_kl_basis(
        nx,
        sigma=sigma,
        correlation_length=correlation_length,
        num_modes=num_modes,
    )
    rng = np.random.default_rng(seed)
    qoi = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        log_kappa, _ = sample_discrete_kl(
            mean=0.0,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            shape=(nx, nx),
            rng=rng,
        )
        kappa = voxel_coefficient_2d(lognormal_transform(log_kappa), linear=True)
        solution = solve_diffusion(fes, kappa)
        qoi[i] = quantity_of_interest(solution, mesh, kappa)

    return GridMCResult(
        nx=nx,
        n_modes=len(eigenvalues),
        retained_variance=float(eigenvalues.sum() / total_variance),
        qoi_samples=qoi,
        wall_time_s=time.perf_counter() - t0,
    )


def make_summary_plot(results: list[GridMCResult], plot_path: Path) -> Figure:
    grids = [r.nx for r in results]
    means = np.array([r.mean for r in results])
    stderrs = np.array([r.stderr for r in results])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].errorbar(grids, means, yerr=1.96 * stderrs, fmt="o-", capsize=4)
    axes[0].set_xlabel("KL grid size nx (= ny)")
    axes[0].set_ylabel(r"MC mean of outflow flux $Q$")
    axes[0].set_title("QoI vs KL grid (fixed FE mesh)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(grids)

    for result in results:
        axes[1].hist(
            result.qoi_samples,
            bins=min(20, max(5, len(result.qoi_samples) // 3)),
            alpha=0.45,
            density=True,
            label=f"{result.nx}x{result.nx}",
        )
    axes[1].set_xlabel(r"$Q$")
    axes[1].set_ylabel("density")
    axes[1].set_title("QoI sample histograms")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("KL-grid refinement Monte Carlo", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    return figure


def print_table(results: list[GridMCResult]) -> None:
    header = (
        f"{'nx':>4}  {'modes':>5}  {'retVar':>7}  {'E[Q]':>12}  "
        f"{'Var(Q)':>12}  {'stderr':>10}  {'time[s]':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.nx:4d}  {r.n_modes:5d}  {r.retained_variance:7.1%}  "
            f"{r.mean:12.6e}  {r.variance:12.6e}  {r.stderr:10.3e}  "
            f"{r.wall_time_s:8.1f}"
        )


def main(
    *,
    n_samples: int = 100,
    grids: tuple[int, ...] = (16, 32, 64),
    maxh: float = 0.18,
    num_modes: int = 100,
    sigma: float = 1.0,
    correlation_length: float = 0.30,
    seed: int = 7,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    show_plot: bool = True,
) -> list[GridMCResult]:
    mesh, fes = build_fixed_mesh(maxh=maxh)
    print(f"Fixed FE mesh: maxh={maxh}, ndof={fes.ndof}")
    print(
        f"MC samples per KL grid: {n_samples}; "
        f"modes<= {num_modes}; ell={correlation_length}; sigma={sigma}"
    )
    print()

    results: list[GridMCResult] = []
    for nx in grids:
        print(f"Running MC on KL grid {nx}x{nx} ...", flush=True)
        result = run_mc_for_grid(
            nx=nx,
            mesh=mesh,
            fes=fes,
            n_samples=n_samples,
            num_modes=num_modes,
            sigma=sigma,
            correlation_length=correlation_length,
            seed=seed + nx,
        )
        results.append(result)
        print(
            f"  done: E[Q]={result.mean:.6e}, stderr={result.stderr:.3e}, "
            f"time={result.wall_time_s:.1f}s",
            flush=True,
        )

    print()
    print_table(results)

    plot_path = plot_dir / "kl_grid_refinement_mc.png"
    figure = make_summary_plot(results, plot_path)
    print(f"\nsaved plot: {plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="KL grid sizes nx (= ny)",
    )
    parser.add_argument("--maxh", type=float, default=0.18)
    parser.add_argument("--num-modes", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()
    main(
        n_samples=args.n_samples,
        grids=tuple(args.grids),
        maxh=args.maxh,
        num_modes=args.num_modes,
        sigma=args.sigma,
        correlation_length=args.correlation_length,
        seed=args.seed,
        plot_dir=args.plot_dir,
        show_plot=not args.no_show,
    )
