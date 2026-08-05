"""Monte Carlo QoI vs FE mesh size (fixed KL grid and sample count).

Fixes one KL grid and number of MC samples ``N``, then compares estimates
of the outflow-flux QoI as the FE mesh is refined, e.g.

    maxh = 0.30, 0.29, ..., 0.02  (step 0.01; stop near grids=64 spacing)

PDE / BCs follow a Reddy-style pressure Darcy setup on the unit square:
``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, and natural
no-flow on top/bottom.  The QoI is the right-edge outflow flux
``∫ (-κ ∂_n p) ds``.

Run from the project root with

    .venv/bin/python examples/one_level_KLMC/qoi_vs_maxh.py
    .venv/bin/python examples/one_level_KLMC/qoi_vs_maxh.py --maxhs 0.2 0.15 0.1 --no-show
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PLOT_DIR = PROJECT_ROOT / "examples" / "plots"
KLMC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(KLMC_DIR))

from KL_expansion import ( 
    lognormal_transform,
    sample_discrete_kl,
    voxel_coefficient_2d,
)
from utils.one_level_utils import (  
    GridMCResult,
    build_fixed_mesh,
    prepare_kl_basis,
    quantity_of_interest,
    solve_diffusion,
)

# Many assemble/solve cycles need a larger NGSolve heap than the default.
ng.SetHeapSize(1_000_000_000)


DEFAULT_MAXHS = np.arange(0.3, 0.016, -0.01).round(3).tolist()


@dataclass(frozen=True)
class MaxhMCResult:
    maxh: float
    ndof: int
    mc: GridMCResult

    @property
    def mean(self) -> float:
        return self.mc.mean

    @property
    def variance(self) -> float:
        return self.mc.variance

    @property
    def stderr(self) -> float:
        return self.mc.stderr


def run_mc_for_mesh(
    *,
    nx: int,
    mesh: ng.Mesh,
    fes: ng.H1,
    n_samples: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    total_variance: float,
    seed: int,
) -> GridMCResult:
    """Monte Carlo on one FE mesh using a precomputed KL basis."""
    t0 = time.perf_counter()
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


def make_summary_plot(results: list[MaxhMCResult], plot_path: Path) -> Figure:
    maxhs = [r.maxh for r in results]
    means = np.array([r.mean for r in results])
    stderrs = np.array([r.stderr for r in results])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].errorbar(maxhs, means, yerr=1.96 * stderrs, fmt="o-", capsize=4)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("FE mesh size maxh")
    axes[0].set_ylabel(r"MC mean of outflow flux $Q$")
    axes[0].set_title("QoI vs FE mesh size")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(maxhs)

    for result in results:
        axes[1].hist(
            result.mc.qoi_samples,
            bins=min(20, max(5, len(result.mc.qoi_samples) // 3)),
            alpha=0.45,
            density=True,
            label=f"maxh={result.maxh:g} (ndof={result.ndof})",
        )
    axes[1].set_xlabel(r"$Q$")
    axes[1].set_ylabel("density")
    axes[1].set_title("QoI sample histograms")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("Monte Carlo QoI vs FE mesh size", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    return figure


def print_table(results: list[MaxhMCResult]) -> None:
    header = (
        f"{'maxh':>6}  {'ndof':>5}  {'N':>5}  {'nx':>4}  {'modes':>5}  "
        f"{'retVar':>7}  {'E[Q]':>12}  {'Var(Q)':>12}  {'stderr':>10}  {'time[s]':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        n = len(r.mc.qoi_samples)
        print(
            f"{r.maxh:6.3f}  {r.ndof:5d}  {n:5d}  {r.mc.nx:4d}  {r.mc.n_modes:5d}  "
            f"{r.mc.retained_variance:7.1%}  {r.mean:12.6e}  {r.variance:12.6e}  "
            f"{r.stderr:10.3e}  {r.mc.wall_time_s:8.1f}"
        )


def main(
    *,
    maxhs: list[float] | tuple[float, ...] = DEFAULT_MAXHS,
    n_samples: int = 1000,
    grids: int = 64,
    num_modes: int = 100,
    sigma: float = 1.0,
    correlation_length: float = 0.30,
    seed: int = 7,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    show_plot: bool = True,
) -> list[MaxhMCResult]:
    print(f"Fixed KL grid: {grids}x{grids}")
    print(f"Fixed MC samples: N={n_samples}")
    print(
        f"FE maxh values: {list(maxhs)}; "
        f"modes<= {num_modes}; ell={correlation_length}; sigma={sigma}"
    )
    print()

    eigenvalues, eigenvectors, total_variance = prepare_kl_basis(
        grids,
        sigma=sigma,
        correlation_length=correlation_length,
        num_modes=num_modes,
    )
    print(
        f"KL basis ready: modes={len(eigenvalues)}, "
        f"retained variance={eigenvalues.sum() / total_variance:.1%}"
    )
    print()

    results: list[MaxhMCResult] = []
    for maxh in maxhs:
        mesh, fes = build_fixed_mesh(maxh=maxh)
        print(
            f"Running MC with maxh={maxh:g}, ndof={fes.ndof} ...",
            flush=True,
        )
        # Distinct seed per mesh size; keep KL basis shared.
        mc = run_mc_for_mesh(
            nx=grids,
            mesh=mesh,
            fes=fes,
            n_samples=n_samples,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            total_variance=total_variance,
            seed=seed + int(round(1000 * maxh)),
        )
        result = MaxhMCResult(maxh=maxh, ndof=fes.ndof, mc=mc)
        results.append(result)
        print(
            f"  done: E[Q]={result.mean:.6e}, stderr={result.stderr:.3e}, "
            f"time={mc.wall_time_s:.1f}s",
            flush=True,
        )

    print()
    print_table(results)

    plot_path = plot_dir / "qoi_vs_maxh.png"
    figure = make_summary_plot(results, plot_path)
    print(f"\nsaved plot: {plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, default=64, help="KL grid size nx (= ny)")
    parser.add_argument(
        "--maxhs",
        nargs="+",
        type=float,
        default=DEFAULT_MAXHS,
        help="FE mesh sizes maxh to compare (default: 0.30, 0.29, ..., 0.02)",
    )
    parser.add_argument("--n-samples", type=int, default=1000, dest="n_samples")
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
        maxhs=tuple(args.maxhs),
        n_samples=args.n_samples,
        grids=args.grids,
        num_modes=args.num_modes,
        sigma=args.sigma,
        correlation_length=args.correlation_length,
        seed=args.seed,
        plot_dir=args.plot_dir,
        show_plot=not args.no_show,
    )
