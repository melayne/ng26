"""Monte Carlo QoI vs number of samples (fixed FE mesh and KL grid).

Fixes one NGSolve mesh and one KL grid size, then compares Monte Carlo
estimates of the outflow-flux QoI as ``N`` increases, e.g.

    N = 100, 150, 200, 300

PDE / BCs follow a Reddy-style pressure Darcy setup on the unit square:
``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, and natural
no-flow on top/bottom.  The QoI is the right-edge outflow flux
``∫ (-κ ∂_n p) ds``.

Run from the project root with

    .venv/bin/python examples/one_level_KLMC/qoi_vs_nsamples.py
    .venv/bin/python examples/one_level_KLMC/qoi_vs_nsamples.py --n-samples 50 100 --no-show
"""

from __future__ import annotations

import argparse
import sys
import time
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


def run_mc_for_nsamples(
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
    """Monte Carlo over KL realizations on a fixed FE mesh and KL grid."""
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
    sample_counts = [len(r.qoi_samples) for r in results]
    means = np.array([r.mean for r in results])
    stderrs = np.array([r.stderr for r in results])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].errorbar(sample_counts, means, yerr=1.96 * stderrs, fmt="o-", capsize=4)
    axes[0].set_xlabel("number of MC samples N")
    axes[0].set_ylabel(r"MC mean of outflow flux $Q$")
    axes[0].set_title("QoI vs number of samples")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(sample_counts)

    for result in results:
        n = len(result.qoi_samples)
        axes[1].hist(
            result.qoi_samples,
            bins=min(20, max(5, n // 3)),
            alpha=0.45,
            density=True,
            label=f"N={n}",
        )
    axes[1].set_xlabel(r"$Q$")
    axes[1].set_ylabel("density")
    axes[1].set_title("QoI sample histograms")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("Monte Carlo QoI vs sample size", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    return figure


def print_table(results: list[GridMCResult]) -> None:
    header = (
        f"{'N':>5}  {'nx':>4}  {'modes':>5}  {'retVar':>7}  {'E[Q]':>12}  "
        f"{'Var(Q)':>12}  {'stderr':>10}  {'time[s]':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        n = len(r.qoi_samples)
        print(
            f"{n:5d}  {r.nx:4d}  {r.n_modes:5d}  {r.retained_variance:7.1%}  "
            f"{r.mean:12.6e}  {r.variance:12.6e}  {r.stderr:10.3e}  "
            f"{r.wall_time_s:8.1f}"
        )


def main(
    *,
    n_samples: list[int] | tuple[int, ...] = (100, 150, 200, 300),
    grids: int = 64,
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
    print(f"Fixed KL grid: {grids}x{grids}")
    print(
        f"MC sample counts: {list(n_samples)}; "
        f"modes<= {num_modes}; ell={correlation_length}; sigma={sigma}"
    )
    print()

    results: list[GridMCResult] = []
    for samples in n_samples:
        print(f"Running MC with N={samples} ...", flush=True)
        result = run_mc_for_nsamples(
            nx=grids,
            mesh=mesh,
            fes=fes,
            n_samples=samples,
            num_modes=num_modes,
            sigma=sigma,
            correlation_length=correlation_length,
            seed=seed + samples,
        )
        results.append(result)
        print(
            f"  done: E[Q]={result.mean:.6e}, stderr={result.stderr:.3e}, "
            f"time={result.wall_time_s:.1f}s",
            flush=True,
        )

    print()
    print_table(results)

    plot_path = plot_dir / "qoi_vs_nsamples.png"
    figure = make_summary_plot(results, plot_path)
    print(f"\nsaved plot: {plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, default=32, help="KL grid size nx (= ny)")
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=[100, 150, 200, 300, 400, 500, 600, 700, 800, 2000],
        help="MC sample counts to compare",
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
        n_samples=tuple(args.n_samples),
        grids=args.grids,
        maxh=args.maxh,
        num_modes=args.num_modes,
        sigma=args.sigma,
        correlation_length=args.correlation_length,
        seed=args.seed,
        plot_dir=args.plot_dir,
        show_plot=not args.no_show,
    )
