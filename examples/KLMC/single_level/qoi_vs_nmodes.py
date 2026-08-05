"""Monte Carlo QoI vs number of retained KL modes.

Fixes one FE mesh, one KL grid, and one sample count ``N``, then compares
Monte Carlo estimates of the outflow-flux QoI as the KL truncation length
increases, e.g.

    n_modes = 10, 25, 50, 75, 100, 150, 200

The full KL basis is computed once at ``max(n_modes)``; each study uses the
leading ``m`` eigenpairs only.

PDE / BCs follow a Reddy-style pressure Darcy setup on the unit square:
``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, and natural
no-flow on top/bottom.  The QoI is the right-edge outflow flux
``∫ (-κ ∂_n p) ds``.

Run from the project root with

    .venv/bin/python examples/one_level_KLMC/qoi_vs_nmodes.py
    .venv/bin/python examples/one_level_KLMC/qoi_vs_nmodes.py --n-modes 10 50 100 --no-show
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

from KL_expansion import (  # noqa: E402
    lognormal_transform,
    sample_discrete_kl,
    voxel_coefficient_2d,
)
from utils.one_level_utils import (  # noqa: E402
    GridMCResult,
    build_fixed_mesh,
    prepare_kl_basis,
    quantity_of_interest,
    solve_diffusion,
)

ng.SetHeapSize(1_000_000_000)

DEFAULT_N_MODES = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def run_mc_for_nmodes(
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
    """Monte Carlo using a fixed truncated KL eigenbasis."""
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


def make_summary_plot(results: list[GridMCResult], plot_path: Path) -> Figure:
    mode_counts = [r.n_modes for r in results]
    means = np.array([r.mean for r in results])
    stderrs = np.array([r.stderr for r in results])
    retained = np.array([r.retained_variance for r in results])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].errorbar(mode_counts, means, yerr=1.96 * stderrs, fmt="o-", capsize=4)
    axes[0].set_xlabel("number of retained KL modes")
    axes[0].set_ylabel(r"MC mean of outflow flux $Q$")
    axes[0].set_title("QoI vs KL truncation")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(mode_counts)

    ax_var = axes[0].twinx()
    ax_var.plot(mode_counts, 100.0 * retained, "s--", color="tab:orange", alpha=0.8)
    ax_var.set_ylabel("retained variance (%)", color="tab:orange")
    ax_var.set_ylim(0.0, 100.0)

    for result in results:
        axes[1].hist(
            result.qoi_samples,
            bins=min(20, max(5, len(result.qoi_samples) // 3)),
            alpha=0.45,
            density=True,
            label=f"m={result.n_modes} ({result.retained_variance:.0%})",
        )
    axes[1].set_xlabel(r"$Q$")
    axes[1].set_ylabel("density")
    axes[1].set_title("QoI sample histograms")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("Monte Carlo QoI vs retained KL modes", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    return figure


def print_table(results: list[GridMCResult]) -> None:
    header = (
        f"{'modes':>5}  {'retVar':>7}  {'N':>5}  {'nx':>4}  {'E[Q]':>12}  "
        f"{'Var(Q)':>12}  {'stderr':>10}  {'time[s]':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        n = len(r.qoi_samples)
        print(
            f"{r.n_modes:5d}  {r.retained_variance:7.1%}  {n:5d}  {r.nx:4d}  "
            f"{r.mean:12.6e}  {r.variance:12.6e}  {r.stderr:10.3e}  "
            f"{r.wall_time_s:8.1f}"
        )


def main(
    *,
    n_modes: list[int] | tuple[int, ...] = DEFAULT_N_MODES,
    n_samples: int = 1000,
    grids: int = 64,
    maxh: float = 0.18,
    sigma: float = 1.0,
    correlation_length: float = 0.30,
    seed: int = 7,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    show_plot: bool = True,
) -> list[GridMCResult]:
    mode_list = sorted({int(m) for m in n_modes})
    max_modes = max(mode_list)
    n_points = grids * grids
    if max_modes > n_points:
        raise ValueError(
            f"max(n_modes)={max_modes} exceeds KL grid size {grids}x{grids}={n_points}."
        )

    mesh, fes = build_fixed_mesh(maxh=maxh)
    print(f"Fixed FE mesh: maxh={maxh}, ndof={fes.ndof}")
    print(f"Fixed KL grid: {grids}x{grids}")
    print(f"Fixed MC samples: N={n_samples}")
    print(
        f"Mode counts: {mode_list}; "
        f"ell={correlation_length}; sigma={sigma}"
    )
    print()

    eigenvalues_full, eigenvectors_full, total_variance = prepare_kl_basis(
        grids,
        sigma=sigma,
        correlation_length=correlation_length,
        num_modes=max_modes,
    )
    print(
        f"KL basis ready: computed {len(eigenvalues_full)} modes, "
        f"max retained variance="
        f"{eigenvalues_full.sum() / total_variance:.1%}"
    )
    print()

    results: list[GridMCResult] = []
    for m in mode_list:
        print(f"Running MC with n_modes={m} ...", flush=True)
        result = run_mc_for_nmodes(
            nx=grids,
            mesh=mesh,
            fes=fes,
            n_samples=n_samples,
            eigenvalues=eigenvalues_full[:m],
            eigenvectors=eigenvectors_full[:, :m],
            total_variance=total_variance,
            seed=seed + m,
        )
        results.append(result)
        print(
            f"  done: E[Q]={result.mean:.6e}, stderr={result.stderr:.3e}, "
            f"retVar={result.retained_variance:.1%}, time={result.wall_time_s:.1f}s",
            flush=True,
        )

    print()
    print_table(results)

    plot_path = plot_dir / "qoi_vs_nmodes.png"
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
    parser.add_argument("--maxh", type=float, default=0.18)
    parser.add_argument("--n-samples", type=int, default=500, dest="n_samples")
    parser.add_argument(
        "--n-modes",
        nargs="+",
        type=int,
        default=DEFAULT_N_MODES,
        help="retained KL mode counts to compare",
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()
    main(
        n_modes=tuple(args.n_modes),
        n_samples=args.n_samples,
        grids=args.grids,
        maxh=args.maxh,
        sigma=args.sigma,
        correlation_length=args.correlation_length,
        seed=args.seed,
        plot_dir=args.plot_dir,
        show_plot=not args.no_show,
    )
