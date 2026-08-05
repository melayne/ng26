"""Monte Carlo QoI vs retained KL modes and FE mesh size (2D study).

Fixes one KL grid and sample count ``N``, then sweeps a Cartesian product of

    n_modes × maxh

comparing Monte Carlo estimates of the outflow-flux QoI.  The full KL basis
is computed once at ``max(n_modes)``; each ``(m, maxh)`` cell uses the leading
``m`` eigenpairs on its own FE mesh.

PDE / BCs follow a Reddy-style pressure Darcy setup on the unit square:
``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, and natural
no-flow on top/bottom.  The QoI is the right-edge outflow flux
``∫ (-κ ∂_n p) ds``.

Defaults use a coarse 2D grid so the product of sweeps stays practical.
Override freely, e.g.

    .venv/bin/python examples/one_level_KLMC/qoi_vs_nmodes_maxh.py
    .venv/bin/python examples/one_level_KLMC/qoi_vs_nmodes_maxh.py \\
        --n-modes 25 50 100 --maxhs 0.3 0.15 0.05 --n-samples 200 --show
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Non-interactive unless --show: avoids a hung IDE run waiting on a GUI window.
if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

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

# Coarser than the 1D studies: cost scales as |modes| * |maxhs| * N.
DEFAULT_N_MODES = [25, 50, 100, 200]
DEFAULT_MAXHS = [0.30, 0.20, 0.10, 0.05]


@dataclass(frozen=True)
class ModesMaxhResult:
    maxh: float
    ndof: int
    mc: GridMCResult

    @property
    def n_modes(self) -> int:
        return self.mc.n_modes

    @property
    def mean(self) -> float:
        return self.mc.mean

    @property
    def variance(self) -> float:
        return self.mc.variance

    @property
    def stderr(self) -> float:
        return self.mc.stderr

    @property
    def retained_variance(self) -> float:
        return self.mc.retained_variance


def run_mc(
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
    """Monte Carlo on one FE mesh with a truncated KL eigenbasis."""
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


def _grid_arrays(
    results: list[ModesMaxhResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted unique axes and mean / stderr matrices shaped (n_maxh, n_modes)."""
    mode_list = sorted({r.n_modes for r in results})
    maxh_list = sorted({r.maxh for r in results}, reverse=True)  # coarse → fine
    mode_index = {m: j for j, m in enumerate(mode_list)}
    maxh_index = {h: i for i, h in enumerate(maxh_list)}

    means = np.full((len(maxh_list), len(mode_list)), np.nan)
    stderrs = np.full_like(means, np.nan)
    for r in results:
        i = maxh_index[r.maxh]
        j = mode_index[r.n_modes]
        means[i, j] = r.mean
        stderrs[i, j] = r.stderr
    return np.asarray(mode_list), np.asarray(maxh_list), means, stderrs


def make_summary_plot(results: list[ModesMaxhResult], plot_path: Path) -> Figure:
    mode_list, maxh_list, means, stderrs = _grid_arrays(results)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Heatmap of E[Q].
    ax = axes[0, 0]
    extent = (
        -0.5,
        len(mode_list) - 0.5,
        len(maxh_list) - 0.5,
        -0.5,
    )
    im = ax.imshow(means, aspect="auto", cmap="viridis", extent=extent)
    ax.set_xticks(range(len(mode_list)))
    ax.set_xticklabels([str(m) for m in mode_list])
    ax.set_yticks(range(len(maxh_list)))
    ax.set_yticklabels([f"{h:g}" for h in maxh_list])
    ax.set_xlabel("n_modes")
    ax.set_ylabel("maxh (coarse → fine)")
    ax.set_title(r"MC mean $E[Q]$")
    figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Heatmap of MC stderr.
    ax = axes[0, 1]
    im = ax.imshow(stderrs, aspect="auto", cmap="magma", extent=extent)
    ax.set_xticks(range(len(mode_list)))
    ax.set_xticklabels([str(m) for m in mode_list])
    ax.set_yticks(range(len(maxh_list)))
    ax.set_yticklabels([f"{h:g}" for h in maxh_list])
    ax.set_xlabel("n_modes")
    ax.set_ylabel("maxh (coarse → fine)")
    ax.set_title(r"MC stderr of $E[Q]$")
    figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Curves vs maxh for each mode count.
    ax = axes[1, 0]
    for j, m in enumerate(mode_list):
        ax.errorbar(
            maxh_list,
            means[:, j],
            yerr=1.96 * stderrs[:, j],
            fmt="o-",
            capsize=3,
            label=f"m={m}",
        )
    ax.invert_xaxis()
    ax.set_xlabel("FE mesh size maxh")
    ax.set_ylabel(r"MC mean of $Q$")
    ax.set_title("QoI vs maxh (per n_modes)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Curves vs n_modes for each maxh.
    ax = axes[1, 1]
    for i, h in enumerate(maxh_list):
        ax.errorbar(
            mode_list,
            means[i, :],
            yerr=1.96 * stderrs[i, :],
            fmt="o-",
            capsize=3,
            label=f"maxh={h:g}",
        )
    ax.set_xlabel("number of retained KL modes")
    ax.set_ylabel(r"MC mean of $Q$")
    ax.set_title("QoI vs n_modes (per maxh)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    figure.suptitle("Monte Carlo QoI vs n_modes × maxh", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    return figure


def print_table(results: list[ModesMaxhResult]) -> None:
    header = (
        f"{'maxh':>6}  {'ndof':>5}  {'modes':>5}  {'retVar':>7}  {'N':>5}  "
        f"{'nx':>4}  {'E[Q]':>12}  {'Var(Q)':>12}  {'stderr':>10}  {'time[s]':>8}"
    )
    print(header)
    print("-" * len(header))
    # Sort coarse→fine maxh, then increasing modes.
    ordered = sorted(results, key=lambda r: (-r.maxh, r.n_modes))
    for r in ordered:
        n = len(r.mc.qoi_samples)
        print(
            f"{r.maxh:6.3f}  {r.ndof:5d}  {r.n_modes:5d}  {r.retained_variance:7.1%}  "
            f"{n:5d}  {r.mc.nx:4d}  {r.mean:12.6e}  {r.variance:12.6e}  "
            f"{r.stderr:10.3e}  {r.mc.wall_time_s:8.1f}"
        )


def main(
    *,
    n_modes: list[int] | tuple[int, ...] = DEFAULT_N_MODES,
    maxhs: list[float] | tuple[float, ...] = DEFAULT_MAXHS,
    n_samples: int = 200,
    grids: int = 64,
    sigma: float = 1.0,
    correlation_length: float = 0.30,
    seed: int = 7,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    show_plot: bool = False,
) -> list[ModesMaxhResult]:
    # IDE / piped runs often fully-buffer stdout; flush so progress is visible.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    mode_list = sorted({int(m) for m in n_modes})
    maxh_list = [float(h) for h in maxhs]
    max_modes = max(mode_list)
    n_points = grids * grids
    if max_modes > n_points:
        raise ValueError(
            f"max(n_modes)={max_modes} exceeds KL grid size {grids}x{grids}={n_points}."
        )

    n_cells = len(mode_list) * len(maxh_list)
    print(f"Fixed KL grid: {grids}x{grids}", flush=True)
    print(f"Fixed MC samples: N={n_samples}", flush=True)
    print(f"Mode counts: {mode_list}", flush=True)
    print(f"FE maxh values: {maxh_list}", flush=True)
    print(
        f"Grid cells: {n_cells} "
        f"({len(mode_list)} modes x {len(maxh_list)} maxh); "
        f"ell={correlation_length}; sigma={sigma}",
        flush=True,
    )
    print(flush=True)

    print(
        f"Building KL basis ({grids}x{grids}, modes={max_modes}) ...",
        flush=True,
    )
    t_kl = time.perf_counter()
    eigenvalues_full, eigenvectors_full, total_variance = prepare_kl_basis(
        grids,
        sigma=sigma,
        correlation_length=correlation_length,
        num_modes=max_modes,
    )
    print(
        f"KL basis ready in {time.perf_counter() - t_kl:.1f}s: "
        f"computed {len(eigenvalues_full)} modes, "
        f"max retained variance="
        f"{eigenvalues_full.sum() / total_variance:.1%}",
        flush=True,
    )
    print(flush=True)

    results: list[ModesMaxhResult] = []
    for maxh in maxh_list:
        mesh, fes = build_fixed_mesh(maxh=maxh)
        print(f"FE mesh maxh={maxh:g}, ndof={fes.ndof}", flush=True)
        for m in mode_list:
            print(f"  Running MC with n_modes={m} ...", flush=True)
            mc = run_mc(
                nx=grids,
                mesh=mesh,
                fes=fes,
                n_samples=n_samples,
                eigenvalues=eigenvalues_full[:m],
                eigenvectors=eigenvectors_full[:, :m],
                total_variance=total_variance,
                seed=seed + m + int(round(1000 * maxh)),
            )
            result = ModesMaxhResult(maxh=maxh, ndof=fes.ndof, mc=mc)
            results.append(result)
            print(
                f"    done: E[Q]={result.mean:.6e}, stderr={result.stderr:.3e}, "
                f"retVar={result.retained_variance:.1%}, time={mc.wall_time_s:.1f}s",
                flush=True,
            )
        print()

    print_table(results)

    plot_path = plot_dir / "qoi_vs_nmodes_maxh.png"
    figure = make_summary_plot(results, plot_path)
    print(f"\nsaved plot: {plot_path}", flush=True)
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, default=64, help="KL grid size nx (= ny)")
    parser.add_argument(
        "--n-modes",
        nargs="+",
        type=int,
        default=DEFAULT_N_MODES,
        help="retained KL mode counts",
    )
    parser.add_argument(
        "--maxhs",
        nargs="+",
        type=float,
        default=DEFAULT_MAXHS,
        help="FE mesh sizes maxh",
    )
    parser.add_argument("--n-samples", type=int, default=200, dest="n_samples")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the matplotlib window (default: save plot only)",
    )
    # Keep --no-show as a harmless alias so old command lines still work.
    parser.add_argument("--no-show", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()
    main(
        n_modes=tuple(args.n_modes),
        maxhs=tuple(args.maxhs),
        n_samples=args.n_samples,
        grids=args.grids,
        sigma=args.sigma,
        correlation_length=args.correlation_length,
        seed=args.seed,
        plot_dir=args.plot_dir,
        show_plot=bool(args.show) and not args.no_show,
    )
