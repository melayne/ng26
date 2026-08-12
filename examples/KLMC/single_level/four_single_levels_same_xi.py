"""Four single-level solves with shared ``xi``, showing QoI difference decay.

For each Monte Carlo sample we draw one ``xi``, build one ``kappa``, and solve
on every mesh in a refining sequence.  Consecutive QoI differences

    Y_ell = Q_ell - Q_{ell-1}

are therefore coupled (same random field).  Averaging over samples makes the
decay of ``|E[Y_ell]|`` (and typically ``E[|Y_ell|]``) with ``maxh`` visible —
a single realization need not be monotone.

Default meshes (coarse → fine):

    maxh = 0.30, 0.15, 0.075, 0.0375

PDE / BCs: ``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, natural
no-flow on top/bottom.  QoI: ``∫_right (-κ ∂_n p) ds``.

Run from the project root with

    .venv/bin/python examples/KLMC/single_level/four_single_levels_same_xi.py
    .venv/bin/python examples/KLMC/single_level/four_single_levels_same_xi.py \\
        --n-samples 50 --no-show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ngsolve as ng
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = PROJECT_ROOT / "examples" / "plots"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import (  # noqa: E402
    cartesian_grid_2d,
    exponential_covariance,
    leading_eigenpairs,
    lognormal_transform,
    sample_discrete_kl,
    voxel_coefficient_2d,
)
from examples.KLMC.utils.one_level_utils import (  # noqa: E402
    build_fixed_mesh,
    quantity_of_interest,
    solve_diffusion,
)

ng.SetHeapSize(1_000_000_000)

DEFAULT_MAXHS = (0.30, 0.15, 0.075, 0.0375)


@dataclass(frozen=True)
class LevelMesh:
    maxh: float
    mesh: ng.Mesh
    fes: ng.H1


def mesh_triangulation(mesh: ng.Mesh) -> mtri.Triangulation:
    points = np.array([vertex.point[:2] for vertex in mesh.vertices])
    triangles = np.array(
        [[vertex.nr for vertex in element.vertices] for element in mesh.Elements(ng.VOL)]
    )
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def overlay_mesh(ax, mesh: ng.Mesh, *, linewidth: float = 0.7) -> None:
    ax.triplot(
        mesh_triangulation(mesh),
        color="white",
        linewidth=linewidth,
        alpha=0.9,
    )


def spatial_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def evaluate_on_grid(coefficient, mesh, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficient(mesh(X.ravel(), Y.ravel())))
    return values.reshape(X.shape)


def make_plot(
    *,
    X,
    Y,
    log_kappa_values,
    kappa_values,
    level_meshes: list[LevelMesh],
    qoi_samples: np.ndarray,
    first_coarse_solution: ng.GridFunction,
    first_fine_solution: ng.GridFunction,
) -> plt.Figure:
    """qoi_samples has shape (n_samples, n_levels)."""
    maxhs = np.array([level.maxh for level in level_meshes])
    mean_q = qoi_samples.mean(axis=0)
    stderr_q = qoi_samples.std(axis=0, ddof=1) / np.sqrt(qoi_samples.shape[0])

    diff_samples = np.diff(qoi_samples, axis=1)
    mean_diff = diff_samples.mean(axis=0)
    mean_abs_diff = np.mean(np.abs(diff_samples), axis=0)
    diff_maxhs = maxhs[1:]

    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)

    ax_z = figure.add_subplot(grid[0, 0])
    ax_k = figure.add_subplot(grid[0, 1])
    ax_q = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])
    ax_sol = [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])]

    log_limit = max(abs(log_kappa_values.min()), abs(log_kappa_values.max()), 1e-12)
    log_plot = ax_z.contourf(
        X,
        Y,
        log_kappa_values,
        levels=np.linspace(-log_limit, log_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    ax_z.scatter(X, Y, s=4, color="black", alpha=0.3)
    spatial_axis(ax_z, r"Example $Z=\log(\kappa)$ (sample 0)")
    figure.colorbar(log_plot, ax=ax_z, label=r"$Z$")

    kappa_plot = ax_k.contourf(X, Y, kappa_values, levels=31, cmap="viridis")
    spatial_axis(ax_k, r"Example $\kappa=\exp(Z)$ (same $\xi$ on all levels)")
    figure.colorbar(kappa_plot, ax=ax_k, label=r"$\kappa$")

    ax_q.errorbar(
        maxhs,
        mean_q,
        yerr=1.96 * stderr_q,
        fmt="o-",
        capsize=4,
        color="tab:blue",
        label=r"$E[Q_\ell]$",
    )
    ax_q.invert_xaxis()
    ax_q.set_xlabel("maxh (coarse → fine)")
    ax_q.set_ylabel(r"$E[Q_\ell]$")
    ax_q.set_title("Mean QoI vs mesh size")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=9)

    ax_d.semilogy(
        diff_maxhs,
        np.abs(mean_diff),
        "o-",
        color="tab:blue",
        label=r"$|E[Q_\ell - Q_{\ell-1}]|$",
    )
    ax_d.semilogy(
        diff_maxhs,
        mean_abs_diff,
        "s--",
        color="tab:red",
        label=r"$E[|Q_\ell - Q_{\ell-1}|]$",
    )
    ax_d.invert_xaxis()
    ax_d.set_xlabel(r"finer maxh in the pair $(\ell-1,\ell)$")
    ax_d.set_ylabel("QoI difference")
    ax_d.set_title("Coupled level differences shrink under refinement")
    ax_d.grid(True, which="both", alpha=0.3)
    ax_d.legend(fontsize=9)
    for maxh, value in zip(diff_maxhs, np.abs(mean_diff)):
        ax_d.annotate(
            f"{value:.2e}",
            (maxh, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    plot_X, plot_Y, _ = cartesian_grid_2d(121, 121)
    coarse, fine = level_meshes[0], level_meshes[-1]
    coarse_values = evaluate_on_grid(
        first_coarse_solution, coarse.mesh, plot_X, plot_Y
    )
    fine_values = evaluate_on_grid(first_fine_solution, fine.mesh, plot_X, plot_Y)
    vmin = float(min(coarse_values.min(), fine_values.min()))
    vmax = float(max(coarse_values.max(), fine_values.max()))
    p_levels = np.linspace(vmin, vmax, 31)

    for ax, level, values, linewidth in (
        (ax_sol[0], coarse, coarse_values, 1.2),
        (ax_sol[1], fine, fine_values, 0.5),
    ):
        pressure_plot = ax.contourf(
            plot_X, plot_Y, values, levels=p_levels, cmap="plasma", extend="both"
        )
        overlay_mesh(ax, level.mesh, linewidth=linewidth)
        spatial_axis(
            ax,
            rf"$p_h$ sample 0, maxh={level.maxh:g}, ndof={level.fes.ndof}",
        )
        figure.colorbar(pressure_plot, ax=ax, label=r"$p_h$")

    n_samples = qoi_samples.shape[0]
    figure.suptitle(
        rf"Coupled single-level solves (same $\xi$ per sample), $N={n_samples}$",
        fontsize=14,
    )
    return figure


def main(
    *,
    maxhs: tuple[float, ...] = DEFAULT_MAXHS,
    n_samples: int = 40,
    grids: int = 32,
    num_modes: int = 100,
    sigma: float = 1.0,
    correlation_length: float = 0.30,
    seed: int = 7,
    show_plot: bool = True,
    plot_dir: Path = DEFAULT_PLOT_DIR,
) -> np.ndarray:
    maxh_list = [float(h) for h in maxhs]
    if len(maxh_list) < 2:
        raise ValueError("Need at least two maxh values.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if any(h <= 0.0 for h in maxh_list):
        raise ValueError("Every maxh must be positive.")
    if any(fine >= coarse for coarse, fine in zip(maxh_list, maxh_list[1:])):
        raise ValueError("maxhs must be strictly decreasing (coarse → fine).")

    level_meshes = [
        LevelMesh(maxh=maxh, mesh=mesh, fes=fes)
        for maxh in maxh_list
        for mesh, fes in [build_fixed_mesh(maxh=maxh)]
    ]

    nx = ny = grids
    X, Y, points = cartesian_grid_2d(nx, ny)
    covariance = exponential_covariance(
        points,
        sigma=sigma,
        correlation_length=correlation_length,
    )
    eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=num_modes)
    total_variance = float(np.trace(covariance))

    print(f"Levels maxh={maxh_list}")
    print(f"ndofs={[level.fes.ndof for level in level_meshes]}")
    print(
        f"KL {nx}x{ny}, modes={len(eigenvalues)}, "
        f"retained={eigenvalues.sum() / total_variance:.1%}, N={n_samples}"
    )
    print()

    rng = np.random.default_rng(seed)
    qoi_samples = np.empty((n_samples, len(level_meshes)), dtype=float)

    first_log_kappa: np.ndarray | None = None
    first_kappa_values: np.ndarray | None = None
    first_coarse_solution: ng.GridFunction | None = None
    first_fine_solution: ng.GridFunction | None = None

    for sample_index in range(n_samples):
        log_kappa_values, _ = sample_discrete_kl(
            mean=0.0,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            shape=(ny, nx),
            rng=rng,
        )
        kappa_values = lognormal_transform(log_kappa_values)
        kappa = voxel_coefficient_2d(kappa_values, linear=True)

        solutions: list[ng.GridFunction] = []
        for level_index, level in enumerate(level_meshes):
            solution = solve_diffusion(level.fes, kappa)
            qoi_samples[sample_index, level_index] = quantity_of_interest(
                solution, level.mesh, kappa
            )
            solutions.append(solution)

        if sample_index == 0:
            first_log_kappa = log_kappa_values
            first_kappa_values = kappa_values
            first_coarse_solution = solutions[0]
            first_fine_solution = solutions[-1]

        if (sample_index + 1) % max(1, n_samples // 5) == 0 or sample_index == 0:
            print(
                f"  sample {sample_index + 1}/{n_samples}: "
                f"Q={np.array2string(qoi_samples[sample_index], precision=4)}",
                flush=True,
            )

    mean_q = qoi_samples.mean(axis=0)
    diff_samples = np.diff(qoi_samples, axis=1)
    mean_diff = diff_samples.mean(axis=0)
    mean_abs_diff = np.mean(np.abs(diff_samples), axis=0)

    print()
    print(
        f"{'maxh':>10}  {'ndof':>6}  {'E[Q]':>12}  "
        f"{'|E[ΔQ]|':>12}  {'E[|ΔQ]|':>12}"
    )
    print("-" * 60)
    for index, level in enumerate(level_meshes):
        if index == 0:
            print(
                f"{level.maxh:10.4g}  {level.fes.ndof:6d}  {mean_q[index]:12.6e}  "
                f"{'—':>12}  {'—':>12}"
            )
        else:
            print(
                f"{level.maxh:10.4g}  {level.fes.ndof:6d}  {mean_q[index]:12.6e}  "
                f"{abs(mean_diff[index - 1]):12.6e}  {mean_abs_diff[index - 1]:12.6e}"
            )

    assert first_log_kappa is not None
    assert first_kappa_values is not None
    assert first_coarse_solution is not None
    assert first_fine_solution is not None

    figure = make_plot(
        X=X,
        Y=Y,
        log_kappa_values=first_log_kappa,
        kappa_values=first_kappa_values,
        level_meshes=level_meshes,
        qoi_samples=qoi_samples,
        first_coarse_solution=first_coarse_solution,
        first_fine_solution=first_fine_solution,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "four_single_levels_same_xi.png"
    figure.savefig(plot_path, dpi=180)
    print(f"\nsaved plot: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return qoi_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maxhs",
        nargs="+",
        type=float,
        default=list(DEFAULT_MAXHS),
        help="strictly decreasing FE mesh sizes",
    )
    parser.add_argument("--n-samples", type=int, default=40, dest="n_samples")
    parser.add_argument("--grids", type=int, default=32)
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
        show_plot=not args.no_show,
        plot_dir=args.plot_dir,
    )
