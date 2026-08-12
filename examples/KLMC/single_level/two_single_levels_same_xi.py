"""Two single-level solves with one shared KL draw ``xi``.

Mirrors ``kl_one_level.py``, but:

1. draws one discrete KL coefficient from a single ``xi``,
2. reuses the same ``kappa`` on a coarse and a fine FE mesh,
3. solves each level independently (no multigrid / MLMC),
4. plots both solutions and reports the outflow-flux QoI on each level.

PDE / BCs follow the Reddy-style pressure Darcy setup:
``-div(κ ∇p)=0`` with ``p=-1`` on left, ``p=0`` on right, natural no-flow
on top/bottom.  The QoI is ``∫_right (-κ ∂_n p) ds``.

Run from the project root with

    .venv/bin/python examples/KLMC/single_level/two_single_levels_same_xi.py
    .venv/bin/python examples/KLMC/single_level/two_single_levels_same_xi.py --no-show
"""

from __future__ import annotations

import argparse
import sys
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
    ax.set_title(title)
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
    coarse_mesh,
    fine_mesh,
    coarse_solution,
    fine_solution,
    coarse_qoi: float,
    fine_qoi: float,
    coarse_maxh: float,
    fine_maxh: float,
    coarse_ndof: int,
    fine_ndof: int,
):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    log_limit = max(abs(log_kappa_values.min()), abs(log_kappa_values.max()), 1e-12)
    log_plot = axes[0, 0].contourf(
        X,
        Y,
        log_kappa_values,
        levels=np.linspace(-log_limit, log_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    axes[0, 0].scatter(X, Y, s=4, color="black", alpha=0.3)
    spatial_axis(axes[0, 0], r"Shared Gaussian KL field $Z=\log(\kappa)$")
    figure.colorbar(log_plot, ax=axes[0, 0], label=r"$Z(x,\omega)$")

    kappa_plot = axes[0, 1].contourf(X, Y, kappa_values, levels=31, cmap="viridis")
    spatial_axis(axes[0, 1], r"Shared $\kappa=\exp(Z)$ (same $\xi$)")
    figure.colorbar(kappa_plot, ax=axes[0, 1], label=r"$\kappa(x,\omega)$")

    plot_X, plot_Y, _ = cartesian_grid_2d(121, 121)
    coarse_values = evaluate_on_grid(coarse_solution, coarse_mesh, plot_X, plot_Y)
    fine_values = evaluate_on_grid(fine_solution, fine_mesh, plot_X, plot_Y)
    vmin = float(min(coarse_values.min(), fine_values.min()))
    vmax = float(max(coarse_values.max(), fine_values.max()))
    levels = np.linspace(vmin, vmax, 31)

    coarse_plot = axes[1, 0].contourf(
        plot_X, plot_Y, coarse_values, levels=levels, cmap="plasma", extend="both"
    )
    overlay_mesh(axes[1, 0], coarse_mesh, linewidth=1.2)
    spatial_axis(
        axes[1, 0],
        rf"Coarse $p_h$ (maxh={coarse_maxh:g}, ndof={coarse_ndof})"
        + "\n"
        + rf"$Q_{{\mathrm{{coarse}}}}={coarse_qoi:.6e}$",
    )
    figure.colorbar(coarse_plot, ax=axes[1, 0], label=r"$p_h$")

    fine_plot = axes[1, 1].contourf(
        plot_X, plot_Y, fine_values, levels=levels, cmap="plasma", extend="both"
    )
    overlay_mesh(axes[1, 1], fine_mesh, linewidth=0.6)
    spatial_axis(
        axes[1, 1],
        rf"Fine $p_h$ (maxh={fine_maxh:g}, ndof={fine_ndof})"
        + "\n"
        + rf"$Q_{{\mathrm{{fine}}}}={fine_qoi:.6e}$",
    )
    figure.colorbar(fine_plot, ax=axes[1, 1], label=r"$p_h$")

    figure.suptitle(
        rf"Two single-level solves, same $\xi$"
        + rf"  |  $\Delta Q = Q_{{\mathrm{{fine}}}}-Q_{{\mathrm{{coarse}}}}"
        + rf" = {fine_qoi - coarse_qoi:.6e}$",
        fontsize=14,
    )
    return figure


def main(
    *,
    coarse_maxh: float = 0.30,
    fine_maxh: float = 0.15,
    grids: int = 16,
    num_modes: int = 100,
    sigma: float = 1.0,
    correlation_length: float = 0.30,
    seed: int = 7,
    show_plot: bool = True,
    plot_dir: Path = DEFAULT_PLOT_DIR,
) -> None:
    if fine_maxh >= coarse_maxh:
        raise ValueError("fine_maxh must be strictly smaller than coarse_maxh.")

    # ------------------------------------------------------------------
    # 1. One discrete KL draw → one kappa shared by both FE levels.
    # ------------------------------------------------------------------
    nx = ny = grids
    X, Y, points = cartesian_grid_2d(nx, ny)
    covariance = exponential_covariance(
        points,
        sigma=sigma,
        correlation_length=correlation_length,
    )
    eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=num_modes)

    log_kappa_values, xi = sample_discrete_kl(
        mean=0.0,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        shape=(ny, nx),
        rng=seed,
    )
    kappa_values = lognormal_transform(log_kappa_values)
    kappa = voxel_coefficient_2d(kappa_values, linear=True)

    # ------------------------------------------------------------------
    # 2. Two independent single-level FE solves (same kappa / same xi).
    # ------------------------------------------------------------------
    coarse_mesh, coarse_fes = build_fixed_mesh(maxh=coarse_maxh)
    fine_mesh, fine_fes = build_fixed_mesh(maxh=fine_maxh)

    coarse_solution = solve_diffusion(coarse_fes, kappa)
    fine_solution = solve_diffusion(fine_fes, kappa)

    coarse_qoi = quantity_of_interest(coarse_solution, coarse_mesh, kappa)
    fine_qoi = quantity_of_interest(fine_solution, fine_mesh, kappa)

    # ------------------------------------------------------------------
    # 3. Plot and report.
    # ------------------------------------------------------------------
    figure = make_plot(
        X=X,
        Y=Y,
        log_kappa_values=log_kappa_values,
        kappa_values=kappa_values,
        coarse_mesh=coarse_mesh,
        fine_mesh=fine_mesh,
        coarse_solution=coarse_solution,
        fine_solution=fine_solution,
        coarse_qoi=coarse_qoi,
        fine_qoi=fine_qoi,
        coarse_maxh=coarse_maxh,
        fine_maxh=fine_maxh,
        coarse_ndof=coarse_fes.ndof,
        fine_ndof=fine_fes.ndof,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "two_single_levels_same_xi.png"
    figure.savefig(plot_path, dpi=180)

    retained_variance = float(eigenvalues.sum() / np.trace(covariance))
    print(f"KL grid: {nx} x {ny}")
    print(f"KL modes: {len(eigenvalues)}")
    print(f"retained discrete variance: {retained_variance:.1%}")
    print(f"first five xi: {np.array2string(xi[:5], precision=3)}")
    print(f"kappa range: [{kappa_values.min():.3f}, {kappa_values.max():.3f}]")
    print(f"coarse: maxh={coarse_maxh:g}, ndof={coarse_fes.ndof}, Q={coarse_qoi:.6e}")
    print(f"fine:   maxh={fine_maxh:g}, ndof={fine_fes.ndof}, Q={fine_qoi:.6e}")
    print(f"Q_fine - Q_coarse = {fine_qoi - coarse_qoi:.6e}")
    print(f"saved plot: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-maxh", type=float, default=0.30)
    parser.add_argument("--fine-maxh", type=float, default=0.15)
    parser.add_argument("--grids", type=int, default=16)
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
        coarse_maxh=args.coarse_maxh,
        fine_maxh=args.fine_maxh,
        grids=args.grids,
        num_modes=args.num_modes,
        sigma=args.sigma,
        correlation_length=args.correlation_length,
        seed=args.seed,
        show_plot=not args.no_show,
        plot_dir=args.plot_dir,
    )
