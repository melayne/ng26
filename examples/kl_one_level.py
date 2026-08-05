"""One-level NGSolve diffusion problem with a discrete KL coefficient.

There is no mesh hierarchy and no multigrid solver in this example.  It does
only four things:

1. sample one discrete KL field on a Cartesian grid,
2. set ``kappa = exp(KL field)``,
3. assemble and directly solve one finite-element system,
4. plot the coefficient, mesh, KL spectrum, and solution.

Run from the project root with

    .venv/bin/python examples/kl_one_level.py

Pass ``--no-show`` to save the plot without opening a window.
"""
#%%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def mesh_triangulation(mesh: ng.Mesh) -> mtri.Triangulation:
    """Convert the NGSolve triangular mesh for Matplotlib."""
    points = np.array([vertex.point[:2] for vertex in mesh.vertices])
    triangles = np.array(
        [[vertex.nr for vertex in element.vertices] for element in mesh.Elements(ng.VOL)]
    )
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def overlay_mesh(ax, mesh: ng.Mesh) -> None:
    ax.triplot(mesh_triangulation(mesh), color="white", linewidth=0.7, alpha=0.9)


def spatial_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def evaluate_on_grid(coefficient, mesh, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Evaluate one scalar NGSolve coefficient on a plotting grid."""
    values = np.asarray(coefficient(mesh(X.ravel(), Y.ravel())))
    return values.reshape(X.shape)


def make_plot(
    X,
    Y,
    log_kappa_values,
    kappa_values,
    eigenvalues,
    total_variance,
    mesh,
    solution,
):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    # Plot the Gaussian random field before exponentiation.
    log_limit = max(abs(log_kappa_values.min()), abs(log_kappa_values.max()))
    log_plot = axes[0, 0].contourf(
        X,
        Y,
        log_kappa_values,
        levels=np.linspace(-log_limit, log_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    axes[0, 0].scatter(X, Y, s=4, color="black", alpha=0.3)
    spatial_axis(axes[0, 0], r"Gaussian KL field $Z=\log(\kappa)$")
    figure.colorbar(log_plot, ax=axes[0, 0], label=r"$Z(x,\omega)$")

    # Plot the positive diffusion coefficient with the one FE mesh over it.
    coefficient_plot = axes[0, 1].contourf(
        X,
        Y,
        kappa_values,
        levels=31,
        cmap="viridis",
    )
    overlay_mesh(axes[0, 1], mesh)
    spatial_axis(axes[0, 1], r"$\kappa=\exp(Z)$ with the NGSolve mesh")
    figure.colorbar(coefficient_plot, ax=axes[0, 1], label=r"$\kappa(x,\omega)$")

    # Plot the retained eigenvalues and cumulative discrete variance.
    modes = np.arange(1, len(eigenvalues) + 1)
    eigenvalue_line = axes[1, 0].semilogy(
        modes,
        eigenvalues,
        "o-",
        markersize=3,
        color="tab:blue",
        label="eigenvalue",
    )
    axes[1, 0].set_xlabel("KL mode")
    axes[1, 0].set_ylabel("eigenvalue", color="tab:blue")
    axes[1, 0].set_title("Discrete KL spectrum")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    variance_axis = axes[1, 0].twinx()
    cumulative_variance = 100.0 * np.cumsum(eigenvalues) / total_variance
    variance_line = variance_axis.plot(
        modes,
        cumulative_variance,
        "--",
        color="tab:orange",
        label="retained variance",
    )
    variance_axis.set_ylabel("cumulative variance (%)", color="tab:orange")
    variance_axis.set_ylim(0.0, 100.0)
    axes[1, 0].legend(
        eigenvalue_line + variance_line,
        ["eigenvalue", "retained variance"],
    )

    # Evaluate and plot the directly solved FE solution.
    solution_X, solution_Y, _ = cartesian_grid_2d(121, 121)
    solution_values = evaluate_on_grid(solution, mesh, solution_X, solution_Y)
    solution_plot = axes[1, 1].contourf(
        solution_X,
        solution_Y,
        solution_values,
        levels=31,
        cmap="plasma",
        extend="both",
    )
    overlay_mesh(axes[1, 1], mesh)
    spatial_axis(axes[1, 1], "Direct finite-element solution")
    figure.colorbar(solution_plot, ax=axes[1, 1], label=r"$u_h(x,\omega)$")

    figure.suptitle("One-level discrete-KL diffusion example", fontsize=16)
    return figure


def main(*, show_plot: bool = True, plot_dir: Path = DEFAULT_PLOT_DIR) -> None:
    # ------------------------------------------------------------------
    # 1. Sample one discrete KL field on a regular background grid.
    # ------------------------------------------------------------------
    nx = ny = 16
    X, Y, points = cartesian_grid_2d(nx, ny)

    covariance = exponential_covariance(
        points,
        sigma=1.0,
        correlation_length=0.30,
    )
    eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=100)

    log_kappa_values, xi = sample_discrete_kl(
        mean=0.0,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        shape=(ny, nx),
        rng=7,
    )

    # The exponential makes kappa positive, which keeps the diffusion problem elliptic.
    kappa_values = lognormal_transform(log_kappa_values)
    kappa = voxel_coefficient_2d(kappa_values, linear=True)

    # ------------------------------------------------------------------
    # 2. Create one NGSolve mesh and one H1 finite-element space.
    # ------------------------------------------------------------------
    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.18))
    fes = ng.H1(
        mesh,
        order=1,
        dirichlet="left|right|top|bottom",
    )
    u, v = fes.TnT()

    # ------------------------------------------------------------------
    # 3. Assemble -div(kappa grad(u)) = 1 with u = 0 on the boundary.
    # ------------------------------------------------------------------
    a = ng.BilinearForm(fes)
    a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
    a.Assemble()

    f = ng.LinearForm(fes)
    f += v * ng.dx
    f.Assemble()

    # ------------------------------------------------------------------
    # 4. Solve A*u = f directly on the free DOFs.
    # ------------------------------------------------------------------
    solution = ng.GridFunction(fes)
    inverse = a.mat.Inverse(fes.FreeDofs())
    solution.vec.data = inverse * f.vec

    # Check the free-DOF algebraic residual after the direct solve.
    residual = solution.vec.CreateVector()
    residual.data = f.vec - a.mat * solution.vec
    residual_values = residual.FV().NumPy()
    free_ids = np.flatnonzero(np.asarray(fes.FreeDofs()))
    residual_norm = float(np.linalg.norm(residual_values[free_ids]))

    # ------------------------------------------------------------------
    # 5. Plot and save the result.
    # ------------------------------------------------------------------
    figure = make_plot(
        X,
        Y,
        log_kappa_values,
        kappa_values,
        eigenvalues,
        float(np.trace(covariance)),
        mesh,
        solution,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "kl_one_level.png"
    figure.savefig(plot_path, dpi=180)

    retained_variance = eigenvalues.sum() / np.trace(covariance)
    print(f"KL grid: {nx} x {ny}")
    print(f"KL modes: {len(eigenvalues)}")
    print(f"retained discrete variance: {retained_variance:.1%}")
    print(f"first five xi: {np.array2string(xi[:5], precision=3)}")
    print(f"kappa range: [{kappa_values.min():.3f}, {kappa_values.max():.3f}]")
    print(f"FE degrees of freedom: {fes.ndof}")
    print(f"direct-solve residual: {residual_norm:.3e}")
    print(f"saved plot: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the plot without opening an interactive window",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="directory in which to save the PNG figure",
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    arguments = parse_args()
    main(show_plot=not arguments.no_show, plot_dir=arguments.plot_dir)

# %%
