"""Solve the variable-coefficient Poisson/reaction problem from the meeting notebook.

The model is

    -div(k grad(u)) + sigma^2 u = 0       in (0, 1)^2,
    k(x, omega) = exp(Z(x, omega)),
    sigma^2 = 0.25,

with u = 1 on the left boundary and u = 0 on the other three boundaries.
The Gaussian field Z is one discrete Karhunen-Loeve (KL) realization with an
exponential covariance, matching ``kl_two_level.py``.
It uses continuous, piecewise-linear finite elements and repeated two-level
geometric multigrid V-cycles.  A direct finite-element solve is also computed
as an independent algebraic check.

Run from the project root with

    .venv/bin/python examples/poisson_reaction_two_level.py

Pass ``--no-show`` when running headlessly.
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
from multigrid_cycles import (  # noqa: E402
    MultigridHierarchy,
    MultigridSolver,
    VCycleConfig,
    build_form_setup,
    build_hierarchy,
)


@dataclass
class RandomFieldSample:
    """One discrete KL realization and its NGSolve coefficient function."""

    X: np.ndarray
    Y: np.ndarray
    log_kappa_values: np.ndarray
    kappa_values: np.ndarray
    kappa: object
    xi: np.ndarray
    retained_variance: float


@dataclass
class ModelResult:
    """Numerical state and diagnostics returned by :func:`solve_model`."""

    hierarchy: MultigridHierarchy
    random_field: RandomFieldSample
    residuals: np.ndarray
    direct_solution: ng.GridFunction
    direct_l2_difference: float


def sample_random_field(*, random_seed: int = 7) -> RandomFieldSample:
    """Sample the lognormal coefficient used on every finite-element level."""
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
        rng=random_seed,
    )
    kappa_values = lognormal_transform(log_kappa_values)
    kappa = voxel_coefficient_2d(kappa_values, linear=True)
    retained_variance = float(eigenvalues.sum() / np.trace(covariance))
    return RandomFieldSample(
        X=X,
        Y=Y,
        log_kappa_values=log_kappa_values,
        kappa_values=kappa_values,
        kappa=kappa,
        xi=xi,
        retained_variance=retained_variance,
    )


def build_model(
    *,
    coarse_maxh: float = 0.2,
    random_seed: int = 7,
) -> tuple[MultigridHierarchy, RandomFieldSample]:
    """Sample one coefficient and assemble it on two finite-element levels."""
    random_field = sample_random_field(random_seed=random_seed)
    kappa = random_field.kappa
    sigma2 = 0.25

    def reaction_diffusion_form(a, u, v):
        a += (
            kappa * ng.InnerProduct(ng.grad(u), ng.grad(v))
            + sigma2 * u * v
        ) * ng.dx

    # The source f is zero, so the assembled LinearForm remains empty.
    form_setup = build_form_setup(bilinear=reaction_diffusion_form)
    coarse_mesh = ng.Mesh(unit_square.GenerateMesh(maxh=coarse_maxh))

    # Dictionary insertion order resolves corner values.  The left condition is
    # applied last because the mathematical statement assigns both corners on
    # x=0 to the left boundary.
    boundary_values = {
        "right": 0.0,
        "top": 0.0,
        "bottom": 0.0,
        "left": 1.0,
    }
    hierarchy = build_hierarchy(
        coarse_mesh,
        form_setup,
        n_refines=1,
        order=1,
        dirichlet="left|right|top|bottom",
        dirichlet_value=boundary_values,
        verbose=True,
    )
    return hierarchy, random_field


def direct_solve(finest_level) -> ng.GridFunction:
    """Solve the same finite-element equations directly on the fine grid."""
    direct = ng.GridFunction(finest_level.fes)
    finest_level.enforce_dirichlet(direct.vec)

    modified_rhs = finest_level.f.vec.CreateVector()
    modified_rhs.data = finest_level.f.vec - finest_level.a.mat * direct.vec
    correction = (
        finest_level.a.mat.Inverse(freedofs=finest_level.fes.FreeDofs())
        * modified_rhs
    )
    direct.vec.data += correction
    finest_level.enforce_dirichlet(direct.vec)
    return direct


def solve_model(
    *,
    coarse_maxh: float = 0.2,
    random_seed: int = 7,
    max_cycles: int = 30,
    relative_tolerance: float = 1.0e-10,
) -> ModelResult:
    """Build and solve the two-level model, then check it against a direct solve."""
    hierarchy, random_field = build_model(
        coarse_maxh=coarse_maxh,
        random_seed=random_seed,
    )
    fine = hierarchy.finest

    # The nonzero Dirichlet extension supplies the effective right-hand side
    # f - A u_D even though the volume source f is zero.
    fine.enforce_dirichlet()
    initial_residual = fine.residual_norm()

    solver = MultigridSolver(
        hierarchy,
        VCycleConfig(pre_sweeps=2, post_sweeps=2, coarse_direct=True),
    )
    history, _ = solver.solve(
        max_cycles=max_cycles,
        tol=relative_tolerance,
        norms=("l2",),
        verbose=True,
    )
    residuals = np.concatenate(([initial_residual], history["l2"]))

    direct = direct_solve(fine)
    direct_l2_difference = float(
        np.sqrt(ng.Integrate((fine.gfu - direct) ** 2, fine.mesh))
    )
    return ModelResult(
        hierarchy=hierarchy,
        random_field=random_field,
        residuals=residuals,
        direct_solution=direct,
        direct_l2_difference=direct_l2_difference,
    )


def mesh_triangulation(mesh: ng.Mesh) -> mtri.Triangulation:
    """Convert a triangular NGSolve mesh into a Matplotlib triangulation."""
    points = np.asarray([vertex.point[:2] for vertex in mesh.vertices])
    triangles = np.asarray(
        [[vertex.nr for vertex in element.vertices] for element in mesh.Elements(ng.VOL)]
    )
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def evaluate_on_grid(coefficient, mesh: ng.Mesh, X, Y) -> np.ndarray:
    """Evaluate an NGSolve coefficient on a Cartesian plotting grid."""
    values = np.asarray(coefficient(mesh(X.ravel(), Y.ravel())))
    return values.reshape(X.shape)


def make_figure(result: ModelResult, relative_tolerance: float):
    """Plot the KL field, lognormal coefficient, solution, and convergence."""
    fine = result.hierarchy.finest
    random_field = result.random_field
    coordinates = np.linspace(0.0, 1.0, 151)
    X, Y = np.meshgrid(coordinates, coordinates)
    solution_values = evaluate_on_grid(fine.gfu, fine.mesh, X, Y)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    log_limit = max(
        abs(random_field.log_kappa_values.min()),
        abs(random_field.log_kappa_values.max()),
    )
    log_plot = axes[0, 0].contourf(
        random_field.X,
        random_field.Y,
        random_field.log_kappa_values,
        levels=np.linspace(-log_limit, log_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    axes[0, 0].scatter(
        random_field.X,
        random_field.Y,
        s=4,
        color="black",
        alpha=0.30,
    )
    axes[0, 0].set_title(r"Gaussian KL field $Z=\log(\kappa)$")
    figure.colorbar(log_plot, ax=axes[0, 0], label=r"$Z(x,\omega)$")

    coefficient_plot = axes[0, 1].contourf(
        random_field.X,
        random_field.Y,
        random_field.kappa_values,
        levels=31,
        cmap="viridis",
        extend="both",
    )
    axes[0, 1].triplot(
        mesh_triangulation(fine.mesh),
        color="white",
        linewidth=0.35,
        alpha=0.65,
    )
    axes[0, 1].set_title(r"Lognormal coefficient $\kappa=\exp(Z)$")
    figure.colorbar(coefficient_plot, ax=axes[0, 1], label=r"$\kappa(x,\omega)$")

    solution_plot = axes[1, 0].contourf(
        X,
        Y,
        solution_values,
        levels=31,
        cmap="plasma",
    )
    axes[1, 0].triplot(
        mesh_triangulation(fine.mesh),
        color="white",
        linewidth=0.35,
        alpha=0.65,
    )
    axes[1, 0].set_title("Fine-grid multigrid solution")
    figure.colorbar(solution_plot, ax=axes[1, 0], label=r"$u_h(x,\omega)$")

    cycles = np.arange(len(result.residuals))
    axes[1, 1].semilogy(cycles, result.residuals, "o-", color="tab:purple")
    axes[1, 1].axhline(
        relative_tolerance * result.residuals[0],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="stopping threshold",
    )
    axes[1, 1].set_xlabel("V-cycle")
    axes[1, 1].set_ylabel(r"free-DOF residual $\|b-Au\|_2$")
    axes[1, 1].set_title("Two-level convergence")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    figure.suptitle(
        r"$-\nabla\cdot(\kappa(x,\omega)\nabla u)+0.25u=0$ on $(0,1)^2$",
        fontsize=15,
    )
    return figure


def main(
    *,
    show_plots: bool = True,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    random_seed: int = 7,
) -> None:
    """Run the model, save its diagnostic figure, and print verification data."""
    relative_tolerance = 1.0e-10
    result = solve_model(
        relative_tolerance=relative_tolerance,
        random_seed=random_seed,
    )
    figure = make_figure(result, relative_tolerance)

    plot_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plot_dir / "poisson_reaction_two_level.png"
    figure.savefig(figure_path, dpi=180)

    fine = result.hierarchy.finest
    reduction = result.residuals[-1] / result.residuals[0]
    print(f"random seed: {random_seed}")
    print(f"retained discrete KL variance: {result.random_field.retained_variance:.1%}")
    print(
        "kappa range on KL grid: "
        f"[{result.random_field.kappa_values.min():.3f}, "
        f"{result.random_field.kappa_values.max():.3f}]"
    )
    print(f"final free-DOF residual: {result.residuals[-1]:.3e}")
    print(f"relative residual: {reduction:.3e}")
    print(f"L2 difference from direct FE solve: {result.direct_l2_difference:.3e}")
    print(f"fine-grid degrees of freedom: {fine.ndof}")
    print(f"saved plot: {figure_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive window",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed for the discrete KL realization",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="directory in which to save the PNG figure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        show_plots=not args.no_show,
        plot_dir=args.plot_dir,
        random_seed=args.seed,
    )
