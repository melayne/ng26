"""Two-level diffusion problem with one discrete KL realization.

Run from the project root with

    .venv/bin/python examples/kl_two_level.py

The script saves three figures in ``examples/plots`` and displays them.  Pass
``--no-show`` when running headlessly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DEFAULT_PLOT_DIR = PROJECT_ROOT / "examples" / "plots"
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

# from KL_expansion import (  # noqa: E402
#     cartesian_grid_2d,
#     exponential_covariance,
#     leading_eigenpairs,
#     lognormal_transform,
#     sample_discrete_kl,
#     voxel_coefficient_2d,
# )
# from multigrid_cycles import (  # noqa: E402
#     MultigridSolver,
#     VCycleConfig,
#     build_form_setup,
#     build_hierarchy,
# )

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = PROJECT_ROOT / "examples" / "plots"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import (  # noqa: E402
    cartesian_grid_2d,
    lognormal_transform,
    voxel_coefficient_2d,
)

from examples.KLMC.utils.analytical_eigenfunctions import (  # noqa: E402
    get_1d_eigenpairs,
    leading_2d_eigenpairs,
    make_2d_kl_evaluator,
)

from multigrid_cycles import (  # noqa: E402
    MultigridSolver,
    VCycleConfig,
    build_form_setup,
    build_hierarchy,
)


def mesh_triangulation(mesh: ng.Mesh) -> mtri.Triangulation:
    """Convert a triangular NGSolve mesh into a Matplotlib triangulation."""
    points = np.array([vertex.point[:2] for vertex in mesh.vertices])
    triangles = np.array(
        [[vertex.nr for vertex in element.vertices] for element in mesh.Elements(ng.VOL)]
    )
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def overlay_mesh(ax, mesh: ng.Mesh, *, color: str = "white", linewidth: float = 0.7) -> None:
    """Draw element edges over an existing Matplotlib field plot."""
    ax.triplot(
        mesh_triangulation(mesh),
        color=color,
        linewidth=linewidth,
        alpha=0.9,
    )


def evaluate_on_grid(coefficient, mesh: ng.Mesh, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Evaluate a scalar NGSolve coefficient on a Cartesian plotting grid."""
    values = np.asarray(coefficient(mesh(X.ravel(), Y.ravel())))
    return values.reshape(X.shape)


def finish_spatial_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def make_field_figure(
    X: np.ndarray,
    Y: np.ndarray,
    log_kappa_values: np.ndarray,
    kappa_values: np.ndarray,
    hierarchy,
):
    """Plot the sampled field and its relation to both FE meshes."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    log_limit = max(abs(log_kappa_values.min()), abs(log_kappa_values.max()))
    log_levels = np.linspace(-log_limit, log_limit, 31)
    log_plot = axes[0, 0].contourf(
        X,
        Y,
        log_kappa_values,
        levels=log_levels,
        cmap="coolwarm",
        extend="both",
    )
    axes[0, 0].scatter(X, Y, s=4, color="black", alpha=0.30)
    finish_spatial_axis(axes[0, 0], r"Gaussian KL field $Z=\log(\kappa)$")
    figure.colorbar(log_plot, ax=axes[0, 0], label=r"$Z(x,\omega)$")

    kappa_levels = np.linspace(kappa_values.min(), kappa_values.max(), 31)
    for ax, title in (
        (axes[0, 1], r"Lognormal coefficient $\kappa=\exp(Z)$"),
        (axes[1, 0], r"$\kappa$ with coarse NGSolve mesh"),
        (axes[1, 1], r"$\kappa$ with fine NGSolve mesh"),
    ):
        kappa_plot = ax.contourf(
            X,
            Y,
            kappa_values,
            levels=kappa_levels,
            cmap="viridis",
            extend="both",
        )
        finish_spatial_axis(ax, title)

    overlay_mesh(axes[1, 0], hierarchy.coarsest.mesh, linewidth=1.3)
    overlay_mesh(axes[1, 1], hierarchy.finest.mesh, linewidth=0.7)
    figure.colorbar(kappa_plot, ax=axes[0, 1], label=r"$\kappa(x,\omega)$")
    figure.colorbar(kappa_plot, ax=axes[1, 0], label=r"$\kappa(x,\omega)$")
    figure.colorbar(kappa_plot, ax=axes[1, 1], label=r"$\kappa(x,\omega)$")
    figure.suptitle("One random coefficient, evaluated on both multigrid levels", fontsize=15)
    return figure


def make_diagnostics_figure(
    eigenvalues: np.ndarray,
    total_variance: float,
    residuals: np.ndarray,
    relative_tolerance: float,
    fine_level,
):
    """Plot KL truncation, V-cycle convergence, and the computed solution."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)

    mode_numbers = np.arange(1, len(eigenvalues) + 1)
    spectrum_line = axes[0].semilogy(
        mode_numbers,
        eigenvalues,
        "o-",
        color="tab:blue",
        label=r"eigenvalue $\lambda_j$",
    )
    axes[0].set_xlabel("KL mode")
    axes[0].set_ylabel("eigenvalue", color="tab:blue")
    axes[0].grid(True, which="both", alpha=0.3)

    variance_axis = axes[0].twinx()
    cumulative_variance = 100.0 * np.cumsum(eigenvalues) / total_variance
    variance_line = variance_axis.plot(
        mode_numbers,
        cumulative_variance,
        "s--",
        color="tab:orange",
        label="retained variance",
    )
    variance_axis.set_ylabel("cumulative variance (%)", color="tab:orange")
    variance_axis.set_ylim(0.0, 100.0)
    axes[0].legend(spectrum_line + variance_line, ["eigenvalue", "retained variance"])
    axes[0].set_title("KL spectrum and truncation")

    cycles = np.arange(len(residuals))
    axes[1].semilogy(cycles, residuals, "o-", color="tab:purple")
    axes[1].axhline(
        relative_tolerance * residuals[0],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="stopping threshold",
    )
    axes[1].set_xlabel("V-cycle")
    axes[1].set_ylabel(r"residual $\|b-Au\|_2$")
    axes[1].set_title("Two-level V-cycle convergence")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    solution_x, solution_y, _ = cartesian_grid_2d(121, 121)
    solution_values = evaluate_on_grid(
        fine_level.gfu,
        fine_level.mesh,
        solution_x,
        solution_y,
    )
    solution_plot = axes[2].contourf(
        solution_x,
        solution_y,
        solution_values,
        levels=31,
        cmap="plasma",
    )
    overlay_mesh(axes[2], fine_level.mesh, color="white", linewidth=0.5)
    finish_spatial_axis(axes[2], "Computed fine-grid solution")
    figure.colorbar(solution_plot, ax=axes[2], label=r"$u_h(x,\omega)$")
    return figure


def make_initial_guess_figure(coarse_level, fine_level, coarse_solution, initial_guess):
    """Plot the coarse solve and the initial guess obtained with ``P*u_c``."""
    plot_x, plot_y, _ = cartesian_grid_2d(121, 121)
    coarse_values = evaluate_on_grid(
        coarse_solution,
        coarse_level.mesh,
        plot_x,
        plot_y,
    )
    initial_values = evaluate_on_grid(
        initial_guess,
        fine_level.mesh,
        plot_x,
        plot_y,
    )

    field_min = min(coarse_values.min(), initial_values.min())
    field_max = max(coarse_values.max(), initial_values.max())
    field_levels = np.linspace(field_min, field_max, 31)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    for ax, values, mesh, title, linewidth in (
        (
            axes[0],
            coarse_values,
            coarse_level.mesh,
            r"Coarse FE solve $u_c=A_c^{-1}b_c$",
            1.3,
        ),
        (
            axes[1],
            initial_values,
            fine_level.mesh,
            r"Fine initial guess $x_f^{(0)}=P u_c$",
            0.7,
        ),
    ):
        field_plot = ax.contourf(
            plot_x,
            plot_y,
            values,
            levels=field_levels,
            cmap="plasma",
            extend="both",
        )
        overlay_mesh(ax, mesh, linewidth=linewidth)
        finish_spatial_axis(ax, title)
        figure.colorbar(field_plot, ax=ax, label="FE solution value")

    difference = initial_values - coarse_values
    difference_limit = max(float(np.max(np.abs(difference))), 1.0e-16)
    difference_plot = axes[2].contourf(
        plot_x,
        plot_y,
        difference,
        levels=np.linspace(-difference_limit, difference_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    overlay_mesh(axes[2], fine_level.mesh, linewidth=0.7)
    finish_spatial_axis(axes[2], r"$P u_c-u_c$ in physical space")
    figure.colorbar(difference_plot, ax=axes[2], label="pointwise difference")
    figure.suptitle(
        "The FE solution is prolonged; the KL coefficient is unchanged",
        fontsize=15,
    )
    return figure


def main(*, show_plots: bool = True, plot_dir: Path = DEFAULT_PLOT_DIR) -> None:
    # ------------------------------------------------------------------
    # 1. Construct one discrete KL realization of log(kappa).
    # ------------------------------------------------------------------
    # nx = ny = 16
    # X, Y, points = cartesian_grid_2d(nx, ny)

    # covariance = exponential_covariance(
    #     points,
    #     sigma=1.,
    #     correlation_length=0.30,
    # )
    # eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=100)
    # log_kappa_values, xi = sample_discrete_kl(
    #     mean=0.0,
    #     eigenvalues=eigenvalues,
    #     eigenvectors=eigenvectors,
    #     shape=(ny, nx),
    #     rng=7,
    # )
    nx = ny = 100
    X, Y, _ = cartesian_grid_2d(nx, ny)

    correlation_length = 0.30
    standard_deviation = 1.0
    variance = standard_deviation**2

    num_modes_2d = 1000

    # Using num_modes_2d 1D modes is conservative but guarantees
    # enough tensor-product candidates.
    num_modes_1d = num_modes_2d

    (
        frequencies_1d,
        normalizations_1d,
        eigenvalues_1d,
        _,
    ) = get_1d_eigenpairs(
        num_modes=num_modes_1d,
        correlation_length=correlation_length,
    )

    (
        unit_eigenvalues_2d,
        mode_indices,
        evaluate_eigenfunctions_2d,
    ) = leading_2d_eigenpairs(
        eigenvalues_1d=eigenvalues_1d,
        frequencies_1d=frequencies_1d,
        normalizations_1d=normalizations_1d,
        correlation_length=correlation_length,
        num_modes_2d=num_modes_2d,
        method="heap",
    )

    # These are the actual covariance eigenvalues after applying sigma².
    eigenvalues = (
        variance * unit_eigenvalues_2d
    )

    evaluate_log_conductivity = make_2d_kl_evaluator(
        eigenvalues_2d=unit_eigenvalues_2d,
        eigenfunction_evaluator=evaluate_eigenfunctions_2d,
        mean_log_conductivity=0.0,
        variance=variance,
    )

    rng = np.random.default_rng(seed=7)

    # One Gaussian coefficient for every selected 2D eigenfunction.
    xi = rng.standard_normal(num_modes_2d)

    log_kappa_values = evaluate_log_conductivity(
        X,
        Y,
        xi,
    )
    # exp(log(kappa)) makes the diffusion coefficient strictly positive.
    kappa_values = lognormal_transform(log_kappa_values)
    kappa = voxel_coefficient_2d(kappa_values, linear=True)

    # ------------------------------------------------------------------
    # 2. Define -div(kappa grad(u)) = 1 with u = 0 on the boundary.
    #    Both levels close over and assemble with this same kappa object.
    # ------------------------------------------------------------------
    def diffusion_form(a, u, v):
        a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx

    def load_form(f, u, v):
        f += v * ng.dx

    form_setup = build_form_setup(bilinear=diffusion_form, linear=load_form)

    # n_refines=1 means exactly two levels: the initial mesh and one refinement.
    coarse_mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.1))
    hierarchy = build_hierarchy(
        coarse_mesh,
        form_setup,
        n_refines=3,
        order=1,
        dirichlet="left|right|top|bottom",
        dirichlet_value=0.0,
        verbose=True,
    )
    # assert hierarchy.nlevels == 2

    # The same physical point has the same coefficient value on both meshes.
    probe = (0.37, 0.42)
    probe_values = np.array(
        [float(kappa(level.mesh(*probe))) for level in hierarchy.levels]
    )
    np.testing.assert_allclose(probe_values, probe_values[0])

    # ------------------------------------------------------------------
    # 3. Solve the coarse FE problem and prolong it as the fine initial guess.
    #    P transfers the FE solution only; it does not act on the KL field.
    # ------------------------------------------------------------------
    coarse = hierarchy.coarsest
    fine = hierarchy.finest

    coarse_solution = ng.GridFunction(coarse.fes)
    coarse_solution.vec.data = coarse.a.mat.Inverse(coarse.fes.FreeDofs()) * coarse.f.vec
    coarse.enforce_dirichlet(coarse_solution.vec)

    initial_guess = ng.GridFunction(fine.fes)
    initial_guess.vec.data = fine.P * coarse_solution.vec
    fine.enforce_dirichlet(initial_guess.vec)
    fine.gfu.vec.data = initial_guess.vec

    # ------------------------------------------------------------------
    # 4. Continue from that initial guess using repeated two-level V-cycles.
    # ------------------------------------------------------------------
    relative_tolerance = 1.0e-8
    initial_residual = hierarchy.finest.residual_norm()
    solver = MultigridSolver(
        hierarchy,
        VCycleConfig(pre_sweeps=2, post_sweeps=2, coarse_direct=True),
    )
    history, _ = solver.solve(
        max_cycles=12,
        tol=relative_tolerance,
        norms=("l2",),
        verbose=True,
    )
    residuals = np.concatenate(([initial_residual], history["l2"]))

    # ------------------------------------------------------------------
    # 5. Make and save the field, initial-guess, and diagnostic plots.
    # ------------------------------------------------------------------
    field_figure = make_field_figure(
        X,
        Y,
        log_kappa_values,
        kappa_values,
        hierarchy,
    )
    diagnostics_figure = make_diagnostics_figure(
        eigenvalues,
        variance,
        residuals,
        relative_tolerance,
        hierarchy.finest,
    )
    initial_guess_figure = make_initial_guess_figure(
        coarse,
        fine,
        coarse_solution,
        initial_guess,
    )

    plot_dir.mkdir(parents=True, exist_ok=True)
    field_path = plot_dir / "kl_two_level_fields.png"
    diagnostics_path = plot_dir / "kl_two_level_diagnostics.png"
    initial_guess_path = plot_dir / "kl_two_level_initial_guess.png"
    field_figure.savefig(field_path, dpi=180)
    diagnostics_figure.savefig(diagnostics_path, dpi=180)
    initial_guess_figure.savefig(initial_guess_path, dpi=180)

    retained_variance = eigenvalues.sum() / variance
    print(f"KL coefficients xi: {np.array2string(xi, precision=3)}")
    print(f"retained discrete variance: {retained_variance:.1%}")
    print(f"kappa range on KL grid: [{kappa_values.min():.3f}, {kappa_values.max():.3f}]")
    print(f"kappa{probe} on coarse/fine meshes: {probe_values}")
    print(f"final residual: {history['l2'][-1]:.3e}")
    print(f"saved field plot: {field_path}")
    print(f"saved diagnostics plot: {diagnostics_path}")
    print(f"saved initial-guess plot: {initial_guess_path}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figures without opening interactive windows",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="directory in which to save the PNG figures",
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    arguments = parse_args()
    main(show_plots=not arguments.no_show, plot_dir=arguments.plot_dir)
