"""Manual two-level KL diffusion solve, written as explicit V-cycle steps.

This example deliberately does not use ``MultigridSolver`` or ``v_cycle``.
It reuses the existing level builder, then performs the two-level algorithm
line by line:

1. pre-smooth on the fine grid,
2. compute the fine residual,
3. restrict it with ``P.T``,
4. solve the coarse error equation,
5. prolong the coarse correction with ``P``,
6. update the fine solution,
7. post-smooth on the fine grid.

Run from the project root with

    .venv/bin/python examples/kl_two_level_manual.py

Pass ``--no-show`` to save the plot without opening a window.
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
from multigrid_cycles import build_form_setup, build_hierarchy  # noqa: E402


def zero_fixed_dofs(level, vector) -> None:
    """A residual or correction must be zero on Dirichlet DOFs."""
    vector.FV().NumPy()[level.fixed_ids] = 0.0


def residual(level, b, x):
    """Return the algebraic residual r = b - A*x on one level."""
    r = x.CreateVector()
    r.data = b - level.a.mat * x
    zero_fixed_dofs(level, r)
    return r


def free_l2_norm(level, vector) -> float:
    """Euclidean norm using only unconstrained entries."""
    values = vector.FV().NumPy()
    return float(np.linalg.norm(values[level.free_ids]))


def manual_two_level_cycle(
    coarse,
    fine,
    b_fine,
    x_fine,
    fine_smoother,
    coarse_inverse,
    *,
    cycle_number: int,
    pre_sweeps: int = 2,
    post_sweeps: int = 2,
) -> tuple[list[str], list[float]]:
    """Perform one two-level correction with every algebraic step visible."""
    stage_names: list[str] = []
    stage_residuals: list[float] = []

    print(f"\n--- manual cycle {cycle_number} ---")

    # ------------------------------------------------------------------
    # Step 1: pre-smooth A_f x_f = b_f.
    # Each update is x_f <- x_f + S_f (b_f - A_f x_f).
    # ------------------------------------------------------------------
    for sweep in range(1, pre_sweeps + 1):
        r_fine = residual(fine, b_fine, x_fine)

        smooth_correction = x_fine.CreateVector()
        smooth_correction.data = fine_smoother * r_fine
        zero_fixed_dofs(fine, smooth_correction)

        x_fine.data += smooth_correction

        value = free_l2_norm(fine, residual(fine, b_fine, x_fine))
        stage_names.append(f"C{cycle_number} pre{sweep}")
        stage_residuals.append(value)
        print(f"pre-smooth {sweep}:      ||r_f||_2 = {value:.6e}")

    # ------------------------------------------------------------------
    # Step 2: form the fine-grid residual r_f = b_f - A_f x_f.
    # ------------------------------------------------------------------
    r_fine = residual(fine, b_fine, x_fine)

    # ------------------------------------------------------------------
    # Step 3: restrict the residual to the coarse dual space.
    #
    #                    r_c = P^T r_f
    # ------------------------------------------------------------------
    r_coarse = fine.PT.CreateColVector()
    r_coarse.data = fine.PT * r_fine
    zero_fixed_dofs(coarse, r_coarse)
    print(
        f"restrict residual:   {len(r_fine)} -- "
        f"P^T({fine.PT.height}x{fine.PT.width}) --> {len(r_coarse)}"
    )

    # ------------------------------------------------------------------
    # Step 4: solve the coarse error equation exactly.
    #
    #                    A_c e_c = r_c
    # ------------------------------------------------------------------
    e_coarse = r_coarse.CreateVector()
    e_coarse.data = coarse_inverse * r_coarse
    zero_fixed_dofs(coarse, e_coarse)
    coarse_equation_residual = residual(coarse, r_coarse, e_coarse)
    print(
        "coarse solve:        "
        f"||r_c - A_c e_c||_2 = {free_l2_norm(coarse, coarse_equation_residual):.6e}"
    )

    # ------------------------------------------------------------------
    # Step 5: prolong the coarse error into the fine space.
    #
    #                    e_f = P e_c
    # ------------------------------------------------------------------
    e_fine = fine.P.CreateColVector()
    e_fine.data = fine.P * e_coarse
    zero_fixed_dofs(fine, e_fine)
    print(
        f"prolong correction:  {len(e_coarse)} -- "
        f"P({fine.P.height}x{fine.P.width}) --> {len(e_fine)}"
    )

    # ------------------------------------------------------------------
    # Step 6: correct the fine approximation.
    #
    #                    x_f <- x_f + e_f
    #
    # The coarse correction targets the smooth error component.  It need not
    # decrease the Euclidean fine-grid residual at this intermediate stage;
    # judge the method after the complete cycle, including post-smoothing.
    # ------------------------------------------------------------------
    x_fine.data += e_fine
    value = free_l2_norm(fine, residual(fine, b_fine, x_fine))
    stage_names.append(f"C{cycle_number} coarse")
    stage_residuals.append(value)
    print(f"after correction:    ||r_f||_2 = {value:.6e}")

    # ------------------------------------------------------------------
    # Step 7: post-smooth on the corrected fine approximation.
    # ------------------------------------------------------------------
    for sweep in range(1, post_sweeps + 1):
        r_fine = residual(fine, b_fine, x_fine)

        smooth_correction = x_fine.CreateVector()
        smooth_correction.data = fine_smoother * r_fine
        zero_fixed_dofs(fine, smooth_correction)

        x_fine.data += smooth_correction

        value = free_l2_norm(fine, residual(fine, b_fine, x_fine))
        stage_names.append(f"C{cycle_number} post{sweep}")
        stage_residuals.append(value)
        print(f"post-smooth {sweep}:     ||r_f||_2 = {value:.6e}")

    return stage_names, stage_residuals


def mesh_triangulation(mesh: ng.Mesh) -> mtri.Triangulation:
    points = np.array([vertex.point[:2] for vertex in mesh.vertices])
    triangles = np.array(
        [[vertex.nr for vertex in element.vertices] for element in mesh.Elements(ng.VOL)]
    )
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def overlay_mesh(ax, mesh: ng.Mesh, *, linewidth: float) -> None:
    ax.triplot(mesh_triangulation(mesh), color="white", linewidth=linewidth, alpha=0.9)


def spatial_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def evaluate_on_grid(coefficient, mesh, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficient(mesh(X.ravel(), Y.ravel())))
    return values.reshape(X.shape)


def make_plot(
    X,
    Y,
    log_kappa_values,
    kappa_values,
    hierarchy,
    stage_residuals,
    cycle_end_indices,
):
    """Summarize the coefficient, meshes, manual stages, and solution."""
    figure, axes = plt.subplot_mosaic(
        [["log", "coarse", "fine"], ["residual", "residual", "solution"]],
        figsize=(16, 9),
        constrained_layout=True,
    )

    log_limit = max(abs(log_kappa_values.min()), abs(log_kappa_values.max()))
    log_plot = axes["log"].contourf(
        X,
        Y,
        log_kappa_values,
        levels=np.linspace(-log_limit, log_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    axes["log"].scatter(X, Y, s=4, color="black", alpha=0.3)
    spatial_axis(axes["log"], r"Gaussian field $Z=\log(\kappa)$")
    figure.colorbar(log_plot, ax=axes["log"], label=r"$Z(x,\omega)$")

    kappa_levels = np.linspace(kappa_values.min(), kappa_values.max(), 31)
    for key, level, title, line_width in (
        ("coarse", hierarchy.coarsest, r"$\kappa$ on the coarse mesh", 1.3),
        ("fine", hierarchy.finest, r"$\kappa$ on the fine mesh", 0.7),
    ):
        field_plot = axes[key].contourf(
            X,
            Y,
            kappa_values,
            levels=kappa_levels,
            cmap="viridis",
            extend="both",
        )
        overlay_mesh(axes[key], level.mesh, linewidth=line_width)
        spatial_axis(axes[key], title)
        figure.colorbar(field_plot, ax=axes[key], label=r"$\kappa(x,\omega)$")

    operation_numbers = np.arange(len(stage_residuals))
    axes["residual"].semilogy(
        operation_numbers,
        stage_residuals,
        ".-",
        color="tab:purple",
        label="residual after each manual operation",
    )
    axes["residual"].scatter(
        cycle_end_indices,
        np.asarray(stage_residuals)[cycle_end_indices],
        color="tab:red",
        zorder=3,
        label="completed two-level cycle",
    )
    axes["residual"].set_xticks(
        [0, *cycle_end_indices],
        ["initial", *[f"cycle {number}" for number in range(1, len(cycle_end_indices) + 1)]],
    )
    axes["residual"].set_xlabel("manual algorithm progress")
    axes["residual"].set_ylabel(r"fine residual $\|b_f-A_fx_f\|_2$")
    axes["residual"].set_title("Residual after smoothing and coarse-correction steps")
    axes["residual"].grid(True, which="both", alpha=0.3)
    axes["residual"].legend()

    solution_X, solution_Y, _ = cartesian_grid_2d(121, 121)
    solution_values = evaluate_on_grid(
        hierarchy.finest.gfu,
        hierarchy.finest.mesh,
        solution_X,
        solution_Y,
    )
    solution_plot = axes["solution"].contourf(
        solution_X,
        solution_Y,
        solution_values,
        levels=31,
        cmap="plasma",
    )
    overlay_mesh(axes["solution"], hierarchy.finest.mesh, linewidth=0.5)
    spatial_axis(axes["solution"], "Fine-grid solution after manual cycles")
    figure.colorbar(solution_plot, ax=axes["solution"], label=r"$u_h(x,\omega)$")

    figure.suptitle("Manual two-level KL diffusion example", fontsize=16)
    return figure


def main(*, show_plot: bool = True, plot_dir: Path = DEFAULT_PLOT_DIR) -> None:
    # ------------------------------------------------------------------
    # A. Build one discrete KL realization.
    # ------------------------------------------------------------------
    nx = ny = 16
    X, Y, points = cartesian_grid_2d(nx, ny)
    covariance = exponential_covariance(
        points,
        sigma=1.0,
        correlation_length=0.30,
    )
    eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=100)
    log_kappa_values, _ = sample_discrete_kl(
        mean=0.0,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        shape=(ny, nx),
        rng=7,
    )
    kappa_values = lognormal_transform(log_kappa_values)
    kappa = voxel_coefficient_2d(kappa_values, linear=True)

    # ------------------------------------------------------------------
    # B. Assemble the same variable-coefficient PDE on two levels.
    # ------------------------------------------------------------------
    def diffusion_form(a, u, v):
        a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx

    def load_form(f, u, v):
        f += v * ng.dx

    form_setup = build_form_setup(bilinear=diffusion_form, linear=load_form)
    initial_mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.35))
    hierarchy = build_hierarchy(
        initial_mesh,
        form_setup,
        n_refines=1,
        order=1,
        dirichlet="left|right|top|bottom",
        dirichlet_value=0.0,
        verbose=True,
    )

    coarse = hierarchy.coarsest
    fine = hierarchy.finest

    # ------------------------------------------------------------------
    # C. Extract the matrices, vectors, and transfer operators.
    # ------------------------------------------------------------------
    b_fine = fine.f.vec
    x_fine = fine.gfu.vec
    fine.enforce_dirichlet(x_fine)

    # S_f approximately applies A_f^{-1} to a residual during smoothing.
    fine_smoother = fine.a.mat.CreateSmoother(fine.fes.FreeDofs(), GS=True)

    # The coarse system is small, so solve it exactly in every cycle.
    coarse_inverse = coarse.a.mat.Inverse(coarse.fes.FreeDofs())

    print("\nmanual objects")
    print(f"A_f: {fine.a.mat.height} x {fine.a.mat.width}")
    print(f"A_c: {coarse.a.mat.height} x {coarse.a.mat.width}")
    print(f"P:   {fine.P.height} x {fine.P.width}")
    print(f"P^T: {fine.PT.height} x {fine.PT.width}")

    # ------------------------------------------------------------------
    # D. Repeatedly execute the seven explicit two-level steps above.
    # ------------------------------------------------------------------
    max_cycles = 12
    relative_tolerance = 1.0e-8
    initial_residual = free_l2_norm(fine, residual(fine, b_fine, x_fine))

    stage_names = ["initial"]
    stage_residuals = [initial_residual]
    cycle_end_indices: list[int] = []

    print(f"initial fine residual: ||r_f||_2 = {initial_residual:.6e}")
    for cycle_number in range(1, max_cycles + 1):
        names, values = manual_two_level_cycle(
            coarse,
            fine,
            b_fine,
            x_fine,
            fine_smoother,
            coarse_inverse,
            cycle_number=cycle_number,
            pre_sweeps=2,
            post_sweeps=2,
        )
        stage_names.extend(names)
        stage_residuals.extend(values)
        cycle_end_indices.append(len(stage_residuals) - 1)

        if stage_residuals[-1] <= relative_tolerance * initial_residual:
            break

    # ------------------------------------------------------------------
    # E. Plot what the manual operations did.
    # ------------------------------------------------------------------
    figure = make_plot(
        X,
        Y,
        log_kappa_values,
        kappa_values,
        hierarchy,
        stage_residuals,
        cycle_end_indices,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "kl_two_level_manual.png"
    figure.savefig(plot_path, dpi=180)

    print(f"\ncompleted cycles: {len(cycle_end_indices)}")
    print(f"final residual: {stage_residuals[-1]:.6e}")
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
