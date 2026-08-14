"""Visualize positivity before and after finite-element projection.

This example isolates the conductivity representation from the PDE and the
Monte Carlo calculation. It constructs a strictly positive lognormal voxel
field and compares:

1. the original ``VoxelCoefficient``;
2. ``GridFunction.Set(voxel)`` in a continuous ``H1(order=1)`` space;
3. ``GridFunction.Set(voxel)`` in a discontinuous ``L2(order=0)`` space.

Run from the project root with

    .venv/bin/python \
        examples/KLMC/multilevel/plot_voxel_projection_positivity.py

Add ``--show`` to open the Matplotlib window as well as saving the figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import voxel_coefficient_2d  # noqa: E402


def evaluate_coefficient(
    coefficient: ng.CoefficientFunction,
    mesh: ng.Mesh,
    *,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one coefficient on a regular grid inside the unit square."""
    # Avoid evaluating exactly on the outer boundary, where a point can be
    # geometrically ambiguous to the mesh-point locator.
    epsilon = 1.0e-10
    x = np.linspace(epsilon, 1.0 - epsilon, resolution)
    y = np.linspace(epsilon, 1.0 - epsilon, resolution)
    values = np.empty((resolution, resolution))

    for row, y_value in enumerate(y):
        for column, x_value in enumerate(x):
            values[row, column] = coefficient(
                mesh(float(x_value), float(y_value))
            )

    return x, y, values


def make_positive_voxel_values(
    *,
    grid_size: int,
    seed: int,
    log_standard_deviation: float,
) -> np.ndarray:
    """Create positive values with enough contrast to reveal undershoot."""
    rng = np.random.default_rng(seed)
    log_values = rng.normal(
        loc=0.0,
        scale=log_standard_deviation,
        size=(grid_size, grid_size),
    )
    return np.exp(log_values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a positive voxel coefficient with its H1 and L2 "
            "finite-element representations."
        )
    )
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-standard-deviation",
        type=float,
        default=2.0,
    )
    parser.add_argument("--plot-resolution", type=int, default=151)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_PATH.with_name(
            "voxel_projection_positivity.png"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the Matplotlib window after saving the plot",
    )
    arguments = parser.parse_args()

    if arguments.grid_size < 4:
        raise ValueError("grid-size must be at least 4.")
    if arguments.maxh <= 0.0:
        raise ValueError("maxh must be positive.")
    if arguments.log_standard_deviation <= 0.0:
        raise ValueError(
            "log-standard-deviation must be positive."
        )
    if arguments.plot_resolution < 2:
        raise ValueError("plot-resolution must be at least 2.")

    voxel_values = make_positive_voxel_values(
        grid_size=arguments.grid_size,
        seed=arguments.seed,
        log_standard_deviation=(
            arguments.log_standard_deviation
        ),
    )
    voxel = voxel_coefficient_2d(
        voxel_values,
        linear=True,
    )

    mesh = ng.Mesh(
        unit_square.GenerateMesh(
            maxh=arguments.maxh
        )
    )

    h1_field = ng.GridFunction(
        ng.H1(mesh, order=1)
    )
    h1_field.Set(voxel)

    l2_field = ng.GridFunction(
        ng.L2(mesh, order=0)
    )
    l2_field.Set(voxel)

    x, y, voxel_plot_values = evaluate_coefficient(
        voxel,
        mesh,
        resolution=arguments.plot_resolution,
    )
    _, _, h1_plot_values = evaluate_coefficient(
        h1_field,
        mesh,
        resolution=arguments.plot_resolution,
    )
    _, _, l2_plot_values = evaluate_coefficient(
        l2_field,
        mesh,
        resolution=arguments.plot_resolution,
    )

    h1_dofs = h1_field.vec.FV().NumPy()
    l2_dofs = l2_field.vec.FV().NumPy()

    print("Strictly positive input voxel data")
    print(
        f"  stored grid values: min={voxel_values.min():.6e}, "
        f"max={voxel_values.max():.6e}"
    )
    print(
        f"  evaluated voxel:    min={voxel_plot_values.min():.6e}, "
        f"max={voxel_plot_values.max():.6e}"
    )
    print()
    print("Finite-element representations created by GridFunction.Set")
    print(
        f"  H1 coefficient DOFs: min={h1_dofs.min():.6e}, "
        f"max={h1_dofs.max():.6e}"
    )
    print(
        f"  evaluated H1 field:  min={h1_plot_values.min():.6e}, "
        f"max={h1_plot_values.max():.6e}"
    )
    print(
        f"  L2 coefficient DOFs: min={l2_dofs.min():.6e}, "
        f"max={l2_dofs.max():.6e}"
    )
    print(
        f"  evaluated L2 field:  min={l2_plot_values.min():.6e}, "
        f"max={l2_plot_values.max():.6e}"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.4),
        constrained_layout=True,
    )

    panels = (
        (
            "Original VoxelCoefficient",
            voxel_plot_values,
        ),
        (
            "H1(order=1) after Set",
            h1_plot_values,
        ),
        (
            "L2(order=0) after Set",
            l2_plot_values,
        ),
    )

    for axis, (title, values) in zip(
        axes,
        panels,
        strict=True,
    ):
        minimum = float(values.min())
        maximum = float(values.max())

        if minimum < 0.0 < maximum:
            color_norm = TwoSlopeNorm(
                vmin=minimum,
                vcenter=0.0,
                vmax=maximum,
            )
            color_map = "coolwarm"
        elif minimum > 0.0:
            color_norm = LogNorm(
                vmin=minimum,
                vmax=maximum,
            )
            color_map = "viridis"
        else:
            color_norm = Normalize(
                vmin=minimum,
                vmax=maximum,
            )
            color_map = "viridis"

        image = axis.imshow(
            values,
            origin="lower",
            extent=(x[0], x[-1], y[0], y[-1]),
            cmap=color_map,
            norm=color_norm,
            interpolation="nearest",
        )
        axis.set_title(
            f"{title}\n"
            f"min={minimum:.3e}, max={maximum:.3e}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

        if values.min() < 0.0 < values.max():
            axis.contour(
                x,
                y,
                values,
                levels=[0.0],
                colors="black",
                linewidths=1.0,
            )
        figure.colorbar(
            image,
            ax=axis,
            label="conductivity",
            shrink=0.86,
        )
    figure.suptitle(
        "A positive voxel field and two finite-element representations"
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        arguments.output,
        dpi=180,
    )
    print()
    print(f"Saved plot to: {arguments.output}")

    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
