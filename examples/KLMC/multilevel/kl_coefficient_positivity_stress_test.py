"""Stress-test positivity of the KL conductivity representations.

The test draws samples from the same analytical two-dimensional KL expansion
used by ``multilevel_test.py``. For every sample it records the minimum and
maximum of:

1. the grid values returned by ``exp(log_kappa)``;
2. the corresponding ``VoxelCoefficient`` on a fixed probe grid;
3. ``GridFunction.Set(voxel)`` in ``H1(order=1)``;
4. ``GridFunction.Set(voxel)`` in ``L2(order=0)``.

No PDE is assembled or solved.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
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
)


RESULT_COLUMNS = (
    "transformed_min",
    "transformed_max",
    "voxel_min",
    "voxel_max",
    "h1_min",
    "h1_max",
    "l2_min",
    "l2_max",
)

REPRESENTATIONS = (
    ("Transformed grid", "transformed_min", "transformed_max"),
    ("VoxelCoefficient", "voxel_min", "voxel_max"),
    ("H1(order=1)", "h1_min", "h1_max"),
    ("L2(order=0)", "l2_min", "l2_max"),
)


def build_scaled_kl_modes(
    *,
    grid_size: int,
    number_of_modes: int,
    correlation_length: float,
    variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X``, ``Y``, and flattened variance-scaled KL modes."""
    X, Y, _ = cartesian_grid_2d(
        grid_size,
        grid_size,
    )

    (
        frequencies_1d,
        normalizations_1d,
        eigenvalues_1d,
        _,
    ) = get_1d_eigenpairs(
        num_modes=number_of_modes,
        correlation_length=correlation_length,
    )

    (
        eigenvalues_2d,
        _,
        evaluate_eigenfunctions_2d,
    ) = leading_2d_eigenpairs(
        eigenvalues_1d=eigenvalues_1d,
        frequencies_1d=frequencies_1d,
        normalizations_1d=normalizations_1d,
        correlation_length=correlation_length,
        num_modes_2d=number_of_modes,
        method="heap",
    )

    eigenfunction_values = evaluate_eigenfunctions_2d(
        X,
        Y,
    )
    scaled_modes = (
        eigenfunction_values
        * np.sqrt(variance * eigenvalues_2d)
    )

    return (
        X,
        Y,
        scaled_modes.reshape(-1, number_of_modes),
    )


def run_stress_test(
    *,
    number_of_samples: int,
    seed: int,
    grid_size: int,
    number_of_modes: int,
    correlation_length: float,
    mean_log_conductivity: float,
    variance: float,
    coarse_maxh: float,
    probe_resolution: int,
    batch_size: int,
    progress_interval: int,
) -> np.ndarray:
    """Draw KL samples and return one row of extrema per sample."""
    _, _, scaled_modes = build_scaled_kl_modes(
        grid_size=grid_size,
        number_of_modes=number_of_modes,
        correlation_length=correlation_length,
        variance=variance,
    )

    mesh = ng.Mesh(
        unit_square.GenerateMesh(
            maxh=coarse_maxh
        )
    )

    h1_field = ng.GridFunction(
        ng.H1(mesh, order=1)
    )
    l2_field = ng.GridFunction(
        ng.L2(mesh, order=0)
    )

    # NGSolve accepts arrays of coordinates and evaluates the coefficient at
    # all returned mesh points in one call. The small offset avoids ambiguous
    # point location exactly on the outer boundary.
    epsilon = 1.0e-10
    probe_axis = np.linspace(
        epsilon,
        1.0 - epsilon,
        probe_resolution,
    )
    probe_X, probe_Y = np.meshgrid(
        probe_axis,
        probe_axis,
        indexing="xy",
    )
    probe_points = mesh(
        probe_X.ravel(),
        probe_Y.ravel(),
    )

    rng = np.random.default_rng(seed)
    results = np.full(
        (number_of_samples, len(RESULT_COLUMNS)),
        np.nan,
    )

    started = time.perf_counter()

    for batch_start in range(
        0,
        number_of_samples,
        batch_size,
    ):
        batch_stop = min(
            batch_start + batch_size,
            number_of_samples,
        )
        current_batch_size = batch_stop - batch_start

        xi = rng.standard_normal(
            (current_batch_size, number_of_modes)
        )
        log_kappa_batch = (
            mean_log_conductivity
            + xi @ scaled_modes.T
        )

        for local_index in range(current_batch_size):
            sample_index = batch_start + local_index
            log_kappa = log_kappa_batch[
                local_index
            ].reshape(grid_size, grid_size)
            transformed_values = lognormal_transform(
                log_kappa
            )

            results[sample_index, 0] = np.min(
                transformed_values
            )
            results[sample_index, 1] = np.max(
                transformed_values
            )

            # Do not pass NaN or infinity into NGSolve. The transformed
            # extrema remain recorded so an overflow is still visible.
            if not np.all(np.isfinite(transformed_values)):
                continue

            voxel = voxel_coefficient_2d(
                transformed_values,
                linear=True,
            )
            voxel_probe_values = np.asarray(
                voxel(probe_points),
                dtype=float,
            ).reshape(-1)

            results[sample_index, 2] = np.min(
                voxel_probe_values
            )
            results[sample_index, 3] = np.max(
                voxel_probe_values
            )

            h1_field.Set(voxel)
            h1_dofs = h1_field.vec.FV().NumPy()
            results[sample_index, 4] = np.min(h1_dofs)
            results[sample_index, 5] = np.max(h1_dofs)

            l2_field.Set(voxel)
            l2_dofs = l2_field.vec.FV().NumPy()
            results[sample_index, 6] = np.min(l2_dofs)
            results[sample_index, 7] = np.max(l2_dofs)

        completed = batch_stop
        if (
            completed == number_of_samples
            or completed % progress_interval == 0
        ):
            elapsed = time.perf_counter() - started
            print(
                f"completed {completed:>6}/{number_of_samples} "
                f"samples in {elapsed:.1f}s"
            )

    return results


def write_results_csv(
    output_path: Path,
    results: np.ndarray,
) -> None:
    """Write all per-sample extrema to a CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("sample", *RESULT_COLUMNS))

        for sample_index, row in enumerate(results):
            writer.writerow(
                (sample_index, *row)
            )


def print_summary(results: np.ndarray) -> None:
    """Print global extrema and nonpositive-sample counts."""
    column_indices = {
        name: index
        for index, name in enumerate(RESULT_COLUMNS)
    }

    print()
    print("KL conductivity positivity stress test")
    print(
        f"{'representation':<20} "
        f"{'global min':>13} "
        f"{'global max':>13} "
        f"{'min <= 0':>10} "
        f"{'nonfinite':>10}"
    )
    print("-" * 70)

    for label, minimum_name, maximum_name in REPRESENTATIONS:
        minimum_values = results[
            :,
            column_indices[minimum_name],
        ]
        maximum_values = results[
            :,
            column_indices[maximum_name],
        ]
        finite_rows = (
            np.isfinite(minimum_values)
            & np.isfinite(maximum_values)
        )
        nonfinite_count = int(
            np.count_nonzero(~finite_rows)
        )
        nonpositive_count = int(
            np.count_nonzero(
                finite_rows
                & (minimum_values <= 0.0)
            )
        )

        if np.any(finite_rows):
            global_minimum = np.min(
                minimum_values[finite_rows]
            )
            global_maximum = np.max(
                maximum_values[finite_rows]
            )
        else:
            global_minimum = np.nan
            global_maximum = np.nan

        print(
            f"{label:<20} "
            f"{global_minimum:>13.6e} "
            f"{global_maximum:>13.6e} "
            f"{nonpositive_count:>10} "
            f"{nonfinite_count:>10}"
        )


def plot_distributions(
    output_path: Path,
    results: np.ndarray,
) -> None:
    """Plot distributions of the per-sample minima and maxima."""
    column_indices = {
        name: index
        for index, name in enumerate(RESULT_COLUMNS)
    }

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16.0, 7.4),
        constrained_layout=True,
    )

    for column, (
        label,
        minimum_name,
        maximum_name,
    ) in enumerate(REPRESENTATIONS):
        minimum_values = results[
            :,
            column_indices[minimum_name],
        ]
        maximum_values = results[
            :,
            column_indices[maximum_name],
        ]
        minimum_values = minimum_values[
            np.isfinite(minimum_values)
        ]
        maximum_values = maximum_values[
            np.isfinite(maximum_values)
        ]

        minimum_axis = axes[0, column]
        maximum_axis = axes[1, column]

        minimum_axis.hist(
            minimum_values,
            bins=50,
            color="tab:blue",
            alpha=0.85,
        )
        minimum_axis.axvline(
            0.0,
            color="black",
            linewidth=1.0,
        )
        minimum_axis.set_title(label)
        minimum_axis.set_xlabel("minimum conductivity")
        minimum_axis.set_ylabel("sample count")

        maximum_axis.hist(
            maximum_values,
            bins=50,
            color="tab:orange",
            alpha=0.85,
        )
        maximum_axis.set_xlabel("maximum conductivity")
        maximum_axis.set_ylabel("sample count")

    figure.suptitle(
        "Extrema from each KL conductivity sample\n"
        "top: per-sample minima, bottom: per-sample maxima"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw KL conductivity samples and test positivity before and "
            "after H1 and L2 finite-element representation."
        )
    )
    parser.add_argument("--number-of-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--number-of-modes", type=int, default=100)
    parser.add_argument("--correlation-length", type=float, default=0.3)
    parser.add_argument("--mean-log-conductivity", type=float, default=0.0)
    parser.add_argument("--standard-deviation", type=float, default=1.0)
    parser.add_argument("--coarse-maxh", type=float, default=0.3)
    parser.add_argument("--probe-resolution", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=SCRIPT_PATH.with_name(
            "kl_coefficient_positivity_results.csv"
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=SCRIPT_PATH.with_name(
            "kl_coefficient_positivity_distributions.png"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the saved distribution plot",
    )
    arguments = parser.parse_args()

    if arguments.number_of_samples < 1:
        raise ValueError("number-of-samples must be positive.")
    if arguments.grid_size < 2:
        raise ValueError("grid-size must be at least 2.")
    if arguments.number_of_modes < 1:
        raise ValueError("number-of-modes must be positive.")
    if arguments.correlation_length <= 0.0:
        raise ValueError("correlation-length must be positive.")
    if arguments.standard_deviation < 0.0:
        raise ValueError("standard-deviation must be nonnegative.")
    if arguments.coarse_maxh <= 0.0:
        raise ValueError("coarse-maxh must be positive.")
    if arguments.probe_resolution < 2:
        raise ValueError("probe-resolution must be at least 2.")
    if arguments.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if arguments.progress_interval < 1:
        raise ValueError("progress-interval must be positive.")

    print("KL coefficient positivity stress test")
    print(f"  samples:             {arguments.number_of_samples}")
    print(f"  KL modes:            {arguments.number_of_modes}")
    print(f"  KL grid:             {arguments.grid_size} x {arguments.grid_size}")
    print(f"  correlation length:  {arguments.correlation_length}")
    print(f"  standard deviation:  {arguments.standard_deviation}")
    print(f"  coarse maxh:         {arguments.coarse_maxh}")
    print(f"  voxel probe grid:    {arguments.probe_resolution} x "
          f"{arguments.probe_resolution}")
    print()

    results = run_stress_test(
        number_of_samples=arguments.number_of_samples,
        seed=arguments.seed,
        grid_size=arguments.grid_size,
        number_of_modes=arguments.number_of_modes,
        correlation_length=arguments.correlation_length,
        mean_log_conductivity=arguments.mean_log_conductivity,
        variance=arguments.standard_deviation**2,
        coarse_maxh=arguments.coarse_maxh,
        probe_resolution=arguments.probe_resolution,
        batch_size=arguments.batch_size,
        progress_interval=arguments.progress_interval,
    )

    print_summary(results)
    write_results_csv(
        arguments.csv_output,
        results,
    )
    plot_distributions(
        arguments.plot_output,
        results,
    )

    print()
    print(f"CSV results: {arguments.csv_output}")
    print(f"Distribution plot: {arguments.plot_output}")

    if arguments.show:
        image = plt.imread(arguments.plot_output)
        figure, axis = plt.subplots(figsize=(14.0, 7.0))
        axis.imshow(image)
        axis.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
