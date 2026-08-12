"""Compare the two-level correction with independent fine-level Monte Carlo.

This diagnostic intentionally omits the coarse expectation ``E[Q_0]``.  It
samples only

    Y_1 = Q_1 - Q_0

with the same KL vector ``xi`` on both meshes, followed by independent
standard Monte Carlo samples of ``Q_1``.

Because ``E[Q_0]`` is omitted, the correction mean by itself is not an MLMC
estimate of ``E[Q_1]``.  This script is intended to compare sample counts,
variances, and costs of the correction and fine-MC estimators.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import (  # noqa: E402
    cartesian_grid_2d,
    voxel_coefficient_2d,
)
from examples.KLMC.multilevel.two_level_mlmc_vs_mc import (  # noqa: E402
    SamplingResult,
    sample_until_tolerance,
    solve_qoi,
)
from examples.KLMC.utils.analytical_eigenfunctions import (  # noqa: E402
    get_1d_eigenpairs,
    leading_2d_eigenpairs,
)
from examples.KLMC.utils.one_level_utils import (  # noqa: E402
    build_fixed_mesh,
)


@dataclass(frozen=True)
class DifferenceComparisonResult:
    """Results for the correction and independent fine-MC estimators."""

    correction: SamplingResult
    fine_mc: SamplingResult


def run_comparison(
    *,
    tolerance: float = 1.0e-2,
    coarse_maxh: float = 0.30,
    fine_maxh: float = 0.15,
    coefficient_grid_size: int = 32,
    num_modes_2d: int = 100,
    correlation_length: float = 0.30,
    standard_deviation: float = 1.0,
    mean_log_conductivity: float = 0.0,
    seed: int = 7,
    minimum_samples: int = 20,
    maximum_samples: int = 30_000,
    batch_size: int = 100,
    same_tolerance: bool = False,
    verbose: bool = True,
) -> DifferenceComparisonResult:
    """Sample ``Q_1-Q_0`` and independent fine-grid ``Q_1``."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if not 0.0 < fine_maxh < coarse_maxh:
        raise ValueError("fine_maxh must be positive and smaller than coarse_maxh.")
    if coefficient_grid_size < 2:
        raise ValueError("coefficient_grid_size must be at least 2.")
    if num_modes_2d < 1:
        raise ValueError("num_modes_2d must be positive.")
    if correlation_length <= 0.0:
        raise ValueError("correlation_length must be positive.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")

    coarse_mesh, coarse_fes = build_fixed_mesh(maxh=coarse_maxh)
    fine_mesh, fine_fes = build_fixed_mesh(maxh=fine_maxh)

    frequencies_1d, normalizations_1d, eigenvalues_1d, _ = get_1d_eigenpairs(
        num_modes=num_modes_2d,
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

    X, Y, _ = cartesian_grid_2d(
        coefficient_grid_size,
        coefficient_grid_size,
    )

    # The eigenfunctions are deterministic, so evaluate them once rather
    # than recalculating every sine and cosine for every Monte Carlo sample.
    eigenfunction_values = evaluate_eigenfunctions_2d(X, Y)
    weighted_eigenfunctions = (
        eigenfunction_values
        * np.sqrt(
            standard_deviation**2
            * unit_eigenvalues_2d
        )
    )

    def coefficient_from_xi(xi: np.ndarray):
        log_kappa = (
            mean_log_conductivity
            + np.tensordot(
                weighted_eigenfunctions,
                xi,
                axes=([-1], [0]),
            )
        )
        return voxel_coefficient_2d(
            np.exp(log_kappa),
            linear=True,
        )

    seed_sequence = np.random.SeedSequence(seed)
    correction_seed, fine_seed = seed_sequence.spawn(2)
    correction_rng = np.random.default_rng(correction_seed)
    fine_rng = np.random.default_rng(fine_seed)

    def draw_correction() -> float:
        xi = correction_rng.standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)

        # The same xi and kappa are used on both spatial levels.
        q_coarse = solve_qoi(coarse_mesh, coarse_fes, kappa)
        q_fine = solve_qoi(fine_mesh, fine_fes, kappa)
        return q_fine - q_coarse

    def draw_fine_mc() -> float:
        xi = fine_rng.standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)
        return solve_qoi(fine_mesh, fine_fes, kappa)

    # By default, reproduce the original two-level MLMC split: the
    # correction target is epsilon/2 while fine MC receives epsilon.
    # --same-tolerance gives both estimators epsilon for a direct comparison.
    correction_tolerance = (
        tolerance
        if same_tolerance
        else tolerance / 2.0
    )

    if verbose:
        largest_indices = np.max(mode_indices, axis=0)
        print("Correction versus independent fine-grid MC")
        print(f"  coarse: maxh={coarse_maxh}, ndof={coarse_fes.ndof}")
        print(f"  fine:   maxh={fine_maxh}, ndof={fine_fes.ndof}")
        print(
            f"  KL modes={num_modes_2d}, grid={coefficient_grid_size}x"
            f"{coefficient_grid_size}, retained variance="
            f"{np.sum(unit_eigenvalues_2d):.2%}"
        )
        print(
            "  largest retained 1D indices: "
            f"({largest_indices[0]}, {largest_indices[1]})"
        )
        print(f"  correction stderr target={correction_tolerance:.3e}")
        print(f"  fine MC stderr target={tolerance:.3e}")
        print()

    correction = sample_until_tolerance(
        name="correction Q_1-Q_0",
        draw_sample=draw_correction,
        target_standard_error=correction_tolerance,
        minimum_samples=minimum_samples,
        maximum_samples=maximum_samples,
        batch_size=batch_size,
        verbose=verbose,
    )
    fine_mc = sample_until_tolerance(
        name="fine MC Q_1",
        draw_sample=draw_fine_mc,
        target_standard_error=tolerance,
        minimum_samples=minimum_samples,
        maximum_samples=maximum_samples,
        batch_size=batch_size,
        verbose=verbose,
    )

    result = DifferenceComparisonResult(
        correction=correction,
        fine_mc=fine_mc,
    )
    print_summary(result)
    return result


def print_summary(result: DifferenceComparisonResult) -> None:
    """Print the correction and fine-MC statistics and work counts."""
    print()
    print("Sampling summary")
    print(
        f"{'term':<24} {'N':>7} {'mean':>14} {'variance':>14} "
        f"{'stderr':>12} {'target':>12} {'time[s]':>10} {'met':>5}"
    )
    print("-" * 106)
    for term in (result.correction, result.fine_mc):
        print(
            f"{term.name:<24} {term.count:7d} {term.mean:14.6e} "
            f"{term.variance:14.6e} {term.standard_error:12.3e} "
            f"{term.target_standard_error:12.3e} "
            f"{term.wall_time_seconds:10.2f} {str(term.converged):>5}"
        )

    print()
    print(
        "Correction PDE solves: "
        f"{result.correction.count} coarse + "
        f"{result.correction.count} fine"
    )
    print(f"Fine-MC PDE solves: {result.fine_mc.count} fine")
    print(
        "The correction mean is E[Q_1-Q_0], not an estimate of E[Q_1] "
        "without adding E[Q_0]."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1.0e-2)
    parser.add_argument("--coarse-maxh", type=float, default=0.30)
    parser.add_argument("--fine-maxh", type=float, default=0.15)
    parser.add_argument("--coefficient-grid-size", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=200, dest="num_modes_2d")
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--sigma", type=float, default=1.0, dest="standard_deviation")
    parser.add_argument("--mean-log-kappa", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--maximum-samples", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--same-tolerance",
        action="store_true",
        help="use epsilon for both estimators instead of epsilon/2 for the correction",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_comparison(
        tolerance=arguments.tolerance,
        coarse_maxh=arguments.coarse_maxh,
        fine_maxh=arguments.fine_maxh,
        coefficient_grid_size=arguments.coefficient_grid_size,
        num_modes_2d=arguments.num_modes_2d,
        correlation_length=arguments.correlation_length,
        standard_deviation=arguments.standard_deviation,
        mean_log_conductivity=arguments.mean_log_kappa,
        seed=arguments.seed,
        minimum_samples=arguments.minimum_samples,
        maximum_samples=arguments.maximum_samples,
        batch_size=arguments.batch_size,
        same_tolerance=arguments.same_tolerance,
        verbose=not arguments.quiet,
    )
