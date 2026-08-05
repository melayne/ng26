"""Compare multilevel Monte Carlo with standard MC on the finest mesh.

For levels 0 through L, the MLMC estimator is based on

    E[Q_L] = E[Q_0] + sum_{ell=1}^L E[Q_ell - Q_{ell-1}].

Every correction sample evaluates its adjacent levels using the same KL
coefficient vector ``xi``.  All MLMC terms use independent sample streams.

The requested global sampling tolerance is divided equally among the MLMC
terms.  With ``number_of_levels = L + 1``, every term must satisfy

    stderr(term) <= tolerance / number_of_levels.

This is conservative because the standard error of their sum is at most
``tolerance / sqrt(number_of_levels)``.  Standard Monte Carlo on the finest
mesh has one term, so its criterion is ``stderr(Q_L) <= tolerance``.

Run from the project root, for example:

    .venv/bin/python \
        examples/KLMC/multilevel/two_level_mlmc_vs_mc.py \
        --tolerance 0.01 \
        --level-maxhs 0.30 0.15 0.075
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ngsolve as ng
import numpy as np


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
    make_2d_kl_evaluator,
)
from examples.KLMC.utils.one_level_utils import (  # noqa: E402
    build_fixed_mesh,
    quantity_of_interest,
    solve_diffusion,
)


@dataclass
class RunningStatistics:
    """Online sample mean and unbiased sample variance using Welford's method."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        difference = value - self.mean
        self.mean += difference / self.count
        difference_from_new_mean = value - self.mean
        self.sum_squared_deviations += difference * difference_from_new_mean

    @property
    def variance(self) -> float:
        if self.count < 2:
            return math.inf
        return self.sum_squared_deviations / (self.count - 1)

    @property
    def standard_error(self) -> float:
        if self.count < 2:
            return math.inf
        return math.sqrt(self.variance / self.count)


@dataclass(frozen=True)
class SamplingResult:
    """Summary of one adaptively sampled Monte Carlo estimator term."""

    name: str
    count: int
    mean: float
    variance: float
    standard_error: float
    target_standard_error: float
    converged: bool
    wall_time_seconds: float


@dataclass(frozen=True)
class ComparisonResult:
    """Results from MLMC and finest-level standard Monte Carlo."""

    level_terms: tuple[SamplingResult, ...]
    fine_mc: SamplingResult

    @property
    def mlmc_estimate(self) -> float:
        return sum(term.mean for term in self.level_terms)

    @property
    def mlmc_standard_error(self) -> float:
        return math.sqrt(
            sum(
                term.standard_error**2
                for term in self.level_terms
            )
        )

    @property
    def fine_mc_estimate(self) -> float:
        return self.fine_mc.mean


def sample_until_tolerance(
    *,
    name: str,
    draw_sample: Callable[[], float],
    target_standard_error: float,
    minimum_samples: int,
    maximum_samples: int,
    batch_size: int,
    verbose: bool,
) -> SamplingResult:
    """Draw samples until the estimated standard error reaches its target."""
    if target_standard_error <= 0.0:
        raise ValueError("target_standard_error must be positive.")
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be at least 2.")
    if maximum_samples < minimum_samples:
        raise ValueError("maximum_samples must be at least minimum_samples.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    statistics = RunningStatistics()
    start = time.perf_counter()

    while statistics.count < maximum_samples:
        remaining = maximum_samples - statistics.count
        current_batch_size = min(batch_size, remaining)

        for _ in range(current_batch_size):
            statistics.update(float(draw_sample()))

        if verbose:
            print(
                f"  {name:<22} N={statistics.count:5d}  "
                f"mean={statistics.mean: .6e}  "
                f"stderr={statistics.standard_error:.3e}",
                flush=True,
            )

        enough_samples = statistics.count >= minimum_samples
        tolerance_met = statistics.standard_error <= target_standard_error
        if enough_samples and tolerance_met:
            break

    converged = (
        statistics.count >= minimum_samples
        and statistics.standard_error <= target_standard_error
    )

    return SamplingResult(
        name=name,
        count=statistics.count,
        mean=statistics.mean,
        variance=statistics.variance,
        standard_error=statistics.standard_error,
        target_standard_error=target_standard_error,
        converged=converged,
        wall_time_seconds=time.perf_counter() - start,
    )


def solve_qoi(
    mesh: ng.Mesh,
    fes: ng.H1,
    kappa,
) -> float:
    """Solve the Darcy problem and return its right-boundary outflow flux."""
    solution = solve_diffusion(fes, kappa)
    return quantity_of_interest(solution, mesh, kappa)


def run_comparison(
    *,
    tolerance: float = 1.0e-2,
    level_maxhs: tuple[float, ...] = (0.30, 0.15),
    coefficient_grid_size: int = 32,
    num_modes_2d: int = 100,
    correlation_length: float = 0.30,
    standard_deviation: float = 1.0,
    mean_log_conductivity: float = 0.0,
    seed: int = 7,
    minimum_samples: int = 20,
    maximum_samples: int = 30_000,
    batch_size: int = 10,
    verbose: bool = True,
) -> ComparisonResult:
    """Run MLMC and independent standard MC on the finest mesh."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if len(level_maxhs) < 2:
        raise ValueError("level_maxhs must contain at least two mesh sizes.")
    if any(maxh <= 0.0 for maxh in level_maxhs):
        raise ValueError("Every entry in level_maxhs must be positive.")
    if any(
        fine_maxh >= coarse_maxh
        for coarse_maxh, fine_maxh in zip(level_maxhs, level_maxhs[1:])
    ):
        raise ValueError(
            "level_maxhs must be strictly decreasing from coarse to fine."
        )
    if coefficient_grid_size < 2:
        raise ValueError("coefficient_grid_size must be at least 2.")
    if num_modes_2d < 1:
        raise ValueError("num_modes_2d must be positive.")
    if correlation_length <= 0.0:
        raise ValueError("correlation_length must be positive.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")

    # ---------------------------------------------------------------
    # Fixed FE meshes and spaces.  Only the coefficient and assembled
    # linear systems change between Monte Carlo realizations.
    # ---------------------------------------------------------------
    levels = tuple(
        build_fixed_mesh(maxh=maxh)
        for maxh in level_maxhs
    )
    number_of_levels = len(levels)

    # ---------------------------------------------------------------
    # Analytical KL basis for the paper's separable L1 covariance.
    # Computing num_modes_2d one-dimensional modes is conservative and
    # guarantees enough candidates for num_modes_2d tensor products.
    # ---------------------------------------------------------------
    (
        frequencies_1d,
        normalizations_1d,
        eigenvalues_1d,
        _,
    ) = get_1d_eigenpairs(
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

    variance = standard_deviation**2
    evaluate_log_conductivity = make_2d_kl_evaluator(
        eigenvalues_2d=unit_eigenvalues_2d,
        eigenfunction_evaluator=evaluate_eigenfunctions_2d,
        mean_log_conductivity=mean_log_conductivity,
        variance=variance,
    )

    X, Y, _ = cartesian_grid_2d(
        coefficient_grid_size,
        coefficient_grid_size,
    )

    def coefficient_from_xi(xi: np.ndarray):
        log_kappa_values = evaluate_log_conductivity(X, Y, xi)
        kappa_values = lognormal_transform(log_kappa_values)
        return voxel_coefficient_2d(kappa_values, linear=True)

    # Separate random streams keep all MLMC terms independent.  Within one
    # correction sample, one xi and one kappa are shared by both adjacent
    # spatial levels.  The final stream is reserved for standard fine MC.
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(number_of_levels + 1)
    term_rngs = tuple(
        np.random.default_rng(child_seed)
        for child_seed in child_seeds[:-1]
    )
    fine_mc_rng = np.random.default_rng(child_seeds[-1])

    def draw_level_zero() -> float:
        xi = term_rngs[0].standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)
        mesh, fes = levels[0]
        return solve_qoi(mesh, fes, kappa)

    def make_correction_sampler(level_index: int) -> Callable[[], float]:
        lower_mesh, lower_fes = levels[level_index - 1]
        upper_mesh, upper_fes = levels[level_index]
        rng = term_rngs[level_index]

        def draw_correction() -> float:
            xi = rng.standard_normal(num_modes_2d)
            kappa = coefficient_from_xi(xi)
            q_lower = solve_qoi(lower_mesh, lower_fes, kappa)
            q_upper = solve_qoi(upper_mesh, upper_fes, kappa)
            return q_upper - q_lower

        return draw_correction

    def draw_fine_mc() -> float:
        xi = fine_mc_rng.standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)
        finest_mesh, finest_fes = levels[-1]
        return solve_qoi(finest_mesh, finest_fes, kappa)

    term_tolerance = tolerance / number_of_levels

    if verbose:
        retained_variance = float(np.sum(unit_eigenvalues_2d))
        largest_indices = np.max(mode_indices, axis=0)
        print(f"{number_of_levels}-level MLMC versus finest-level MC")
        for level_index, ((_, fes), maxh) in enumerate(zip(levels, level_maxhs)):
            print(
                f"  level {level_index}: maxh={maxh}, ndof={fes.ndof}"
            )
        print(
            f"  KL modes: {num_modes_2d}, ell={correlation_length}, "
            f"sigma={standard_deviation}"
        )
        print(
            "  largest retained 1D indices: "
            f"({largest_indices[0]}, {largest_indices[1]})"
        )
        print(f"  retained integrated variance: {retained_variance:.2%}")
        print(f"  requested tolerance: {tolerance:.3e}")
        print(f"  MLMC tolerance per term: {term_tolerance:.3e}")
        print()

    level_results: list[SamplingResult] = []
    level_results.append(
        sample_until_tolerance(
            name="level 0: Q_0",
            draw_sample=draw_level_zero,
            target_standard_error=term_tolerance,
            minimum_samples=minimum_samples,
            maximum_samples=maximum_samples,
            batch_size=batch_size,
            verbose=verbose,
        )
    )

    for level_index in range(1, number_of_levels):
        level_results.append(
            sample_until_tolerance(
                name=f"level {level_index}: Q_{level_index}-Q_{level_index - 1}",
                draw_sample=make_correction_sampler(level_index),
                target_standard_error=term_tolerance,
                minimum_samples=minimum_samples,
                maximum_samples=maximum_samples,
                batch_size=batch_size,
                verbose=verbose,
            )
        )

    fine_mc_result = sample_until_tolerance(
        name=f"fine MC Q_{number_of_levels - 1}",
        draw_sample=draw_fine_mc,
        target_standard_error=tolerance,
        minimum_samples=minimum_samples,
        maximum_samples=maximum_samples,
        batch_size=batch_size,
        verbose=verbose,
    )

    result = ComparisonResult(
        level_terms=tuple(level_results),
        fine_mc=fine_mc_result,
    )
    print_summary(result)
    return result


def print_summary(result: ComparisonResult) -> None:
    """Print estimator accuracy, work counts, and agreement."""
    print()
    print("Sampling summary")
    print(
        f"{'term':<28} {'N':>7} {'mean':>14} {'variance':>14} "
        f"{'stderr':>12} {'target':>12} {'time[s]':>10} {'met':>5}"
    )
    print("-" * 110)

    for term in (*result.level_terms, result.fine_mc):
        print(
            f"{term.name:<28} {term.count:7d} {term.mean:14.6e} "
            f"{term.variance:14.6e} {term.standard_error:12.3e} "
            f"{term.target_standard_error:12.3e} "
            f"{term.wall_time_seconds:10.2f} {str(term.converged):>5}"
        )

    solves_per_level = [0] * len(result.level_terms)
    solves_per_level[0] += result.level_terms[0].count
    for level_index, correction in enumerate(result.level_terms[1:], start=1):
        solves_per_level[level_index - 1] += correction.count
        solves_per_level[level_index] += correction.count

    print()
    print(
        f"MLMC:           {result.mlmc_estimate:.8e} "
        f"+/- {result.mlmc_standard_error:.3e} (one standard error)"
    )
    print(
        f"finest-grid MC: {result.fine_mc_estimate:.8e} "
        f"+/- {result.fine_mc.standard_error:.3e} (one standard error)"
    )
    print(
        "estimate difference (MLMC - fine MC): "
        f"{result.mlmc_estimate - result.fine_mc_estimate:.3e}"
    )
    print("MLMC PDE solves by spatial level:")
    for level_index, count in enumerate(solves_per_level):
        print(f"  level {level_index}: {count}")
    print(
        "finest-grid MC PDE solves on level "
        f"{len(result.level_terms) - 1}: {result.fine_mc.count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1.0e-2)
    parser.add_argument(
        "--level-maxhs",
        nargs="+",
        type=float,
        default=[0.30, 0.15],
        help=(
            "strictly decreasing mesh sizes from coarse to fine, "
            "for example: --level-maxhs 0.30 0.15 0.075"
        ),
    )
    parser.add_argument("--coefficient-grid-size", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=100, dest="num_modes_2d")
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--sigma", type=float, default=1.0, dest="standard_deviation")
    parser.add_argument("--mean-log-kappa", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--maximum-samples", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_comparison(
        tolerance=arguments.tolerance,
        level_maxhs=tuple(arguments.level_maxhs),
        coefficient_grid_size=arguments.coefficient_grid_size,
        num_modes_2d=arguments.num_modes_2d,
        correlation_length=arguments.correlation_length,
        standard_deviation=arguments.standard_deviation,
        mean_log_conductivity=arguments.mean_log_kappa,
        seed=arguments.seed,
        minimum_samples=arguments.minimum_samples,
        maximum_samples=arguments.maximum_samples,
        batch_size=arguments.batch_size,
        verbose=not arguments.quiet,
    )
