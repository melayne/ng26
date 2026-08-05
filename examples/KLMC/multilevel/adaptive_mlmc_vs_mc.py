"""Paper-style adaptive MLMC compared with MC on the finest active level.

The MLMC terms are

    Y_0 = Q_0,
    Y_l = Q_l - Q_{l-1},  l >= 1.

For every correction sample, both spatial levels use the same KL coefficient
vector ``xi``.  Pilot samples estimate each term variance ``V_l`` and average
cost ``C_l``.  Sample targets are then allocated according to

    N_l = ceil(
        2 / tolerance**2
        * sqrt(V_l / C_l)
        * sum_j sqrt(V_j * C_j)
    ).

This targets a total MLMC sampling variance no larger than
``tolerance**2 / 2``.  The other half of the squared-error budget is reserved
for spatial bias.  If the finest correction does not pass the bias test, the
next mesh supplied through ``--level-maxhs`` is activated.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
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
    """Online mean and unbiased sample variance."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        difference = value - self.mean
        self.mean += difference / self.count
        self.sum_squared_deviations += difference * (value - self.mean)

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


@dataclass
class TermState:
    """Samples, cost measurements, and sampler for one MLMC term."""

    name: str
    draw_sample: Callable[[], float]
    statistics: RunningStatistics = field(default_factory=RunningStatistics)
    total_cost_seconds: float = 0.0

    @property
    def count(self) -> int:
        return self.statistics.count

    @property
    def mean(self) -> float:
        return self.statistics.mean

    @property
    def variance(self) -> float:
        return self.statistics.variance

    @property
    def average_cost(self) -> float:
        if self.count == 0:
            return math.inf
        return self.total_cost_seconds / self.count

    def add_samples(self, number: int) -> None:
        if number < 0:
            raise ValueError("number must be nonnegative.")

        start = time.perf_counter()
        for _ in range(number):
            self.statistics.update(float(self.draw_sample()))
        self.total_cost_seconds += time.perf_counter() - start


@dataclass(frozen=True)
class AdaptiveMLMCResult:
    """Final adaptive MLMC and finest-grid MC estimates."""

    terms: tuple[TermState, ...]
    fine_mc: TermState
    finest_level: int
    sampling_variance: float
    sampling_variance_target: float
    bias_estimate: float
    bias_target: float
    sampling_converged: bool
    bias_converged: bool

    @property
    def mlmc_estimate(self) -> float:
        return sum(term.mean for term in self.terms)

    @property
    def mlmc_standard_error(self) -> float:
        return math.sqrt(self.sampling_variance)


def solve_qoi(mesh: ng.Mesh, fes: ng.H1, kappa) -> float:
    """Solve the Darcy problem and return the right-boundary flux QoI."""
    solution = solve_diffusion(fes, kappa)
    return quantity_of_interest(solution, mesh, kappa)


def estimated_sampling_variance(terms: list[TermState]) -> float:
    """Estimate ``sum_l V_l / N_l`` for independent MLMC terms."""
    return float(
        sum(term.variance / term.count for term in terms)
    )


def optimal_sample_counts(
    terms: list[TermState],
    tolerance: float,
    minimum_samples: int,
) -> np.ndarray:
    """Calculate the cost-minimizing MLMC sample targets from pilot data."""
    variances = np.array(
        [term.variance for term in terms],
        dtype=float,
    )
    costs = np.array(
        [term.average_cost for term in terms],
        dtype=float,
    )

    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        raise RuntimeError("All terms need at least two finite pilot samples.")
    if not np.all(np.isfinite(costs)) or np.any(costs <= 0.0):
        raise RuntimeError("Measured sample costs must be finite and positive.")

    # Floors prevent a zero pilot variance or timer-resolution artifact from
    # assigning zero samples to a genuinely random term.
    positive_variances = variances[variances > 0.0]
    variance_floor = (
        max(float(np.min(positive_variances)) * 1.0e-6, 1.0e-16)
        if positive_variances.size
        else 1.0e-16
    )
    variances = np.maximum(variances, variance_floor)
    costs = np.maximum(costs, 1.0e-12)

    normalizing_sum = float(np.sum(np.sqrt(variances * costs)))
    targets = np.ceil(
        (2.0 / tolerance**2)
        * np.sqrt(variances / costs)
        * normalizing_sum
    ).astype(int)

    return np.maximum(targets, minimum_samples)


def allocate_until_sampling_converged(
    *,
    terms: list[TermState],
    tolerance: float,
    pilot_samples: int,
    maximum_samples_per_term: int,
    maximum_allocation_iterations: int,
    verbose: bool,
) -> bool:
    """Pilot, allocate, and update samples until the variance budget is met."""
    for term in terms:
        if term.count < pilot_samples:
            term.add_samples(pilot_samples - term.count)

    variance_target = tolerance**2 / 2.0

    for allocation_iteration in range(1, maximum_allocation_iterations + 1):
        targets = optimal_sample_counts(
            terms,
            tolerance,
            pilot_samples,
        )
        targets = np.minimum(targets, maximum_samples_per_term)

        for term, target in zip(terms, targets):
            if term.count < int(target):
                term.add_samples(int(target) - term.count)

        current_variance = estimated_sampling_variance(terms)

        if verbose:
            print(f"allocation iteration {allocation_iteration}")
            for term, target in zip(terms, targets):
                print(
                    f"  {term.name:<22} N={term.count:6d}  "
                    f"target={int(target):6d}  "
                    f"V={term.variance:.3e}  "
                    f"C={term.average_cost:.3e}s"
                )
            print(
                f"  estimator variance={current_variance:.3e}, "
                f"target={variance_target:.3e}"
            )

        refreshed_targets = optimal_sample_counts(
            terms,
            tolerance,
            pilot_samples,
        )
        counts_sufficient = all(
            term.count >= min(int(target), maximum_samples_per_term)
            for term, target in zip(terms, refreshed_targets)
        )

        if current_variance <= variance_target and counts_sufficient:
            return True

        if all(term.count >= maximum_samples_per_term for term in terms):
            return False

    return estimated_sampling_variance(terms) <= variance_target


def estimate_remaining_bias(
    *,
    finest_correction_mean: float,
    coarse_maxh: float,
    fine_maxh: float,
    weak_rate_alpha: float,
) -> float:
    """Estimate the residual bias assuming ``|E[Q-Q_h]| ~ h**alpha``."""
    refinement_ratio = coarse_maxh / fine_maxh
    denominator = refinement_ratio**weak_rate_alpha - 1.0
    if denominator <= 0.0:
        raise ValueError("The refinement ratio and weak rate must give a positive denominator.")
    return abs(finest_correction_mean) / denominator


def run_adaptive_comparison(
    *,
    tolerance: float = 1.0e-2,
    level_maxhs: tuple[float, ...] = (0.30, 0.15, 0.075),
    coefficient_grid_size: int = 32,
    num_modes_2d: int = 100,
    correlation_length: float = 0.30,
    standard_deviation: float = 1.0,
    mean_log_conductivity: float = 0.0,
    weak_rate_alpha: float = 1.0,
    pilot_samples: int = 20,
    maximum_samples_per_term: int = 100_000,
    maximum_allocation_iterations: int = 10,
    seed: int = 7,
    verbose: bool = True,
) -> AdaptiveMLMCResult:
    """Run the adaptive MLMC algorithm over the supplied candidate levels."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if len(level_maxhs) < 2:
        raise ValueError("At least two candidate spatial levels are required.")
    if any(value <= 0.0 for value in level_maxhs):
        raise ValueError("All level_maxhs values must be positive.")
    if any(
        fine >= coarse
        for coarse, fine in zip(level_maxhs, level_maxhs[1:])
    ):
        raise ValueError("level_maxhs must decrease strictly from coarse to fine.")
    if weak_rate_alpha <= 0.0:
        raise ValueError("weak_rate_alpha must be positive.")
    if pilot_samples < 2:
        raise ValueError("pilot_samples must be at least 2.")
    if coefficient_grid_size < 2:
        raise ValueError("coefficient_grid_size must be at least 2.")
    if num_modes_2d < 1:
        raise ValueError("num_modes_2d must be positive.")
    if correlation_length <= 0.0:
        raise ValueError("correlation_length must be positive.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    if maximum_samples_per_term < pilot_samples:
        raise ValueError(
            "maximum_samples_per_term must be at least pilot_samples."
        )
    if maximum_allocation_iterations < 1:
        raise ValueError("maximum_allocation_iterations must be positive.")

    levels = tuple(
        build_fixed_mesh(maxh=maxh)
        for maxh in level_maxhs
    )

    frequencies_1d, normalizations_1d, eigenvalues_1d, _ = get_1d_eigenpairs(
        num_modes=num_modes_2d,
        correlation_length=correlation_length,
    )
    (
        unit_eigenvalues_2d,
        _,
        evaluate_eigenfunctions_2d,
    ) = leading_2d_eigenpairs(
        eigenvalues_1d=eigenvalues_1d,
        frequencies_1d=frequencies_1d,
        normalizations_1d=normalizations_1d,
        correlation_length=correlation_length,
        num_modes_2d=num_modes_2d,
        method="heap",
    )

    evaluate_log_conductivity = make_2d_kl_evaluator(
        eigenvalues_2d=unit_eigenvalues_2d,
        eigenfunction_evaluator=evaluate_eigenfunctions_2d,
        mean_log_conductivity=mean_log_conductivity,
        variance=standard_deviation**2,
    )
    X, Y, _ = cartesian_grid_2d(
        coefficient_grid_size,
        coefficient_grid_size,
    )

    def coefficient_from_xi(xi: np.ndarray):
        log_kappa = evaluate_log_conductivity(X, Y, xi)
        return voxel_coefficient_2d(
            lognormal_transform(log_kappa),
            linear=True,
        )

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(levels) + 1)
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

        def draw() -> float:
            xi = rng.standard_normal(num_modes_2d)
            kappa = coefficient_from_xi(xi)
            q_lower = solve_qoi(lower_mesh, lower_fes, kappa)
            q_upper = solve_qoi(upper_mesh, upper_fes, kappa)
            return q_upper - q_lower

        return draw

    all_terms = [TermState("level 0: Q_0", draw_level_zero)]
    all_terms.extend(
        TermState(
            f"level {level}: Q_{level}-Q_{level - 1}",
            make_correction_sampler(level),
        )
        for level in range(1, len(levels))
    )

    bias_target = tolerance / math.sqrt(2.0)
    active_finest_level = 1
    sampling_converged = False
    bias_converged = False
    bias_estimate = math.inf

    while True:
        active_terms = all_terms[: active_finest_level + 1]
        if verbose:
            print()
            print(f"active finest level: {active_finest_level}")

        sampling_converged = allocate_until_sampling_converged(
            terms=active_terms,
            tolerance=tolerance,
            pilot_samples=pilot_samples,
            maximum_samples_per_term=maximum_samples_per_term,
            maximum_allocation_iterations=maximum_allocation_iterations,
            verbose=verbose,
        )

        bias_estimate = estimate_remaining_bias(
            finest_correction_mean=active_terms[-1].mean,
            coarse_maxh=level_maxhs[active_finest_level - 1],
            fine_maxh=level_maxhs[active_finest_level],
            weak_rate_alpha=weak_rate_alpha,
        )
        bias_converged = bias_estimate <= bias_target

        if verbose:
            print(
                f"estimated bias={bias_estimate:.3e}, "
                f"target={bias_target:.3e}"
            )

        if sampling_converged and bias_converged:
            break
        if not sampling_converged:
            break
        if active_finest_level + 1 >= len(levels):
            break

        active_finest_level += 1

    active_terms = all_terms[: active_finest_level + 1]
    finest_mesh, finest_fes = levels[active_finest_level]

    def draw_fine_mc() -> float:
        xi = fine_mc_rng.standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)
        return solve_qoi(finest_mesh, finest_fes, kappa)

    fine_mc = TermState(
        f"fine MC Q_{active_finest_level}",
        draw_fine_mc,
    )
    fine_mc.add_samples(pilot_samples)

    fine_sampling_variance_target = tolerance**2 / 2.0
    for _ in range(maximum_allocation_iterations):
        target = max(
            pilot_samples,
            math.ceil(2.0 * fine_mc.variance / tolerance**2),
        )
        target = min(target, maximum_samples_per_term)
        if fine_mc.count < target:
            fine_mc.add_samples(target - fine_mc.count)
            continue
        if fine_mc.variance / fine_mc.count <= fine_sampling_variance_target:
            break
        if fine_mc.count >= maximum_samples_per_term:
            break

    result = AdaptiveMLMCResult(
        terms=tuple(active_terms),
        fine_mc=fine_mc,
        finest_level=active_finest_level,
        sampling_variance=estimated_sampling_variance(active_terms),
        sampling_variance_target=tolerance**2 / 2.0,
        bias_estimate=bias_estimate,
        bias_target=bias_target,
        sampling_converged=sampling_converged,
        bias_converged=bias_converged,
    )
    print_summary(result)
    return result


def print_summary(result: AdaptiveMLMCResult) -> None:
    """Print allocations, convergence checks, and estimator comparison."""
    print()
    print("Adaptive MLMC summary")
    print(
        f"{'term':<28} {'N':>7} {'mean':>14} {'variance':>14} "
        f"{'cost/sample[s]':>15}"
    )
    print("-" * 84)
    for term in result.terms:
        print(
            f"{term.name:<28} {term.count:7d} {term.mean:14.6e} "
            f"{term.variance:14.6e} {term.average_cost:15.3e}"
        )

    print()
    print(
        f"MLMC estimate: {result.mlmc_estimate:.8e} "
        f"+/- {result.mlmc_standard_error:.3e}"
    )
    print(
        f"sampling variance: {result.sampling_variance:.3e} "
        f"<= {result.sampling_variance_target:.3e}: "
        f"{result.sampling_converged}"
    )
    print(
        f"estimated bias: {result.bias_estimate:.3e} "
        f"<= {result.bias_target:.3e}: {result.bias_converged}"
    )
    print(
        f"fine MC estimate: {result.fine_mc.mean:.8e} "
        f"+/- {result.fine_mc.statistics.standard_error:.3e} "
        f"(N={result.fine_mc.count})"
    )
    print(
        "estimate difference (MLMC - fine MC): "
        f"{result.mlmc_estimate - result.fine_mc.mean:.3e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1.0e-2)
    parser.add_argument(
        "--level-maxhs",
        nargs="+",
        type=float,
        default=[0.30, 0.15, 0.075],
    )
    parser.add_argument("--coefficient-grid-size", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=100, dest="num_modes_2d")
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--sigma", type=float, default=1.0, dest="standard_deviation")
    parser.add_argument("--mean-log-kappa", type=float, default=0.0)
    parser.add_argument("--weak-rate-alpha", type=float, default=1.0)
    parser.add_argument("--pilot-samples", type=int, default=20)
    parser.add_argument("--maximum-samples-per-term", type=int, default=100_000)
    parser.add_argument("--maximum-allocation-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_adaptive_comparison(
        tolerance=arguments.tolerance,
        level_maxhs=tuple(arguments.level_maxhs),
        coefficient_grid_size=arguments.coefficient_grid_size,
        num_modes_2d=arguments.num_modes_2d,
        correlation_length=arguments.correlation_length,
        standard_deviation=arguments.standard_deviation,
        mean_log_conductivity=arguments.mean_log_kappa,
        weak_rate_alpha=arguments.weak_rate_alpha,
        pilot_samples=arguments.pilot_samples,
        maximum_samples_per_term=arguments.maximum_samples_per_term,
        maximum_allocation_iterations=arguments.maximum_allocation_iterations,
        seed=arguments.seed,
        verbose=not arguments.quiet,
    )
