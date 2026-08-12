"""Adaptive MLMC and finest-level MC solved with geometric multigrid.

The nested spatial hierarchy is built once.  For every KL realization, the
conductivity fields on the required levels are updated, the stiffness forms
are reassembled, cached smoothers are refreshed, and the requested PDEs are
solved with V-cycles.  A correction sample uses the same ``xi`` for
``Q_l`` and ``Q_{l-1}``.

The MLMC sample allocation and bias test follow the adaptive implementation
in ``adaptive_mlmc_vs_mc.py``.  Run from the project root, for example:

    .venv/bin/python \
        examples/KLMC/multilevel/multilevel_mlmc_vs_mv_multigridsolver.py \
        --tolerance 0.02 --maximum-levels 4
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from multigrid_cycles import (  # noqa: E402
    MultigridHierarchy,
    MultigridSolver,
    VCycleConfig,
    build_hierarchy,
)
from examples.KLMC.multilevel.adaptive_mlmc_vs_mc import (  # noqa: E402
    AdaptiveMLMCResult,
    TermState,
    allocate_until_sampling_converged,
    estimated_sampling_variance,
    estimate_remaining_bias,
    print_summary,
)
from examples.KLMC.utils.analytical_eigenfunctions import (  # noqa: E402
    get_1d_eigenpairs,
    leading_2d_eigenpairs,
    make_2d_kl_evaluator,
)
from examples.KLMC.utils.one_level_utils import (  # noqa: E402
    quantity_of_interest,
)


@dataclass
class ReusableMultigridProblem:
    """Nested hierarchy whose conductivity fields can be updated in place."""

    hierarchy: MultigridHierarchy
    conductivity_fields: tuple[ng.GridFunction, ...]
    solver_config: VCycleConfig
    relative_solver_tolerance: float
    maximum_vcycles: int
    allow_direct_fallback: bool
    direct_fallback_counts: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        coarse_maxh: float,
        number_of_levels: int,
        order: int,
        relative_solver_tolerance: float,
        maximum_vcycles: int,
        pre_sweeps: int,
        post_sweeps: int,
        allow_direct_fallback: bool,
    ) -> "ReusableMultigridProblem":
        """Build nested geometry, FE spaces, forms, and grid transfers once."""
        if number_of_levels < 2:
            raise ValueError("number_of_levels must be at least 2.")
        if coarse_maxh <= 0.0:
            raise ValueError("coarse_maxh must be positive.")
        if order < 1:
            raise ValueError("order must be positive.")
        if relative_solver_tolerance <= 0.0:
            raise ValueError("relative_solver_tolerance must be positive.")
        if maximum_vcycles < 1:
            raise ValueError("maximum_vcycles must be positive.")

        conductivity_fields: list[ng.GridFunction] = []

        def form_setup(fes):
            coefficient_space = ng.H1(fes.mesh, order=1)
            conductivity = ng.GridFunction(coefficient_space)
            conductivity.Set(1.0)
            conductivity_fields.append(conductivity)

            u, v = fes.TnT()
            a = ng.BilinearForm(fes, symmetric=True)
            a += (
                conductivity
                * ng.InnerProduct(ng.grad(u), ng.grad(v))
                * ng.dx
            )
            f = ng.LinearForm(fes)
            return a, f

        coarse_mesh = ng.Mesh(
            unit_square.GenerateMesh(maxh=coarse_maxh)
        )
        hierarchy = build_hierarchy(
            coarse_mesh,
            form_setup,
            n_refines=number_of_levels - 1,
            order=order,
            dirichlet="left|right",
            dirichlet_value={"left": -1.0, "right": 0.0},
            verbose=False,
        )

        if len(conductivity_fields) != hierarchy.nlevels:
            raise RuntimeError(
                "Expected one conductivity field per multigrid level."
            )

        return cls(
            hierarchy=hierarchy,
            conductivity_fields=tuple(conductivity_fields),
            solver_config=VCycleConfig(
                pre_sweeps=pre_sweeps,
                post_sweeps=post_sweeps,
                coarse_direct=True,
            ),
            relative_solver_tolerance=relative_solver_tolerance,
            maximum_vcycles=maximum_vcycles,
            allow_direct_fallback=allow_direct_fallback,
            direct_fallback_counts=np.zeros(
                number_of_levels,
                dtype=int,
            ),
        )

    def update_conductivity(self, kappa, finest_level: int) -> None:
        """Interpolate one sample and reassemble levels 0 through finest."""
        if not 0 <= finest_level < self.hierarchy.nlevels:
            raise IndexError("finest_level is outside the hierarchy.")

        for level_index in range(finest_level + 1):
            conductivity = self.conductivity_fields[level_index]
            level = self.hierarchy.levels[level_index]

            conductivity.Set(kappa)
            level.a.Assemble()
            level.refresh()

    def reset_solution(
        self,
        level_index: int,
        *,
        use_coarse_initial_guess: bool,
    ) -> None:
        """Initialize one level and impose its nonzero boundary data."""
        level = self.hierarchy.levels[level_index]

        if use_coarse_initial_guess:
            if level_index == 0 or level.P is None:
                raise ValueError(
                    "A coarse initial guess requires level_index >= 1."
                )
            coarse_solution = self.hierarchy.levels[level_index - 1].gfu
            level.gfu.vec.data = level.P * coarse_solution.vec
        else:
            level.gfu.vec.FV().NumPy()[:] = 0.0

        level.enforce_dirichlet(level.gfu.vec)

    def solve_level(
        self,
        level_index: int,
        *,
        use_coarse_initial_guess: bool = False,
    ) -> ng.GridFunction:
        """Solve level_index directly at level zero or by repeated V-cycles."""
        self.reset_solution(
            level_index,
            use_coarse_initial_guess=use_coarse_initial_guess,
        )
        level = self.hierarchy.levels[level_index]

        if level_index == 0:
            residual = level.residual(level.f.vec, level.gfu.vec)
            correction = level.gfu.vec.CreateVector()
            correction.data = (
                level.a.mat.Inverse(level.fes.FreeDofs())
                * residual
            )
            level.gfu.vec.data += correction
            level.enforce_dirichlet(level.gfu.vec)
            return level.gfu

        subhierarchy = MultigridHierarchy(
            self.hierarchy.levels[: level_index + 1]
        )
        solver = MultigridSolver(
            subhierarchy,
            self.solver_config,
        )
        initial_residual = level.residual_norm()
        solver.solve(
            max_cycles=self.maximum_vcycles,
            tol=self.relative_solver_tolerance,
            norms=("l2",),
            stop_norm="l2",
            verbose=False,
        )
        final_residual = level.residual_norm()

        if (
            initial_residual > 0.0
            and final_residual
            > self.relative_solver_tolerance * initial_residual
        ):
            if not self.allow_direct_fallback:
                raise RuntimeError(
                    f"Level {level_index} V-cycle solve did not reach its "
                    f"relative tolerance after {self.maximum_vcycles} cycles: "
                    f"initial residual={initial_residual:.3e}, "
                    f"final residual={final_residual:.3e}. Increase "
                    "--maximum-vcycles, relax --linear-tolerance, or allow "
                    "the default direct fallback."
                )

            # Complete the unresolved residual exactly on the free DOFs.
            # This preserves sample correctness while recording that the
            # V-cycle was not robust for this conductivity realization.
            residual = level.residual(level.f.vec, level.gfu.vec)
            correction = level.gfu.vec.CreateVector()
            correction.data = (
                level.a.mat.Inverse(level.fes.FreeDofs())
                * residual
            )
            level.gfu.vec.data += correction
            level.enforce_dirichlet(level.gfu.vec)
            self.direct_fallback_counts[level_index] += 1

        return subhierarchy.finest.gfu

    def evaluate_qoi(self, level_index: int) -> float:
        """Evaluate the flux QoI using the coefficient assembled on this level."""
        level = self.hierarchy.levels[level_index]
        conductivity = self.conductivity_fields[level_index]
        return quantity_of_interest(
            level.gfu,
            level.mesh,
            conductivity,
        )

    def solve_qoi(
        self,
        level_index: int,
        *,
        use_coarse_initial_guess: bool = False,
    ) -> float:
        """Solve one level and evaluate its QoI."""
        self.solve_level(
            level_index,
            use_coarse_initial_guess=use_coarse_initial_guess,
        )
        return self.evaluate_qoi(level_index)


def run_comparison(
    *,
    tolerance: float = 2.0e-2,
    coarse_maxh: float = 0.30,
    maximum_levels: int = 4,
    order: int = 1,
    coefficient_grid_size: int = 32,
    num_modes_2d: int = 100,
    correlation_length: float = 0.30,
    standard_deviation: float = 1.0,
    mean_log_conductivity: float = 0.0,
    weak_rate_alpha: float = 1.0,
    refinement_ratio: float = 2.0,
    pilot_samples: int = 20,
    maximum_samples_per_term: int = 100_000,
    maximum_allocation_iterations: int = 10,
    relative_solver_tolerance: float = 1.0e-10,
    maximum_vcycles: int = 100,
    pre_sweeps: int = 2,
    post_sweeps: int = 2,
    allow_direct_fallback: bool = True,
    seed: int = 7,
    verbose: bool = True,
) -> AdaptiveMLMCResult:
    """Run adaptive MLMC and fine MC using the reusable MG hierarchy."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if maximum_levels < 2:
        raise ValueError("maximum_levels must be at least 2.")
    if coefficient_grid_size < 2:
        raise ValueError("coefficient_grid_size must be at least 2.")
    if num_modes_2d < 1:
        raise ValueError("num_modes_2d must be positive.")
    if correlation_length <= 0.0:
        raise ValueError("correlation_length must be positive.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    if weak_rate_alpha <= 0.0:
        raise ValueError("weak_rate_alpha must be positive.")
    if refinement_ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one.")
    if pilot_samples < 2:
        raise ValueError("pilot_samples must be at least 2.")
    if maximum_samples_per_term < pilot_samples:
        raise ValueError(
            "maximum_samples_per_term must be at least pilot_samples."
        )

    problem = ReusableMultigridProblem.build(
        coarse_maxh=coarse_maxh,
        number_of_levels=maximum_levels,
        order=order,
        relative_solver_tolerance=relative_solver_tolerance,
        maximum_vcycles=maximum_vcycles,
        pre_sweeps=pre_sweeps,
        post_sweeps=post_sweeps,
        allow_direct_fallback=allow_direct_fallback,
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
    child_seeds = seed_sequence.spawn(maximum_levels + 1)
    term_rngs = tuple(
        np.random.default_rng(child_seed)
        for child_seed in child_seeds[:-1]
    )
    fine_mc_rng = np.random.default_rng(child_seeds[-1])

    def draw_level_zero() -> float:
        xi = term_rngs[0].standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)
        problem.update_conductivity(kappa, finest_level=0)
        return problem.solve_qoi(0)

    def make_correction_sampler(level_index: int) -> Callable[[], float]:
        rng = term_rngs[level_index]

        def draw() -> float:
            xi = rng.standard_normal(num_modes_2d)
            kappa = coefficient_from_xi(xi)
            problem.update_conductivity(
                kappa,
                finest_level=level_index,
            )
            q_lower = problem.solve_qoi(level_index - 1)
            q_upper = problem.solve_qoi(
                level_index,
                use_coarse_initial_guess=True,
            )
            return q_upper - q_lower

        return draw

    all_terms = [TermState("level 0: Q_0", draw_level_zero)]
    all_terms.extend(
        TermState(
            f"level {level}: Q_{level}-Q_{level - 1}",
            make_correction_sampler(level),
        )
        for level in range(1, maximum_levels)
    )

    bias_target = tolerance / math.sqrt(2.0)
    active_finest_level = 1
    sampling_converged = False
    bias_converged = False
    bias_estimate = math.inf

    if verbose:
        print("Reusable nested multigrid hierarchy")
        problem.hierarchy.info()
        print(f"linear solver tolerance: {relative_solver_tolerance:.3e}")

    while True:
        active_terms = all_terms[: active_finest_level + 1]
        if verbose:
            print()
            print(f"active finest MLMC level: {active_finest_level}")

        sampling_converged = allocate_until_sampling_converged(
            terms=active_terms,
            tolerance=tolerance,
            pilot_samples=pilot_samples,
            maximum_samples_per_term=maximum_samples_per_term,
            maximum_allocation_iterations=maximum_allocation_iterations,
            verbose=verbose,
        )

        nominal_coarse_h = coarse_maxh / (
            refinement_ratio ** (active_finest_level - 1)
        )
        nominal_fine_h = coarse_maxh / (
            refinement_ratio**active_finest_level
        )
        bias_estimate = estimate_remaining_bias(
            finest_correction_mean=active_terms[-1].mean,
            coarse_maxh=nominal_coarse_h,
            fine_maxh=nominal_fine_h,
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
        if active_finest_level + 1 >= maximum_levels:
            break

        active_finest_level += 1

    active_terms = all_terms[: active_finest_level + 1]

    def draw_fine_mc() -> float:
        xi = fine_mc_rng.standard_normal(num_modes_2d)
        kappa = coefficient_from_xi(xi)
        problem.update_conductivity(
            kappa,
            finest_level=active_finest_level,
        )
        return problem.solve_qoi(active_finest_level)

    fine_mc = TermState(
        f"fine MC Q_{active_finest_level}",
        draw_fine_mc,
    )
    fine_mc.add_samples(pilot_samples)

    fine_variance_target = tolerance**2 / 2.0
    for _ in range(maximum_allocation_iterations):
        target = max(
            pilot_samples,
            math.ceil(2.0 * fine_mc.variance / tolerance**2),
        )
        target = min(target, maximum_samples_per_term)
        if fine_mc.count < target:
            fine_mc.add_samples(target - fine_mc.count)
            continue
        if fine_mc.variance / fine_mc.count <= fine_variance_target:
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
    print()
    print("Direct fallbacks after stalled V-cycles:")
    for level_index, count in enumerate(problem.direct_fallback_counts):
        if level_index <= active_finest_level:
            print(f"  level {level_index}: {count}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=2.0e-2)
    parser.add_argument("--coarse-maxh", type=float, default=0.30)
    parser.add_argument("--maximum-levels", type=int, default=4)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--coefficient-grid-size", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=100, dest="num_modes_2d")
    parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
    parser.add_argument("--sigma", type=float, default=1.0, dest="standard_deviation")
    parser.add_argument("--mean-log-kappa", type=float, default=0.0)
    parser.add_argument("--weak-rate-alpha", type=float, default=1.0)
    parser.add_argument("--refinement-ratio", type=float, default=2.0)
    parser.add_argument("--pilot-samples", type=int, default=20)
    parser.add_argument("--maximum-samples-per-term", type=int, default=100_000)
    parser.add_argument("--maximum-allocation-iterations", type=int, default=10)
    parser.add_argument("--linear-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--maximum-vcycles", type=int, default=500)
    parser.add_argument("--pre-sweeps", type=int, default=2)
    parser.add_argument("--post-sweeps", type=int, default=2)
    parser.add_argument(
        "--no-direct-fallback",
        action="store_true",
        help="raise an error instead of finishing a stalled V-cycle solve directly",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_comparison(
        tolerance=arguments.tolerance,
        coarse_maxh=arguments.coarse_maxh,
        maximum_levels=arguments.maximum_levels,
        order=arguments.order,
        coefficient_grid_size=arguments.coefficient_grid_size,
        num_modes_2d=arguments.num_modes_2d,
        correlation_length=arguments.correlation_length,
        standard_deviation=arguments.standard_deviation,
        mean_log_conductivity=arguments.mean_log_kappa,
        weak_rate_alpha=arguments.weak_rate_alpha,
        refinement_ratio=arguments.refinement_ratio,
        pilot_samples=arguments.pilot_samples,
        maximum_samples_per_term=arguments.maximum_samples_per_term,
        maximum_allocation_iterations=arguments.maximum_allocation_iterations,
        relative_solver_tolerance=arguments.linear_tolerance,
        maximum_vcycles=arguments.maximum_vcycles,
        pre_sweeps=arguments.pre_sweeps,
        post_sweeps=arguments.post_sweeps,
        allow_direct_fallback=not arguments.no_direct_fallback,
        seed=arguments.seed,
        verbose=not arguments.quiet,
    )
