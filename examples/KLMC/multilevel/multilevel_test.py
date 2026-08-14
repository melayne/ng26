
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


def create_voxel_coefficient(
    xi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    evaluate_log_conductivity: Callable[
        [
            np.ndarray | float,
            np.ndarray | float,
            np.ndarray,
        ],
        np.ndarray,
    ],
):
    """
    Create one lognormal VoxelCoefficient from a KL coefficient vector.

    The same xi can be used to create the coefficient shared by
    multiple spatial levels.
    """
    log_kappa = evaluate_log_conductivity(X, Y, xi)

    return voxel_coefficient_2d(
        lognormal_transform(log_kappa),
        linear=True,
    )

def make_coefficient_from_xi(
    X: np.ndarray,
    Y: np.ndarray,
    evaluate_log_conductivity: Callable[
        [
            np.ndarray | float,
            np.ndarray | float,
            np.ndarray,
        ],
        np.ndarray,
    ],
) -> Callable[
    [np.ndarray],
    ng.CoefficientFunction,
]:
    """
    Construct a function that maps one KL coefficient vector to kappa.

    X, Y, and the KL evaluator remain fixed. The returned function only
    requires the newly sampled Gaussian coefficient vector xi.
    """

    def coefficient_from_xi(
        xi: np.ndarray,
    ) -> ng.CoefficientFunction:
        return create_voxel_coefficient(
            xi=xi,
            X=X,
            Y=Y,
            evaluate_log_conductivity=evaluate_log_conductivity,
        )

    return coefficient_from_xi


def create_rngs(seed: int, num_levels: int) -> tuple[np.random.Generator, ...]:
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(num_levels + 1)
    return tuple(
        np.random.default_rng(child_seed)
        for child_seed in child_seeds[:-1]
    )

@dataclass
class MCLevel:
    """ 
    Single level of the multilevel monte carlo simulation. 
    """

    level_index: int
    hierarchy: MultigridHierarchy
    solver_config: VCycleConfig
    relative_solver_tolerance: float
    maximum_vcycles: int
    conductivity_fields: tuple[ng.GridFunction, ...]

    @classmethod
    def build(
        cls,
        *,
        coarse_maxh: float,
        level_index: int,
        relative_solver_tolerance: float,
        maximum_vcycles: int,
        pre_sweeps: int,
        post_sweeps: int,
        coarse_direct: bool,
        coarse_sweeps: int,
        dirichlet: str,
        dirichlet_value: dict[str, float],
        order: int,
    ) -> "MCLevel":


        if level_index < 0:
            raise ValueError("level_index must be nonnegative.")

        if coarse_maxh <= 0.0:
            raise ValueError(
                "coarse_maxh must be positive."
            )

        if order < 1:
            raise ValueError(
                "order must be positive."
            )

        if relative_solver_tolerance <= 0.0:
            raise ValueError(
                "relative_solver_tolerance must be positive."
            )

        if maximum_vcycles < 1:
            raise ValueError(
                "maximum_vcycles must be positive."
            )

        conductivity_fields: list[ng.GridFunction] = []

        def form_setup(fes):
            coefficient_space = ng.L2(fes.mesh, order=0)
            # coefficient_space = ng.H1(fes.mesh, order=1)

            conductivity = ng.GridFunction(coefficient_space)
            conductivity.Set(ng.CoefficientFunction(1.0))
            conductivity_fields.append(conductivity)

            u, v = fes.TnT()
            a = ng.BilinearForm(fes, symmetric=True)
            a += (
                conductivity
                * ng.InnerProduct(ng.grad(u), ng.grad(v))
                * ng.dx
            )
            f = ng.LinearForm(fes)
            # f += v * ng.dx

            return a, f

    
        coarse_mesh = ng.Mesh(unit_square.GenerateMesh(maxh=coarse_maxh))
        
        hierarchy = build_hierarchy(
            coarse_mesh,
            form_setup,
            n_refines=level_index,
            order=order,
            dirichlet=dirichlet,
            dirichlet_value=dirichlet_value,
            verbose=False,
        )

        if len(conductivity_fields) != hierarchy.nlevels:
            raise RuntimeError(
                "Expected one conductivity field per hierarchy level."
            )

        return cls(
            level_index=level_index,
            hierarchy=hierarchy,
            solver_config=VCycleConfig(
                pre_sweeps=pre_sweeps,
                post_sweeps=post_sweeps,
                coarse_direct=coarse_direct,
                coarse_sweeps=coarse_sweeps,
            ),
            relative_solver_tolerance=relative_solver_tolerance,
            maximum_vcycles=maximum_vcycles,
            conductivity_fields=tuple(conductivity_fields),
        )

    def update_conductivity(    
        self,
        kappa: ng.CoefficientFunction,
    ) -> None:
        """
        Update the conductivity on every level required to solve Q_l.

        Parameters
        ----------
        kappa:
            One realization of the physical conductivity field
            ``kappa(x, y)``. This may be an ``ng.VoxelCoefficient``.

            A ``VoxelCoefficient`` is an NGSolve ``CoefficientFunction``,
            so it can be passed directly to ``GridFunction.Set()``. It is
            defined in physical coordinates and is not tied to a particular
            finite-element mesh.

        Notes
        -----
        Each entry in ``conductivity_fields`` is a ``GridFunction`` associated
        with the coefficient space on a particular mesh level. Calling

            conductivity.Set(kappa)

        evaluates the same physical-space coefficient ``kappa`` and represents
        it in that level's coefficient space. This remains valid even though
        the levels have different meshes and different degrees of freedom.

        Consequently, all updated levels use the same random-field realization,
        while retaining their own level-dependent discrete representations.
        This is required for an MLMC correction

            Q_l(kappa) - Q_(l-1)(kappa),

        where both PDE solves must use the same random input.

        Updating a coefficient ``GridFunction`` does not automatically update
        the stiffness matrix. The bilinear form must therefore be reassembled,
        and matrix-dependent multigrid data must subsequently be refreshed.
        """

        if (
            len(self.conductivity_fields)
            != self.hierarchy.nlevels
        ):
            raise RuntimeError(
                "Expected one conductivity field per hierarchy level."
            )


        for level, conductivity in zip(
            self.hierarchy.levels,
            self.conductivity_fields,
            strict=True,
        ):
            # GridFunction.Set accepts a CoefficientFunction. In particular,
            # passing the same VoxelCoefficient to every level is valid:
            # each level evaluates it on its own mesh and stores its own
            # discrete representation of that physical conductivity field.

            conductivity.Set(kappa)
            level.a.Assemble()
            level.refresh()

    def reset_solutions(self) -> None:
        """
        Reset the stored PDE solution on every level in this hierarchy.

        Each solution is set to zero and then its physical Dirichlet
        boundary values are restored.
        """
        for level in self.hierarchy.levels:
            level.gfu.vec.FV().NumPy()[:] = 0.0
            level.enforce_dirichlet(level.gfu.vec)

    def solve_q0_direct(self) -> ng.GridFunction:
        """
        Solve the physical PDE directly on level zero.

        The solution vector already contains the prescribed Dirichlet
        boundary values. Therefore, solve for a zero-boundary correction
        using the remaining residual.
        """
        if self.level_index != 0:
            raise RuntimeError(
                "solve_q0_direct may only be called for level zero."
            )

        level = self.hierarchy.finest

        residual = level.residual(
            level.f.vec,
            level.gfu.vec,
        )

        correction = level.gfu.vec.CreateVector()
        correction.data = (
            level.a.mat.Inverse(
                level.fes.FreeDofs()
            )
            * residual
        )

        level.gfu.vec.data += correction
        level.enforce_dirichlet(level.gfu.vec)

        return level.gfu

    def solve_with_vcycles(self) -> ng.GridFunction:
        """
        Solve Q_l using repeated V-cycles.

        This method is only used for levels greater than zero.
        """
        if self.level_index == 0:
            raise RuntimeError(
                "Level zero must be solved directly."
            )

        level = self.hierarchy.finest

        solver = MultigridSolver(
            self.hierarchy,
            self.solver_config,
        )

        initial_residual = level.residual_norm()

        if initial_residual == 0.0:
            return level.gfu

        solver.solve(
            max_cycles=self.maximum_vcycles,
            tol=self.relative_solver_tolerance,
            norms=("l2",),
            stop_norm="l2",
            verbose=False,
        )

        final_residual = level.residual_norm()
        relative_residual = (
            final_residual / initial_residual
        )

        if relative_residual > self.relative_solver_tolerance:
            raise RuntimeError(
                f"Level {self.level_index} did not converge after "
                f"{self.maximum_vcycles} V-cycles: "
                f"relative residual={relative_residual:.3e}, "
                f"required={self.relative_solver_tolerance:.3e}."
            )

        return level.gfu
    
    def solve(self) -> ng.GridFunction:
        """
        Solve this object's Q_l problem.

        Q_0 is solved directly. Higher levels are solved using
        repeated V-cycles.
        """

        self.reset_solutions()

        if self.level_index == 0:
            return self.solve_q0_direct()

        return self.solve_with_vcycles()

    def evaluate_qoi(self) -> float:
        """
        Evaluate the QoI for this object's solved discretization Q_l.
        """
        level = self.hierarchy.finest
        conductivity = self.conductivity_fields[-1]

        return quantity_of_interest(
            level.gfu,
            level.mesh,
            conductivity,
        )

    def solve_qoi(
        self,
        kappa: ng.CoefficientFunction,
    ) -> float:
        """
        Update the conductivity, solve Q_l, and evaluate its QoI.
        """
        self.update_conductivity(kappa)
        self.solve()

        return self.evaluate_qoi()


@dataclass
class MLMCTerm:
    """
    One independent MLMC estimator term.

    For level zero:

        Y_0 = Q_0.

    For higher levels:

        Y_l = Q_l - Q_(l-1).

    The upper and lower MCLevel objects are privately owned by this
    term and are not shared with other terms.
    """

    level_index: int

    upper_level: MCLevel
    lower_level: MCLevel | None

    rng: np.random.Generator

    coefficient_from_xi: Callable[
        [np.ndarray],
        ng.CoefficientFunction,
    ]

    number_of_modes: int

    sample_count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0 # sum of squares sum(xi - mean)^2

    upper_qois: list[float] = field(
        default_factory=list
    )

    lower_qois: list[float] = field(
        default_factory=list
    )

    sample_durations_seconds: list[float] = field(
        default_factory=list
    )

    def draw_sample(self) -> float:
        """
        Draw, solve, and store one sample of this MLMC term.

        The recorded duration includes drawing ``xi``, constructing
        ``kappa``, solving the required PDE level or level pair,
        evaluating the QoI, and updating the correction statistics.

        A duration is stored only after the complete sample succeeds.
        """
        sample_started = time.perf_counter()

        xi = self.rng.standard_normal(
            self.number_of_modes
        )

        # Both discretizations within this term receive the same
        # physical random-field realization.
        kappa = self.coefficient_from_xi(xi)

        if self.level_index == 0:
            q_lower = None
            q_upper = self.upper_level.solve_qoi(
                kappa
            )
            correction = q_upper
        else:
            if self.lower_level is None:
                raise RuntimeError(
                    "A correction term requires a lower MCLevel."
                )

            q_lower = self.lower_level.solve_qoi(
                kappa
            )

            q_upper = self.upper_level.solve_qoi(
                kappa
            )
            correction = q_upper - q_lower

        self.update_statistics(
            q_upper=q_upper,
            q_lower=q_lower,
        )

        elapsed_seconds = (
            time.perf_counter() - sample_started
        )
        self.sample_durations_seconds.append(
            elapsed_seconds
        )

        return correction

    def add_samples(
        self,
        number_of_samples: int,
    ) -> None:
        """
        Draw and store additional samples of this MLMC correction term.
        """
        if number_of_samples < 0:
            raise ValueError(
                "number_of_samples must be nonnegative."
            )

        for _ in range(number_of_samples):
            self.draw_sample()


    def update_statistics(
        self,
        *,
        q_upper: float,
        q_lower: float | None = None,
    ) -> None:
        q_upper = float(q_upper)


        if self.level_index == 0:
            if q_lower is not None:
                raise ValueError("Y_0 must not receive q_lower.")
            correction = q_upper
        else:
            if q_lower is None:
                raise ValueError(
                    "A correction term requires q_lower."
                )

            q_lower = float(q_lower)

            correction = q_upper - q_lower


        self.upper_qois.append(q_upper)

        if q_lower is not None:
            self.lower_qois.append(q_lower)

        self.sample_count += 1

        difference = correction - self.mean
        self.mean += difference / self.sample_count

        difference_from_new_mean = (
            correction - self.mean
        )

        self.sum_squared_deviations += (
            difference
            * difference_from_new_mean
        )

    @property
    def sample_variance(self) -> float:
        if self.sample_count < 2:
            return float("inf")

        return (
            self.sum_squared_deviations
            / (self.sample_count - 1)
        )

    @property
    def variance_of_mean(self) -> float:
        if self.sample_count < 2:
            return float("inf")

        return self.sample_variance / self.sample_count

    @property
    def corrections(self) -> np.ndarray:
        """
        Return all stored samples of Y_l.
        """
        upper = np.asarray(
            self.upper_qois,
            dtype=float,
        )

        if self.level_index == 0:
            return upper

        lower = np.asarray(
            self.lower_qois,
            dtype=float,
        )

        return upper - lower

    @property
    def total_sampling_time_seconds(self) -> float:
        """
        Return the total time spent generating successful Y_l samples.
        """
        return float(
            sum(self.sample_durations_seconds)
        )

    @property
    def mean_sample_time_seconds(self) -> float:
        """
        Return the average time required for one successful Y_l sample.
        """
        if not self.sample_durations_seconds:
            return float("nan")

        return float(
            np.mean(self.sample_durations_seconds)
        )

    def run_to_mean_variance_target(
        self,
        target_variance: float,
        *,
        minimum_samples: int = 10,
        maximum_samples: int = 100_000,
        samples_per_iteration: int = 1,
        verbose: bool = False,
    ) -> bool:
        """
        Sample this correction term until its estimated mean variance
        reaches the requested target.

        The stopping criterion is

            sample_variance(Y_l) / N_l <= target_variance.

        Parameters
        ----------
        target_variance:
            Target variance for the estimated correction mean

                mean(Y_l).

            This is not the raw variance of individual Y_l samples.

        minimum_samples:
            Minimum initial number of samples. At least two samples are
            required to estimate a sample variance.

        maximum_samples:
            Maximum total number of samples allowed for this term.

        samples_per_iteration:
            Number of samples added before checking the stopping criterion
            again. The adaptive loop is allowed at most

                ceil((maximum_samples - minimum_samples)
                     / samples_per_iteration)

            iterations so the sample budget can be fully used.

        verbose:
            Print the current variance after each sampling iteration.

        Returns
        -------
        bool:
            True if the term reached its target. False if a safety limit was
            reached first.

        Notes
        -----
        Existing samples are retained. Therefore, calling this method again
        continues from the term's current state.
        """
        if (
            not np.isfinite(target_variance)
            or target_variance <= 0.0
        ):
            raise ValueError(
                "target_variance must be finite and positive."
            )

        if minimum_samples < 2:
            raise ValueError(
                "minimum_samples must be at least 2."
            )

        if maximum_samples < minimum_samples:
            raise ValueError(
                "maximum_samples must be at least minimum_samples."
            )

        if samples_per_iteration < 1:
            raise ValueError(
                "samples_per_iteration must be positive."
            )

        maximum_iterations = math.ceil(
            (maximum_samples - minimum_samples)
            / samples_per_iteration
        )

        # Generate the pilot samples needed to estimate Var(Y_l).
        if self.sample_count < minimum_samples:
            number_of_pilot_samples = (
                minimum_samples - self.sample_count
            )

            # Do not exceed the sample limit.
            number_of_pilot_samples = min(
                number_of_pilot_samples,
                maximum_samples - self.sample_count,
            )

            self.add_samples(number_of_pilot_samples)

        # The pilot samples might already satisfy the target.
        if self.variance_of_mean <= target_variance:
            return True

        for iteration in range(
            1,
            maximum_iterations + 1,
        ):
            remaining_sample_budget = (
                maximum_samples - self.sample_count
            )

            if remaining_sample_budget <= 0:
                return False

            number_to_add = min(
                samples_per_iteration,
                remaining_sample_budget,
            )

            self.add_samples(number_to_add)

            if verbose:
                print(
                    f"Y_{self.level_index}: "
                    f"iteration={iteration}, "
                    f"N={self.sample_count}, "
                    f"V={self.sample_variance:.3e}, "
                    f"V/N={self.variance_of_mean:.3e}, "
                    f"target={target_variance:.3e}"
                )

            if self.variance_of_mean <= target_variance:
                return True

        return False
#`------------------------------------------------
#`------------------------------------------------
#`------------------------------------------------

@dataclass(frozen=True)
class MCLevelFactory:
    coarse_maxh: float
    relative_solver_tolerance: float
    maximum_vcycles: int
    pre_sweeps: int
    post_sweeps: int
    coarse_direct: bool
    coarse_sweeps: int
    dirichlet: str
    dirichlet_value: dict[str, float]
    order: int

    def __call__(
        self,
        level_index: int,
    ) -> MCLevel:
        """
        Construct a new independent MCLevel for Q_l.
        """
        return MCLevel.build(
            coarse_maxh=self.coarse_maxh,
            level_index=level_index,
            relative_solver_tolerance=(
                self.relative_solver_tolerance
            ),
            maximum_vcycles=self.maximum_vcycles,
            pre_sweeps=self.pre_sweeps,
            post_sweeps=self.post_sweeps,
            coarse_direct=self.coarse_direct,
            coarse_sweeps=self.coarse_sweeps,
            dirichlet=self.dirichlet,
            dirichlet_value=self.dirichlet_value,
            order=self.order,
        )


@dataclass
class MultilevelMonteCarlo:
    """
    Manage independent MLMC correction terms.
    """

    terms: tuple[MLMCTerm, ...]

    @classmethod
    def create(
        cls,
        *,
        number_of_levels: int,
        mc_level_factory: MCLevelFactory,
        coefficient_from_xi: Callable[
            [np.ndarray],
            ng.CoefficientFunction,
        ],
        number_of_modes: int,
        seed: int,
    ) -> "MultilevelMonteCarlo":
        """
        Construct all independent MLMC terms.

        ``mc_level_factory`` must construct a new MCLevel every time
        it is called. It must not return a cached MCLevel because
        different terms require independent mutable solver state.
        """
        if number_of_levels < 1:
            raise ValueError(
                "number_of_levels must be positive."
            )

        if number_of_modes < 1:
            raise ValueError(
                "number_of_modes must be positive."
            )

        seed_sequence = np.random.SeedSequence(
            seed
        )

        child_seeds = seed_sequence.spawn(
            number_of_levels
        )

        terms: list[MLMCTerm] = []

        for level_index in range(
            number_of_levels
        ):
            # A new private Q_l solver.
            upper_level = mc_level_factory(
                level_index
            )

            # A separate private Q_(l-1) solver.
            if level_index == 0:
                lower_level = None
            else:
                lower_level = mc_level_factory(
                    level_index - 1
                )

            rng = np.random.default_rng(
                child_seeds[level_index]
            )

            term = MLMCTerm(
                level_index=level_index,
                upper_level=upper_level,
                lower_level=lower_level,
                rng=rng,
                coefficient_from_xi=(
                    coefficient_from_xi
                ),
                number_of_modes=number_of_modes,
            )

            terms.append(term)

        return cls(
            terms=tuple(terms)
        )

    def add_samples(
        self,
        level_index: int,
        number_of_samples: int,
    ) -> None:
        """
        Generate additional samples for one MLMC term.

        Parameters
        ----------
        level_index:
            Index of the term to sample:

                level_index = 0  ->  Y_0 = Q_0

                level_index > 0  ->  Y_l = Q_l - Q_(l-1)

        number_of_samples:
            Number of new independent samples to generate. Existing
            samples and statistics are retained.
        """

        if not 0 <= level_index < len(self.terms):
            raise IndexError(
                f"level_index must be between 0 and "
                f"{len(self.terms) - 1}, but received "
                f"{level_index}."
            )

        if number_of_samples < 0:
            raise ValueError(
                "number_of_samples must be nonnegative."
            )

        term = self.terms[level_index]
        term.add_samples(number_of_samples)

    def run_fixed(
        self,
        sample_counts: tuple[int, ...],
    ) -> None:
        """
        Run a fixed number of additional samples for every MLMC term.

        ``sample_counts[l]`` is the number of new samples of Y_l.
        Existing samples are retained, so calling this method twice adds
        both requested batches rather than replacing the first batch.
        """
        if len(sample_counts) != len(self.terms):
            raise ValueError(
                f"Expected {len(self.terms)} sample counts, "
                f"but received {len(sample_counts)}."
            )

        if any(count < 0 for count in sample_counts):
            raise ValueError(
                "Every sample count must be nonnegative."
            )

        for level_index, number_of_samples in enumerate(
            sample_counts
        ):
            self.add_samples(
                level_index=level_index,
                number_of_samples=number_of_samples,
            )

    def run_target_variance(
        self,
        target_variance: float,
        *,
        minimum_samples: int = 10,
        maximum_samples_per_term: int = 100_000,
        samples_per_iteration: int = 1,
        verbose: bool = True,
    ) -> bool:
        """
        Run every MLMC term until the total estimator-variance target
        is reached.

        The total target is divided equally among the MLMC terms. If
        there are K terms, each term receives the target

            target_variance / K.

        Each term independently checks

            sample_variance(Y_l) / N_l
                <= target_variance / K.

        Returns
        -------
        bool:
            True if every term reached its assigned target and the total
            estimator variance is no greater than ``target_variance``.
            False if one or more terms reached a safety limit first.
        """
        if (
            not np.isfinite(target_variance)
            or target_variance <= 0.0
        ):
            raise ValueError(
                "target_variance must be finite and positive."
            )

        number_of_terms = len(self.terms)

        if number_of_terms == 0:
            raise RuntimeError(
                "There are no MLMC terms to sample."
            )

        term_variance_target = (
            target_variance / number_of_terms
        )

        term_results: list[bool] = []

        for term in self.terms:
            converged = (
                term.run_to_mean_variance_target(
                    target_variance=term_variance_target,
                    minimum_samples=minimum_samples,
                    maximum_samples=(
                        maximum_samples_per_term
                    ),
                    samples_per_iteration=(
                        samples_per_iteration
                    ),
                    verbose=verbose,
                )
            )

            term_results.append(converged)

        all_terms_converged = all(term_results)

        total_variance_converged = (
            all_terms_converged
            and self.estimator_variance <= target_variance
        )

        if verbose:
            print()
            print("MLMC sampling summary")

            for term, converged in zip(
                self.terms,
                term_results,
                strict=True,
            ):
                print(
                    f"Y_{term.level_index}: "
                    f"N={term.sample_count}, "
                    f"V={term.sample_variance:.3e}, "
                    f"V/N={term.variance_of_mean:.3e}, "
                    f"target={term_variance_target:.3e}, "
                    f"converged={converged}"
                )

            print()
            print(
                "MLMC estimator variance: "
                f"{self.estimator_variance:.3e}"
            )
            print(
                "Target estimator variance: "
                f"{target_variance:.3e}"
            )
            print(
                "Total variance converged: "
            f"{total_variance_converged}"
        )

        return total_variance_converged

    @property
    def estimate_qoi(self) -> float:
        """
        Return the current MLMC estimate of E[Q_L].

        The estimator is

            sum_l mean(Y_l),

        where Y_0 = Q_0 and Y_l = Q_l - Q_(l-1).
        """
        if any(
            term.sample_count == 0
            for term in self.terms
        ):
            missing_levels = [
                term.level_index
                for term in self.terms
                if term.sample_count == 0
            ]

            raise RuntimeError(
                "Every MLMC term requires at least one sample. "
                f"Missing samples for levels {missing_levels}."
            )

        return float(
            sum(
                term.mean
                for term in self.terms
            )
        )

    @property
    def estimator_variance(self) -> float:
        """
        Return the estimated sampling variance of the MLMC estimator.

        Since the MLMC terms use independent random-number streams,

            Var(Q_ML) = sum_l Var(Y_l) / N_l.
        """
        if any(
            term.sample_count < 2
            for term in self.terms
        ):
            insufficient_levels = [
                term.level_index
                for term in self.terms
                if term.sample_count < 2
            ]

            raise RuntimeError(
                "At least two samples per term are required to "
                "estimate sampling variance. Insufficient samples "
                f"for levels {insufficient_levels}."
            )

        return float(
            sum(
                term.variance_of_mean
                for term in self.terms
            )
        )

    @property   
    def standard_error(self) -> float:
        """
        Return the estimated sampling standard deviation of the MLMC estimate.

        This measures sampling uncertainty only. It does not include the
        remaining spatial-discretization bias.
        """
        return float(
            np.sqrt(
                self.estimator_variance
            )
        )




if __name__ == "__main__":
    SEED = 7

    correlation_length = 0.3
    num_modes_2d = 1000
    mean_log_conductivity = 0.0
    standard_deviation = 1.0
    variance = standard_deviation**2
    grid_size = 64
    n_levels = 5

    KL_x, KL_y = grid_size, grid_size
    X, Y, KL_points = cartesian_grid_2d(KL_x, KL_y)


    frequencies_1d, normalizations_1d, eigenvalues_1d, _ = get_1d_eigenpairs(
        num_modes=num_modes_2d,
        correlation_length=correlation_length,
    )

    unit_eigenvalues_2d, mode_indices_2d, evaluate_eigenfunctions_2d = leading_2d_eigenpairs(
        eigenvalues_1d=eigenvalues_1d,
        frequencies_1d=frequencies_1d,
        normalizations_1d=normalizations_1d,
        correlation_length=correlation_length,
        num_modes_2d=num_modes_2d,
        method="heap",
    )

    # ===============================================================
    # evaluate_log_conductivity returns log(kappa) on the KL grid.
    # ===============================================================
    evaluate_log_conductivity = make_2d_kl_evaluator(
        eigenvalues_2d=unit_eigenvalues_2d,
        eigenfunction_evaluator=evaluate_eigenfunctions_2d,
        mean_log_conductivity=mean_log_conductivity,
        variance=variance,
    )


    print("Done with KL expansion")
    # rngs = create_rngs(SEED, n_levels)

    coefficient_from_xi = make_coefficient_from_xi(
        X=X,
        Y=Y,
        evaluate_log_conductivity=evaluate_log_conductivity,
    )


    # Every call to this factory creates an independent MCLevel.
    mc_level_factory = MCLevelFactory(
        coarse_maxh=0.3,
        relative_solver_tolerance=1.0e-6,
        maximum_vcycles=1000,
        pre_sweeps=2,
        post_sweeps=2,
        coarse_direct=True,
        coarse_sweeps=20,
        dirichlet="left|right",
        dirichlet_value={
            "left": 1.0,
            "right": 0.0,
        },
        order=1,
    )

    mlmc = MultilevelMonteCarlo.create(
        number_of_levels=n_levels,
        mc_level_factory=mc_level_factory,
        coefficient_from_xi=coefficient_from_xi,
        number_of_modes=num_modes_2d,
        seed=SEED,
    )

    target_variance = 2.0e-4

    run_started = time.perf_counter()

    converged = mlmc.run_target_variance(
        target_variance=target_variance,
        minimum_samples=5,
        maximum_samples_per_term=15_000,
        samples_per_iteration=2,
        verbose=False,
    )

    total_wall_time_seconds = (
        time.perf_counter() - run_started
    )

    per_term_target = (
        target_variance / n_levels
    )

    sum_of_term_times_seconds = sum(
        term.total_sampling_time_seconds
        for term in mlmc.terms
    )

    print()
    print("MLMC variance-target test")
    print(
        f"Total target: {target_variance:.6e}"
    )
    print(
        f"Per-term target: {per_term_target:.6e}"
    )
    print()

    for term in mlmc.terms:
        print(
            f"Y_{term.level_index}: "
            f"N={term.sample_count:4d}, "
            f"V={term.sample_variance:.6e}, "
            f"V/N={term.variance_of_mean:.6e}, "
            f"passed="
            f"{term.variance_of_mean <= per_term_target}, "
            f"total time="
            f"{term.total_sampling_time_seconds:.3f}s, "
            f"time/sample="
            f"{term.mean_sample_time_seconds:.6f}s"
        )

    print()
    print(
        "Sum of measured Y_l sample times: "
        f"{sum_of_term_times_seconds:.3f}s"
    )
    print(
        "Complete run wall time: "
        f"{total_wall_time_seconds:.3f}s"
    )
    print()
    print(
        f"MLMC estimate: {mlmc.estimate_qoi:.8e}"
    )
    print(
        "Estimator variance: "
        f"{mlmc.estimator_variance:.8e}"
    )
    print(
        f"Standard error: {mlmc.standard_error:.8e}"
    )
    print(f"Converged: {converged}")

    # Check the stored samples and running statistics.
    for term in mlmc.terms:
        if (
            term.sample_count
            != len(term.sample_durations_seconds)
        ):
            raise RuntimeError(
                f"Y_{term.level_index} has inconsistent "
                "stored timing counts."
            )

        if term.sample_count != len(term.corrections):
            raise RuntimeError(
                f"Y_{term.level_index} has inconsistent "
                "stored sample counts."
            )

        if not np.isclose(
            term.mean,
            np.mean(term.corrections),
        ):
            raise RuntimeError(
                f"Y_{term.level_index} has an incorrect "
                "running mean."
            )

    if not converged:
        raise RuntimeError(
            "The MLMC test did not reach its target."
        )

    if mlmc.estimator_variance > target_variance:
        raise RuntimeError(
            "The reported estimator variance is larger "
            "than the target."
        )

#================================================================================
#================================================================================
#================================================================================
#
#  SCRATCH
#
#================================================================================
#================================================================================
#================================================================================

# @dataclass
# class MultilevelMonteCarlo:
#     """
#     Manage the levels and correction samples in an MLMC estimator.

#     The MLMC terms are

#         Y_0 = Q_0,

#         Y_l = Q_l - Q_(l-1),  l > 0.

#     Each correction uses the same random vector xi, and therefore the
#     same physical conductivity realization, for both discretizations.
#     Different MLMC terms use independent random-number streams.
#     """

#     levels: tuple[MCLevel, ...]

#     coefficient_from_xi: Callable[
#         [np.ndarray],
#         ng.CoefficientFunction,
#     ]

#     number_of_modes: int
#     rngs: tuple[np.random.Generator, ...]

#     statistics: tuple[RunningStatistics, ...] = field(
#         init=False
#     )

#     def __post_init__(self) -> None:
#         if len(self.levels) < 1:
#             raise ValueError(
#                 "At least one MC level is required."
#             )

#         if self.number_of_modes < 1:
#             raise ValueError(
#                 "number_of_modes must be positive."
#             )

#         if len(self.rngs) != len(self.levels):
#             raise ValueError(
#                 "Expected one random-number generator per MLMC term."
#             )

#         for expected_index, level in enumerate(self.levels):
#             if level.level_index != expected_index:
#                 raise ValueError(
#                     "MC levels must be ordered consecutively "
#                     "from level zero."
#                 )

#             expected_number_of_levels = expected_index + 1

#             if (
#                 level.hierarchy.nlevels
#                 != expected_number_of_levels
#             ):
#                 raise ValueError(
#                     f"MCLevel({expected_index}) should contain "
#                     f"{expected_number_of_levels} hierarchy levels, "
#                     f"but contains {level.hierarchy.nlevels}."
#                 )

#         self.statistics = tuple(
#             RunningStatistics()
#             for _ in self.levels
#         )

#     @classmethod
#     def create(
#         cls,
#         *,
#         levels: tuple[MCLevel, ...],
#         coefficient_from_xi: Callable[
#             [np.ndarray],
#             ng.CoefficientFunction,
#         ],
#         number_of_modes: int,
#         seed: int,
#     ) -> "MultilevelMonteCarlo":
#         """
#         Create one independent random-number stream per MLMC term.
#         """
#         seed_sequence = np.random.SeedSequence(seed)

#         child_seeds = seed_sequence.spawn(
#             len(levels)
#         )

#         rngs = tuple(
#             np.random.default_rng(child_seed)
#             for child_seed in child_seeds
#         )

#         return cls(
#             levels=levels,
#             coefficient_from_xi=coefficient_from_xi,
#             number_of_modes=number_of_modes,
#             rngs=rngs,
#         )

#     def draw_level_sample(
#         self,
#         level_index: int,
#     ) -> float:
#         """
#         Draw one sample of Y_l.

#         For level zero, return Q_0. For every higher level, solve
#         Q_l and Q_(l-1) with the same conductivity and return their
#         difference.
#         """
#         if not 0 <= level_index < len(self.levels):
#             raise IndexError(
#                 "level_index is outside the MLMC hierarchy."
#             )

#         rng = self.rngs[level_index]

#         xi = rng.standard_normal(
#             self.number_of_modes
#         )

#         # Construct one physical-space conductivity realization.
#         kappa = self.coefficient_from_xi(xi)

#         if level_index == 0:
#             return self.levels[0].solve_qoi(kappa)

#         # Both solves receive exactly the same VoxelCoefficient.
#         q_lower = self.levels[
#             level_index - 1
#         ].solve_qoi(kappa)

#         q_upper = self.levels[
#             level_index
#         ].solve_qoi(kappa)

#         return q_upper - q_lower

#     def add_samples(
#         self,
#         level_index: int,
#         number_of_samples: int,
#     ) -> None:
#         """
#         Generate additional samples for one MLMC term.
#         """
#         if number_of_samples < 0:
#             raise ValueError(
#                 "number_of_samples must be nonnegative."
#             )

#         statistics = self.statistics[level_index]

#         for _ in range(number_of_samples):
#             sample = self.draw_level_sample(
#                 level_index
#             )
#             statistics.update(sample)

#     def run_fixed(
#         self,
#         sample_counts: tuple[int, ...],
#     ) -> None:
#         """
#         Run a specified number of samples for every MLMC term.
#         """
#         if len(sample_counts) != len(self.levels):
#             raise ValueError(
#                 "Expected one sample count per MLMC term."
#             )

#         for level_index, number_of_samples in enumerate(
#             sample_counts
#         ):
#             self.add_samples(
#                 level_index,
#                 number_of_samples,
#             )

#     @property
#     def estimate(self) -> float:
#         """
#         Return the MLMC estimate sum_l E[Y_l].
#         """
#         if any(
#             statistics.count == 0
#             for statistics in self.statistics
#         ):
#             raise RuntimeError(
#                 "Every MLMC term needs at least one sample."
#             )

#         return sum(
#             statistics.mean
#             for statistics in self.statistics
#         )

#     @property
#     def estimator_variance(self) -> float:
#         """
#         Return sum_l Var(Y_l) / N_l.
#         """
#         return sum(
#             statistics.variance_of_mean
#             for statistics in self.statistics
#         )

#     @property
#     def standard_error(self) -> float:
#         """
#         Estimated sampling standard deviation of the MLMC estimator.
#         """
#         return np.sqrt(
#             self.estimator_variance
#         )

#     def print_summary(self) -> None:
#         print(
#             f"{'term':<18}"
#             f"{'N':>10}"
#             f"{'mean':>16}"
#             f"{'variance':>16}"
#         )
#         print("-" * 60)

#         for level_index, statistics in enumerate(
#             self.statistics
#         ):
#             if level_index == 0:
#                 name = "Q_0"
#             else:
#                 name = (
#                     f"Q_{level_index}"
#                     f"-Q_{level_index - 1}"
#                 )

#             print(
#                 f"{name:<18}"
#                 f"{statistics.count:>10d}"
#                 f"{statistics.mean:>16.6e}"
#                 f"{statistics.sample_variance:>16.6e}"
#             )

#         print()
#         print(
#             f"MLMC estimate: {self.estimate:.8e}"
#         )
#         print(
#             f"sampling standard error: "
#             f"{self.standard_error:.3e}"
#         )



#     def main(*, show_plots: bool = True, plot_dir: Path = DEFAULT_PLOT_DIR) -> None:
#         # ------------------------------------------------------------------
#         # 1. Construct one discrete KL realization of log(kappa).
#         # ------------------------------------------------------------------
#         nx = ny = 16
#         X, Y, points = cartesian_grid_2d(nx, ny)

#         covariance = exponential_covariance(
#             points,
#             sigma=1.,
#             correlation_length=0.30,
#         )
#         eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=100)
#         log_kappa_values, xi = sample_discrete_kl(
#             mean=0.0,
#             eigenvalues=eigenvalues,
#             eigenvectors=eigenvectors,
#             shape=(ny, nx),
#             rng=7,
#         )

#         # exp(log(kappa)) makes the diffusion coefficient strictly positive.
#         kappa_values = lognormal_transform(log_kappa_values)
#         kappa = voxel_coefficient_2d(kappa_values, linear=True)

#         # ------------------------------------------------------------------
#         # 2. Define -div(kappa grad(u)) = 1 with u = 0 on the boundary.
#         #    Both levels close over and assemble with this same kappa object.
#         # ------------------------------------------------------------------
#         def diffusion_form(a, u, v):
#             a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx

#         def load_form(f, u, v):
#             f += v * ng.dx

#         form_setup = build_form_setup(bilinear=diffusion_form, linear=load_form)

#         # n_refines=1 means exactly two levels: the initial mesh and one refinement.
#         coarse_mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.35))
#         hierarchy = build_hierarchy(
#             coarse_mesh,
#             form_setup,
#             n_refines=1,
#             order=1,
#             dirichlet="left|right|top|bottom",
#             dirichlet_value=0.0,
#             verbose=True,
#         )
#         assert hierarchy.nlevels == 2

#         # The same physical point has the same coefficient value on both meshes.
#         probe = (0.37, 0.42)
#         probe_values = np.array(
#             [float(kappa(level.mesh(*probe))) for level in hierarchy.levels]
#         )
#         np.testing.assert_allclose(probe_values, probe_values[0])

#         # ------------------------------------------------------------------
#         # 3. Solve the coarse FE problem and prolong it as the fine initial guess.
#         #    P transfers the FE solution only; it does not act on the KL field.
#         # ------------------------------------------------------------------
#         coarse = hierarchy.coarsest
#         fine = hierarchy.finest

#         coarse_solution = ng.GridFunction(coarse.fes)
#         coarse_solution.vec.data = coarse.a.mat.Inverse(coarse.fes.FreeDofs()) * coarse.f.vec
#         coarse.enforce_dirichlet(coarse_solution.vec)

#         initial_guess = ng.GridFunction(fine.fes)
#         initial_guess.vec.data = fine.P * coarse_solution.vec
#         fine.enforce_dirichlet(initial_guess.vec)
#         fine.gfu.vec.data = initial_guess.vec

#         # ------------------------------------------------------------------
#         # 4. Continue from that initial guess using repeated two-level V-cycles.
#         # ------------------------------------------------------------------
#         relative_tolerance = 1.0e-8
#         initial_residual = hierarchy.finest.residual_norm()
#         solver = MultigridSolver(
#             hierarchy,
#             VCycleConfig(pre_sweeps=2, post_sweeps=2, coarse_direct=True),
#         )
#         history, _ = solver.solve(
#             max_cycles=12,
#             tol=relative_tolerance,
#             norms=("l2",),
#             verbose=True,
#         )
#         residuals = np.concatenate(([initial_residual], history["l2"]))

#         # ------------------------------------------------------------------
#         # 5. Make and save the field, initial-guess, and diagnostic plots.
#         # ------------------------------------------------------------------
#         field_figure = make_field_figure(
#             X,
#             Y,
#             log_kappa_values,
#             kappa_values,
#             hierarchy,
#         )
#         diagnostics_figure = make_diagnostics_figure(
#             eigenvalues,
#             float(np.trace(covariance)),
#             residuals,
#             relative_tolerance,
#             hierarchy.finest,
#         )
#         initial_guess_figure = make_initial_guess_figure(
#             coarse,
#             fine,
#             coarse_solution,
#             initial_guess,
#         )

#         plot_dir.mkdir(parents=True, exist_ok=True)
#         field_path = plot_dir / "kl_two_level_fields.png"
#         diagnostics_path = plot_dir / "kl_two_level_diagnostics.png"
#         initial_guess_path = plot_dir / "kl_two_level_initial_guess.png"
#         field_figure.savefig(field_path, dpi=180)
#         diagnostics_figure.savefig(diagnostics_path, dpi=180)
#         initial_guess_figure.savefig(initial_guess_path, dpi=180)

#         retained_variance = eigenvalues.sum() / np.trace(covariance)
#         print(f"KL coefficients xi: {np.array2string(xi, precision=3)}")
#         print(f"retained discrete variance: {retained_variance:.1%}")
#         print(f"kappa range on KL grid: [{kappa_values.min():.3f}, {kappa_values.max():.3f}]")
#         print(f"kappa{probe} on coarse/fine meshes: {probe_values}")
#         print(f"final residual: {history['l2'][-1]:.3e}")
#         print(f"saved field plot: {field_path}")
#         print(f"saved diagnostics plot: {diagnostics_path}")
#         print(f"saved initial-guess plot: {initial_guess_path}")

#         if show_plots:
#             plt.show()
#         else:
#             plt.close("all")


# @dataclass
# class ReusableMultigridProblem:
#     """Nested hierarchy whose conductivity fields can be updated in place."""

#     hierarchy: MultigridHierarchy
#     conductivity_fields: tuple[ng.GridFunction, ...]
#     solver_config: VCycleConfig
#     relative_solver_tolerance: float
#     maximum_vcycles: int
#     allow_direct_fallback: bool
#     direct_fallback_counts: np.ndarray

#     @classmethod
#     def build(
#         cls,
#         *,
#         coarse_maxh: float,
#         number_of_levels: int,
#         order: int,
#         relative_solver_tolerance: float,
#         maximum_vcycles: int,
#         pre_sweeps: int,
#         post_sweeps: int,
#         allow_direct_fallback: bool,
#     ) -> "ReusableMultigridProblem":
#         """Build nested geometry, FE spaces, forms, and grid transfers once."""
#         if number_of_levels < 2:
#             raise ValueError("number_of_levels must be at least 2.")
#         if coarse_maxh <= 0.0:
#             raise ValueError("coarse_maxh must be positive.")
#         if order < 1:
#             raise ValueError("order must be positive.")
#         if relative_solver_tolerance <= 0.0:
#             raise ValueError("relative_solver_tolerance must be positive.")
#         if maximum_vcycles < 1:
#             raise ValueError("maximum_vcycles must be positive.")

#         conductivity_fields: list[ng.GridFunction] = []

#         def form_setup(fes):
#             coefficient_space = ng.H1(fes.mesh, order=1)
#             conductivity = ng.GridFunction(coefficient_space)
#             conductivity.Set(1.0)
#             conductivity_fields.append(conductivity)

#             u, v = fes.TnT()
#             a = ng.BilinearForm(fes, symmetric=True)
#             a += (
#                 conductivity
#                 * ng.InnerProduct(ng.grad(u), ng.grad(v))
#                 * ng.dx
#             )
#             f = ng.LinearForm(fes)
#             return a, f

#         coarse_mesh = ng.Mesh(
#             unit_square.GenerateMesh(maxh=coarse_maxh)
#         )
#         hierarchy = build_hierarchy(
#             coarse_mesh,
#             form_setup,
#             n_refines=number_of_levels - 1,
#             order=order,
#             dirichlet="left|right",
#             dirichlet_value={"left": -1.0, "right": 0.0},
#             verbose=False,
#         )

#         if len(conductivity_fields) != hierarchy.nlevels:
#             raise RuntimeError(
#                 "Expected one conductivity field per multigrid level."
#             )

#         return cls(
#             hierarchy=hierarchy,
#             conductivity_fields=tuple(conductivity_fields),
#             solver_config=VCycleConfig(
#                 pre_sweeps=pre_sweeps,
#                 post_sweeps=post_sweeps,
#                 coarse_direct=True,
#             ),
#             relative_solver_tolerance=relative_solver_tolerance,
#             maximum_vcycles=maximum_vcycles,
#             allow_direct_fallback=allow_direct_fallback,
#             direct_fallback_counts=np.zeros(
#                 number_of_levels,
#                 dtype=int,
#             ),
#         )

#     def update_conductivity(self, kappa, finest_level: int) -> None:
#         """Interpolate one sample and reassemble levels 0 through finest."""
#         if not 0 <= finest_level < self.hierarchy.nlevels:
#             raise IndexError("finest_level is outside the hierarchy.")

#         for level_index in range(finest_level + 1):
#             conductivity = self.conductivity_fields[level_index]
#             level = self.hierarchy.levels[level_index]

#             conductivity.Set(kappa)
#             level.a.Assemble()
#             level.refresh()

#     def reset_solution(
#         self,
#         level_index: int,
#         *,
#         use_coarse_initial_guess: bool,
#     ) -> None:
#         """Initialize one level and impose its nonzero boundary data."""
#         level = self.hierarchy.levels[level_index]

#         if use_coarse_initial_guess:
#             if level_index == 0 or level.P is None:
#                 raise ValueError(
#                     "A coarse initial guess requires level_index >= 1."
#                 )
#             coarse_solution = self.hierarchy.levels[level_index - 1].gfu
#             level.gfu.vec.data = level.P * coarse_solution.vec
#         else:
#             level.gfu.vec.FV().NumPy()[:] = 0.0

#         level.enforce_dirichlet(level.gfu.vec)

#     def solve_level(
#         self,
#         level_index: int,
#         *,
#         use_coarse_initial_guess: bool = False,
#     ) -> ng.GridFunction:
#         """Solve level_index directly at level zero or by repeated V-cycles."""
#         self.reset_solution(
#             level_index,
#             use_coarse_initial_guess=use_coarse_initial_guess,
#         )
#         level = self.hierarchy.levels[level_index]

#         if level_index == 0:
#             residual = level.residual(level.f.vec, level.gfu.vec)
#             correction = level.gfu.vec.CreateVector()
#             correction.data = (
#                 level.a.mat.Inverse(level.fes.FreeDofs())
#                 * residual
#             )
#             level.gfu.vec.data += correction
#             level.enforce_dirichlet(level.gfu.vec)
#             return level.gfu

#         subhierarchy = MultigridHierarchy(
#             self.hierarchy.levels[: level_index + 1]
#         )
#         solver = MultigridSolver(
#             subhierarchy,
#             self.solver_config,
#         )
#         initial_residual = level.residual_norm()
#         solver.solve(
#             max_cycles=self.maximum_vcycles,
#             tol=self.relative_solver_tolerance,
#             norms=("l2",),
#             stop_norm="l2",
#             verbose=False,
#         )
#         final_residual = level.residual_norm()

#         if (
#             initial_residual > 0.0
#             and final_residual
#             > self.relative_solver_tolerance * initial_residual
#         ):
#             if not self.allow_direct_fallback:
#                 raise RuntimeError(
#                     f"Level {level_index} V-cycle solve did not reach its "
#                     f"relative tolerance after {self.maximum_vcycles} cycles: "
#                     f"initial residual={initial_residual:.3e}, "
#                     f"final residual={final_residual:.3e}. Increase "
#                     "--maximum-vcycles, relax --linear-tolerance, or allow "
#                     "the default direct fallback."
#                 )

#             # Complete the unresolved residual exactly on the free DOFs.
#             # This preserves sample correctness while recording that the
#             # V-cycle was not robust for this conductivity realization.
#             residual = level.residual(level.f.vec, level.gfu.vec)
#             correction = level.gfu.vec.CreateVector()
#             correction.data = (
#                 level.a.mat.Inverse(level.fes.FreeDofs())
#                 * residual
#             )
#             level.gfu.vec.data += correction
#             level.enforce_dirichlet(level.gfu.vec)
#             self.direct_fallback_counts[level_index] += 1

#         return subhierarchy.finest.gfu

#     def evaluate_qoi(self, level_index: int) -> float:
#         """Evaluate the flux QoI using the coefficient assembled on this level."""
#         level = self.hierarchy.levels[level_index]
#         conductivity = self.conductivity_fields[level_index]
#         return quantity_of_interest(
#             level.gfu,
#             level.mesh,
#             conductivity,
#         )

#     def solve_qoi(
#         self,
#         level_index: int,
#         *,
#         use_coarse_initial_guess: bool = False,
#     ) -> float:
#         """Solve one level and evaluate its QoI."""
#         self.solve_level(
#             level_index,
#             use_coarse_initial_guess=use_coarse_initial_guess,
#         )
#         return self.evaluate_qoi(level_index)


# def run_comparison(
#     *,
#     tolerance: float = 2.0e-2,
#     coarse_maxh: float = 0.30,
#     maximum_levels: int = 4,
#     order: int = 1,
#     coefficient_grid_size: int = 32,
#     num_modes_2d: int = 100,
#     correlation_length: float = 0.30,
#     standard_deviation: float = 1.0,
#     mean_log_conductivity: float = 0.0,
#     weak_rate_alpha: float = 1.0,
#     refinement_ratio: float = 2.0,
#     pilot_samples: int = 20,
#     maximum_samples_per_term: int = 100_000,
#     maximum_allocation_iterations: int = 10,
#     relative_solver_tolerance: float = 1.0e-10,
#     maximum_vcycles: int = 100,
#     pre_sweeps: int = 2,
#     post_sweeps: int = 2,
#     allow_direct_fallback: bool = True,
#     seed: int = 7,
#     verbose: bool = True,
# ) -> AdaptiveMLMCResult:
#     """Run adaptive MLMC and fine MC using the reusable MG hierarchy."""
#     if tolerance <= 0.0:
#         raise ValueError("tolerance must be positive.")
#     if maximum_levels < 2:
#         raise ValueError("maximum_levels must be at least 2.")
#     if coefficient_grid_size < 2:
#         raise ValueError("coefficient_grid_size must be at least 2.")
#     if num_modes_2d < 1:
#         raise ValueError("num_modes_2d must be positive.")
#     if correlation_length <= 0.0:
#         raise ValueError("correlation_length must be positive.")
#     if standard_deviation <= 0.0:
#         raise ValueError("standard_deviation must be positive.")
#     if weak_rate_alpha <= 0.0:
#         raise ValueError("weak_rate_alpha must be positive.")
#     if refinement_ratio <= 1.0:
#         raise ValueError("refinement_ratio must be greater than one.")
#     if pilot_samples < 2:
#         raise ValueError("pilot_samples must be at least 2.")
#     if maximum_samples_per_term < pilot_samples:
#         raise ValueError(
#             "maximum_samples_per_term must be at least pilot_samples."
#         )

#     problem = ReusableMultigridProblem.build(
#         coarse_maxh=coarse_maxh,
#         number_of_levels=maximum_levels,
#         order=order,
#         relative_solver_tolerance=relative_solver_tolerance,
#         maximum_vcycles=maximum_vcycles,
#         pre_sweeps=pre_sweeps,
#         post_sweeps=post_sweeps,
#         allow_direct_fallback=allow_direct_fallback,
#     )

#     frequencies_1d, normalizations_1d, eigenvalues_1d, _ = get_1d_eigenpairs(
#         num_modes=num_modes_2d,
#         correlation_length=correlation_length,
#     )
#     (
#         unit_eigenvalues_2d,
#         _,
#         evaluate_eigenfunctions_2d,
#     ) = leading_2d_eigenpairs(
#         eigenvalues_1d=eigenvalues_1d,
#         frequencies_1d=frequencies_1d,
#         normalizations_1d=normalizations_1d,
#         correlation_length=correlation_length,
#         num_modes_2d=num_modes_2d,
#         method="heap",
#     )
#     evaluate_log_conductivity = make_2d_kl_evaluator(
#         eigenvalues_2d=unit_eigenvalues_2d,
#         eigenfunction_evaluator=evaluate_eigenfunctions_2d,
#         mean_log_conductivity=mean_log_conductivity,
#         variance=standard_deviation**2,
#     )

#     X, Y, _ = cartesian_grid_2d(
#         coefficient_grid_size,
#         coefficient_grid_size,
#     )

#     def coefficient_from_xi(xi: np.ndarray):
#         log_kappa = evaluate_log_conductivity(X, Y, xi)
#         return voxel_coefficient_2d(
#             lognormal_transform(log_kappa),
#             linear=True,
#         )

#     seed_sequence = np.random.SeedSequence(seed)
#     child_seeds = seed_sequence.spawn(maximum_levels + 1)
#     term_rngs = tuple(
#         np.random.default_rng(child_seed)
#         for child_seed in child_seeds[:-1]
#     )
#     fine_mc_rng = np.random.default_rng(child_seeds[-1])

#     def draw_level_zero() -> float:
#         xi = term_rngs[0].standard_normal(num_modes_2d)
#         kappa = coefficient_from_xi(xi)
#         problem.update_conductivity(kappa, finest_level=0)
#         return problem.solve_qoi(0)

#     def make_correction_sampler(level_index: int) -> Callable[[], float]:
#         rng = term_rngs[level_index]

#         def draw() -> float:
#             xi = rng.standard_normal(num_modes_2d)
#             kappa = coefficient_from_xi(xi)
#             problem.update_conductivity(
#                 kappa,
#                 finest_level=level_index,
#             )
#             q_lower = problem.solve_qoi(level_index - 1)
#             q_upper = problem.solve_qoi(
#                 level_index,
#                 use_coarse_initial_guess=True,
#             )
#             return q_upper - q_lower

#         return draw

#     all_terms = [TermState("level 0: Q_0", draw_level_zero)]
#     all_terms.extend(
#         TermState(
#             f"level {level}: Q_{level}-Q_{level - 1}",
#             make_correction_sampler(level),
#         )
#         for level in range(1, maximum_levels)
#     )

#     bias_target = tolerance / math.sqrt(2.0)
#     active_finest_level = 1
#     sampling_converged = False
#     bias_converged = False
#     bias_estimate = math.inf

#     if verbose:
#         print("Reusable nested multigrid hierarchy")
#         problem.hierarchy.info()
#         print(f"linear solver tolerance: {relative_solver_tolerance:.3e}")

#     while True:
#         active_terms = all_terms[: active_finest_level + 1]
#         if verbose:
#             print()
#             print(f"active finest MLMC level: {active_finest_level}")

#         sampling_converged = allocate_until_sampling_converged(
#             terms=active_terms,
#             tolerance=tolerance,
#             pilot_samples=pilot_samples,
#             maximum_samples_per_term=maximum_samples_per_term,
#             maximum_allocation_iterations=maximum_allocation_iterations,
#             verbose=verbose,
#         )

#         nominal_coarse_h = coarse_maxh / (
#             refinement_ratio ** (active_finest_level - 1)
#         )
#         nominal_fine_h = coarse_maxh / (
#             refinement_ratio**active_finest_level
#         )
#         bias_estimate = estimate_remaining_bias(
#             finest_correction_mean=active_terms[-1].mean,
#             coarse_maxh=nominal_coarse_h,
#             fine_maxh=nominal_fine_h,
#             weak_rate_alpha=weak_rate_alpha,
#         )
#         bias_converged = bias_estimate <= bias_target

#         if verbose:
#             print(
#                 f"estimated bias={bias_estimate:.3e}, "
#                 f"target={bias_target:.3e}"
#             )

#         if sampling_converged and bias_converged:
#             break
#         if not sampling_converged:
#             break
#         if active_finest_level + 1 >= maximum_levels:
#             break

#         active_finest_level += 1

#     active_terms = all_terms[: active_finest_level + 1]

#     def draw_fine_mc() -> float:
#         xi = fine_mc_rng.standard_normal(num_modes_2d)
#         kappa = coefficient_from_xi(xi)
#         problem.update_conductivity(
#             kappa,
#             finest_level=active_finest_level,
#         )
#         return problem.solve_qoi(active_finest_level)

#     fine_mc = TermState(
#         f"fine MC Q_{active_finest_level}",
#         draw_fine_mc,
#     )
#     fine_mc.add_samples(pilot_samples)

#     fine_variance_target = tolerance**2 / 2.0
#     for _ in range(maximum_allocation_iterations):
#         target = max(
#             pilot_samples,
#             math.ceil(2.0 * fine_mc.variance / tolerance**2),
#         )
#         target = min(target, maximum_samples_per_term)
#         if fine_mc.count < target:
#             fine_mc.add_samples(target - fine_mc.count)
#             continue
#         if fine_mc.variance / fine_mc.count <= fine_variance_target:
#             break
#         if fine_mc.count >= maximum_samples_per_term:
#             break

#     result = AdaptiveMLMCResult(
#         terms=tuple(active_terms),
#         fine_mc=fine_mc,
#         finest_level=active_finest_level,
#         sampling_variance=estimated_sampling_variance(active_terms),
#         sampling_variance_target=tolerance**2 / 2.0,
#         bias_estimate=bias_estimate,
#         bias_target=bias_target,
#         sampling_converged=sampling_converged,
#         bias_converged=bias_converged,
#     )
#     print_summary(result)
#     print()
#     print("Direct fallbacks after stalled V-cycles:")
#     for level_index, count in enumerate(problem.direct_fallback_counts):
#         if level_index <= active_finest_level:
#             print(f"  level {level_index}: {count}")
#     return result


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--tolerance", type=float, default=2.0e-2)
#     parser.add_argument("--coarse-maxh", type=float, default=0.30)
#     parser.add_argument("--maximum-levels", type=int, default=4)
#     parser.add_argument("--order", type=int, default=1)
#     parser.add_argument("--coefficient-grid-size", type=int, default=32)
#     parser.add_argument("--num-modes", type=int, default=100, dest="num_modes_2d")
#     parser.add_argument("--ell", type=float, default=0.30, dest="correlation_length")
#     parser.add_argument("--sigma", type=float, default=1.0, dest="standard_deviation")
#     parser.add_argument("--mean-log-kappa", type=float, default=0.0)
#     parser.add_argument("--weak-rate-alpha", type=float, default=1.0)
#     parser.add_argument("--refinement-ratio", type=float, default=2.0)
#     parser.add_argument("--pilot-samples", type=int, default=20)
#     parser.add_argument("--maximum-samples-per-term", type=int, default=100_000)
#     parser.add_argument("--maximum-allocation-iterations", type=int, default=10)
#     parser.add_argument("--linear-tolerance", type=float, default=1.0e-10)
#     parser.add_argument("--maximum-vcycles", type=int, default=500)
#     parser.add_argument("--pre-sweeps", type=int, default=2)
#     parser.add_argument("--post-sweeps", type=int, default=2)
#     parser.add_argument(
#         "--no-direct-fallback",
#         action="store_true",
#         help="raise an error instead of finishing a stalled V-cycle solve directly",
#     )
#     parser.add_argument("--seed", type=int, default=7)
#     parser.add_argument("--quiet", action="store_true")
#     return parser.parse_args()


# if __name__ == "__main__":
#     arguments = parse_args()
#     run_comparison(
#         tolerance=arguments.tolerance,
#         coarse_maxh=arguments.coarse_maxh,
#         maximum_levels=arguments.maximum_levels,
#         order=arguments.order,
#         coefficient_grid_size=arguments.coefficient_grid_size,
#         num_modes_2d=arguments.num_modes_2d,
#         correlation_length=arguments.correlation_length,
#         standard_deviation=arguments.standard_deviation,
#         mean_log_conductivity=arguments.mean_log_kappa,
#         weak_rate_alpha=arguments.weak_rate_alpha,
#         refinement_ratio=arguments.refinement_ratio,
#         pilot_samples=arguments.pilot_samples,
#         maximum_samples_per_term=arguments.maximum_samples_per_term,
#         maximum_allocation_iterations=arguments.maximum_allocation_iterations,
#         relative_solver_tolerance=arguments.linear_tolerance,
#         maximum_vcycles=arguments.maximum_vcycles,
#         pre_sweeps=arguments.pre_sweeps,
#         post_sweeps=arguments.post_sweeps,
#         allow_direct_fallback=not arguments.no_direct_fallback,
#         seed=arguments.seed,
#         verbose=not arguments.quiet,
#     )
