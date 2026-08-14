"""Prototype hierarchical SPDE sampling inside the MLMC Darcy workflow.

This file is deliberately separate from ``multilevel_test.py`` so that the
SPDE sampler can be validated before it is moved into ``src`` or exposed as an
alternative to the KL sampler.

The Gaussian log-conductivity is sampled with the mixed finite-element SPDE

    (u, v) + (div(v), theta) = 0,

    (div(u), q) - kappa_s**2 (theta, q) = -g (W, q),

using lowest-order ``HDiv`` for ``u`` and piecewise-constant ``L2`` for
``theta``.  The white-noise load is

    f_h = W_h**(1/2) xi_h,    xi_h ~ N(0, I),

where ``W_h`` is the diagonal mass matrix of the L2 space.

For an MLMC correction ``Y_l = Q_l - Q_(l-1)``, the coarse stochastic load is
not sampled independently.  It is obtained from the fine load by

    f_(l-1) = P_theta.T f_l.

For piecewise constants on uniformly refined nested meshes, this restriction
is exactly the sum of the fine child-element loads inside every coarse
element.  This is the geometric-hierarchy version of the coupling described
by Osborn, Vassilevski, and Villa (2017).

The current prototype uses a reusable direct inverse for the mixed SPDE on
each level.  This is appropriate for validation on small meshes, but it is not
the scalable H(div) solver from the paper.  The solved mesh-dependent SPDE
field is evaluated on a Cartesian grid and wrapped as a coordinate-based
``VoxelCoefficient`` so it can be passed safely to the independently built
Darcy ``MCLevel`` objects.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import (  # noqa: E402
    cartesian_grid_2d,
    voxel_coefficient_2d,
)
from examples.KLMC.multilevel.multilevel_test import (  # noqa: E402
    MCLevel,
    MCLevelFactory,
)


@dataclass(frozen=True)
class SPDEConfig:
    """Parameters of the Gaussian reaction-diffusion SPDE.

    Notes
    -----
    ``inverse_correlation_scale`` is the parameter denoted by ``kappa`` in
    the SPDE paper.  The longer name avoids confusion with the hydraulic
    conductivity used in the Darcy equation.

    In two dimensions, the ordinary reaction-diffusion SPDE corresponds to a
    Matern field with smoothness ``nu=1``.  For unit marginal variance on an
    unbounded domain, its white-noise scaling is

        g = 2 * sqrt(pi) * inverse_correlation_scale.

    Artificial boundary conditions change the marginal variance near the
    boundary.  Domain embedding can be added after this prototype is
    validated.
    """

    inverse_correlation_scale: float
    mean_log_conductivity: float = 0.0
    standard_deviation: float = 1.0
    white_noise_scale: float | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.inverse_correlation_scale)
            or self.inverse_correlation_scale <= 0.0
        ):
            raise ValueError(
                "inverse_correlation_scale must be finite and positive."
            )

        if not np.isfinite(self.mean_log_conductivity):
            raise ValueError(
                "mean_log_conductivity must be finite."
            )

        if (
            not np.isfinite(self.standard_deviation)
            or self.standard_deviation < 0.0
        ):
            raise ValueError(
                "standard_deviation must be finite and nonnegative."
            )

        if self.white_noise_scale is not None and (
            not np.isfinite(self.white_noise_scale)
            or self.white_noise_scale <= 0.0
        ):
            raise ValueError(
                "white_noise_scale must be finite and positive."
            )

    @property
    def forcing_scale(self) -> float:
        """Return ``g`` in the stochastic right-hand side ``g W``."""
        if self.white_noise_scale is not None:
            return float(self.white_noise_scale)

        return float(
            2.0
            * np.sqrt(np.pi)
            * self.inverse_correlation_scale
        )


@dataclass(frozen=True)
class CoupledConductivitySample:
    """Fine and coarse conductivities for one MLMC correction sample."""

    upper: ng.CoefficientFunction
    lower: ng.CoefficientFunction | None

    upper_log_values: np.ndarray
    lower_log_values: np.ndarray | None


@dataclass
class SPDELevel:
    """One mixed finite-element discretization of the sampling SPDE."""

    level_index: int
    mesh: ng.Mesh
    flux_space: ng.FESpace
    field_space: ng.FESpace
    mixed_space: ng.FESpace
    system: ng.BilinearForm
    inverse: ng.BaseMatrix
    field_mass_diagonal: np.ndarray
    solution: ng.GridFunction
    config: SPDEConfig

    @classmethod
    def build(
        cls,
        *,
        level_index: int,
        mesh: ng.Mesh,
        config: SPDEConfig,
    ) -> "SPDELevel":
        """Assemble one reusable mixed SPDE system and direct inverse."""
        flux_space = ng.HDiv(
            mesh,
            order=1,
            dirichlet=".*",
        )
        field_space = ng.L2(
            mesh,
            order=0,
        )
        mixed_space = ng.FESpace(
            [flux_space, field_space]
        )

        (u, theta), (v, q) = mixed_space.TnT()
        inverse_scale = config.inverse_correlation_scale

        system = ng.BilinearForm(
            mixed_space,
            symmetric=True,
        )
        system += (
            ng.InnerProduct(u, v)
            + ng.div(v) * theta
            + ng.div(u) * q
            - inverse_scale**2 * theta * q
        ) * ng.dx
        system.Assemble()

        field_mass = ng.BilinearForm(
            field_space,
            symmetric=True,
        )
        field_trial, field_test = field_space.TnT()
        field_mass += field_trial * field_test * ng.dx
        field_mass.Assemble()

        # With L2(order=0), the mass matrix has exactly one positive diagonal
        # entry per element and no off-diagonal entries.
        field_mass_diagonal = (
            field_mass.mat.AsVector()
            .FV()
            .NumPy()
            .copy()
        )

        if field_mass_diagonal.shape != (field_space.ndof,):
            raise RuntimeError(
                "Expected a diagonal L2(order=0) mass matrix with one "
                "entry per field degree of freedom."
            )

        if (
            not np.all(np.isfinite(field_mass_diagonal))
            or np.any(field_mass_diagonal <= 0.0)
        ):
            raise RuntimeError(
                "The SPDE field mass diagonal must be finite and positive."
            )

        inverse = system.mat.Inverse(
            mixed_space.FreeDofs(),
            inverse="sparsecholesky",
        )

        return cls(
            level_index=level_index,
            mesh=mesh,
            flux_space=flux_space,
            field_space=field_space,
            mixed_space=mixed_space,
            system=system,
            inverse=inverse,
            field_mass_diagonal=field_mass_diagonal,
            solution=ng.GridFunction(mixed_space),
            config=config,
        )

    def draw_white_noise_load(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw the discrete load ``W_h**(1/2) xi``, ``xi ~ N(0,I)``."""
        xi = rng.standard_normal(
            self.field_space.ndof
        )

        return (
            np.sqrt(self.field_mass_diagonal)
            * xi
        )

    def solve(
        self,
        white_noise_load: np.ndarray,
    ) -> ng.GridFunction:
        """Solve the mixed SPDE for an already coupled stochastic load."""
        load = np.asarray(
            white_noise_load,
            dtype=float,
        )

        if load.shape != (self.field_space.ndof,):
            raise ValueError(
                "white_noise_load must have shape "
                f"({self.field_space.ndof},), but received {load.shape}."
            )

        if not np.all(np.isfinite(load)):
            raise ValueError(
                "white_noise_load must contain only finite values."
            )

        right_hand_side = self.solution.vec.CreateVector()
        right_hand_side[:] = 0.0

        field_range = self.mixed_space.Range(1)
        right_hand_side_values = (
            right_hand_side.FV().NumPy()
        )
        right_hand_side_values[
            field_range.start:field_range.stop
        ] = -self.config.forcing_scale * load

        self.solution.vec[:] = 0.0
        self.solution.vec.data = (
            self.inverse * right_hand_side
        )

        theta_values = (
            self.solution.components[1]
            .vec.FV()
            .NumPy()
        )

        if not np.all(np.isfinite(theta_values)):
            raise RuntimeError(
                f"SPDE level {self.level_index} produced a nonfinite field."
            )

        return self.solution

    @property
    def log_field(self) -> ng.GridFunction:
        """Return the piecewise-constant Gaussian field ``theta_h``."""
        return self.solution.components[1]


def element_parent_indices(
    *,
    coarse_mesh: ng.Mesh,
    fine_mesh: ng.Mesh,
) -> np.ndarray:
    """Map every fine element to its containing coarse element.

    For nested uniformly refined meshes and piecewise-constant fields, this
    array is the complete information needed to apply ``P_theta.T`` to an L2
    load vector.
    """
    parents = np.empty(
        fine_mesh.ne,
        dtype=int,
    )

    for fine_element in fine_mesh.Elements(ng.VOL):
        vertex_points = [
            fine_mesh.vertices[vertex.nr].point
            for vertex in fine_element.vertices
        ]
        barycenter = np.mean(
            np.asarray(vertex_points, dtype=float),
            axis=0,
        )
        coarse_point = coarse_mesh(
            float(barycenter[0]),
            float(barycenter[1]),
        )
        parents[fine_element.nr] = coarse_point.nr

    if np.any(parents < 0) or np.any(parents >= coarse_mesh.ne):
        raise RuntimeError(
            "Failed to map every fine element to a coarse parent."
        )

    return parents


def restrict_piecewise_constant_load(
    fine_load: np.ndarray,
    *,
    parent_indices: np.ndarray,
    number_of_coarse_elements: int,
) -> np.ndarray:
    """Apply ``P_theta.T`` by summing loads over fine child elements."""
    fine_values = np.asarray(
        fine_load,
        dtype=float,
    )
    parents = np.asarray(
        parent_indices,
        dtype=int,
    )

    if fine_values.shape != parents.shape:
        raise ValueError(
            "fine_load and parent_indices must have the same shape."
        )

    return np.bincount(
        parents,
        weights=fine_values,
        minlength=number_of_coarse_elements,
    ).astype(float, copy=False)


@dataclass
class HierarchicalSPDESampler:
    """Generate a coupled SPDE conductivity pair for one MLMC term."""

    level_index: int
    upper_level: SPDELevel
    lower_level: SPDELevel | None
    fine_to_coarse_parent: np.ndarray | None
    evaluation_X: np.ndarray
    evaluation_Y: np.ndarray
    config: SPDEConfig

    @classmethod
    def build(
        cls,
        *,
        level_index: int,
        coarse_maxh: float,
        evaluation_grid_size: int,
        config: SPDEConfig,
    ) -> "HierarchicalSPDESampler":
        """Build only the one or two SPDE levels needed by ``Y_l``."""
        if level_index < 0:
            raise ValueError(
                "level_index must be nonnegative."
            )

        if coarse_maxh <= 0.0:
            raise ValueError(
                "coarse_maxh must be positive."
            )

        if evaluation_grid_size < 2:
            raise ValueError(
                "evaluation_grid_size must be at least 2."
            )

        working_mesh = ng.Mesh(
            unit_square.GenerateMesh(
                maxh=coarse_maxh
            )
        )

        requested_levels = {
            level_index,
        }
        if level_index > 0:
            requested_levels.add(level_index - 1)

        built_levels: dict[int, SPDELevel] = {}

        for current_level in range(level_index + 1):
            if current_level in requested_levels:
                snapshot = ng.Mesh(
                    working_mesh.ngmesh.Copy()
                )
                built_levels[current_level] = SPDELevel.build(
                    level_index=current_level,
                    mesh=snapshot,
                    config=config,
                )

            if current_level < level_index:
                working_mesh.Refine()

        upper_level = built_levels[level_index]
        lower_level = (
            None
            if level_index == 0
            else built_levels[level_index - 1]
        )

        if lower_level is None:
            parent_indices = None
        else:
            parent_indices = element_parent_indices(
                coarse_mesh=lower_level.mesh,
                fine_mesh=upper_level.mesh,
            )

            child_counts = np.bincount(
                parent_indices,
                minlength=lower_level.mesh.ne,
            )
            if np.any(child_counts == 0):
                raise RuntimeError(
                    "At least one coarse element has no fine children."
                )

            # If f_f ~ N(0, W_f), then P_theta.T f_f must have covariance
            # P_theta.T W_f P_theta = W_c.  For L2(order=0), this identity
            # says that the areas (mass entries) of all fine children sum to
            # the area of their coarse parent.  Check it explicitly so a bad
            # element map cannot silently destroy the MLMC coupling.
            restricted_fine_mass = np.bincount(
                parent_indices,
                weights=upper_level.field_mass_diagonal,
                minlength=lower_level.field_space.ndof,
            )
            if not np.allclose(
                restricted_fine_mass,
                lower_level.field_mass_diagonal,
                rtol=1.0e-12,
                atol=1.0e-14,
            ):
                raise RuntimeError(
                    "The fine-to-coarse L2 transfer does not satisfy "
                    "P_theta.T W_f P_theta = W_c."
                )

        evaluation_X, evaluation_Y, _ = cartesian_grid_2d(
            evaluation_grid_size,
            evaluation_grid_size,
        )

        return cls(
            level_index=level_index,
            upper_level=upper_level,
            lower_level=lower_level,
            fine_to_coarse_parent=parent_indices,
            evaluation_X=evaluation_X,
            evaluation_Y=evaluation_Y,
            config=config,
        )

    def evaluate_log_conductivity(
        self,
        level: SPDELevel,
    ) -> np.ndarray:
        """Evaluate ``mean + sigma*theta_h`` on the bridge grid."""
        mesh_points = level.mesh(
            self.evaluation_X.ravel(),
            self.evaluation_Y.ravel(),
        )
        theta_values = np.asarray(
            level.log_field(mesh_points),
            dtype=float,
        ).reshape(self.evaluation_X.shape)

        log_values = (
            self.config.mean_log_conductivity
            + self.config.standard_deviation
            * theta_values
        )

        if not np.all(np.isfinite(log_values)):
            raise RuntimeError(
                f"SPDE level {level.level_index} produced nonfinite "
                "log-conductivity values."
            )

        return log_values

    @staticmethod
    def conductivity_from_log_values(
        log_values: np.ndarray,
    ) -> ng.CoefficientFunction:
        """Exponentiate a sampled log field and create a voxel bridge."""
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            conductivity_values = np.exp(log_values)

        if (
            not np.all(np.isfinite(conductivity_values))
            or np.any(conductivity_values <= 0.0)
        ):
            raise RuntimeError(
                "The SPDE conductivity must be finite and positive."
            )

        return voxel_coefficient_2d(
            conductivity_values,
            linear=True,
        )

    def draw(
        self,
        rng: np.random.Generator,
    ) -> CoupledConductivitySample:
        """Draw one coupled upper/lower SPDE conductivity realization."""
        upper_load = self.upper_level.draw_white_noise_load(
            rng
        )

        if self.lower_level is None:
            lower_log_values = None
            lower_conductivity = None
        else:
            if self.fine_to_coarse_parent is None:
                raise RuntimeError(
                    "A correction sampler requires an element-parent map."
                )

            lower_load = restrict_piecewise_constant_load(
                upper_load,
                parent_indices=self.fine_to_coarse_parent,
                number_of_coarse_elements=(
                    self.lower_level.field_space.ndof
                ),
            )
            self.lower_level.solve(lower_load)
            lower_log_values = self.evaluate_log_conductivity(
                self.lower_level
            )
            lower_conductivity = self.conductivity_from_log_values(
                lower_log_values
            )

        self.upper_level.solve(upper_load)
        upper_log_values = self.evaluate_log_conductivity(
            self.upper_level
        )
        upper_conductivity = self.conductivity_from_log_values(
            upper_log_values
        )

        return CoupledConductivitySample(
            upper=upper_conductivity,
            lower=lower_conductivity,
            upper_log_values=upper_log_values,
            lower_log_values=lower_log_values,
        )


@dataclass
class SPDEMLMCTerm:
    """One independently sampled MLMC term driven by an SPDE field."""

    level_index: int
    upper_level: MCLevel
    lower_level: MCLevel | None
    sampler: HierarchicalSPDESampler
    rng: np.random.Generator

    sample_count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0

    upper_qois: list[float] = field(default_factory=list)
    lower_qois: list[float] = field(default_factory=list)
    sample_durations_seconds: list[float] = field(default_factory=list)

    def draw_sample(self) -> float:
        """Draw the SPDE field, solve the Darcy problem, and store ``Y_l``."""
        sample_started = time.perf_counter()
        conductivity = self.sampler.draw(self.rng)

        if self.level_index == 0:
            q_lower = None
            q_upper = self.upper_level.solve_qoi(
                conductivity.upper
            )
        else:
            if self.lower_level is None or conductivity.lower is None:
                raise RuntimeError(
                    "A correction term requires coupled lower objects."
                )

            q_lower = self.lower_level.solve_qoi(
                conductivity.lower
            )
            q_upper = self.upper_level.solve_qoi(
                conductivity.upper
            )

        correction = self.update_statistics(
            q_upper=q_upper,
            q_lower=q_lower,
        )
        self.sample_durations_seconds.append(
            time.perf_counter() - sample_started
        )
        return correction

    def add_samples(self, number_of_samples: int) -> None:
        """Draw a requested number of additional correction samples."""
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
        q_lower: float | None,
    ) -> float:
        """Store QoIs and update Welford statistics for ``Y_l``."""
        q_upper = float(q_upper)

        if not np.isfinite(q_upper):
            raise RuntimeError(
                f"Y_{self.level_index} received a nonfinite upper QoI."
            )

        if self.level_index == 0:
            if q_lower is not None:
                raise ValueError(
                    "Y_0 must not receive q_lower."
                )
            correction = q_upper
        else:
            if q_lower is None:
                raise ValueError(
                    "A correction term requires q_lower."
                )
            q_lower = float(q_lower)
            if not np.isfinite(q_lower):
                raise RuntimeError(
                    f"Y_{self.level_index} received a nonfinite lower QoI."
                )
            correction = q_upper - q_lower

        if not np.isfinite(correction):
            raise RuntimeError(
                f"Y_{self.level_index} produced a nonfinite correction."
            )

        self.upper_qois.append(q_upper)
        if q_lower is not None:
            self.lower_qois.append(q_lower)

        self.sample_count += 1
        difference = correction - self.mean
        self.mean += difference / self.sample_count
        self.sum_squared_deviations += (
            difference
            * (correction - self.mean)
        )

        return correction

    @property
    def corrections(self) -> np.ndarray:
        """Return all stored samples of this correction term."""
        upper = np.asarray(self.upper_qois, dtype=float)
        if self.level_index == 0:
            return upper
        return upper - np.asarray(self.lower_qois, dtype=float)

    @property
    def sample_variance(self) -> float:
        """Return the unbiased sample variance of individual ``Y_l`` values."""
        if self.sample_count < 2:
            return float("inf")
        return self.sum_squared_deviations / (self.sample_count - 1)

    @property
    def variance_of_mean(self) -> float:
        """Return the estimated variance of the sample mean of ``Y_l``."""
        if self.sample_count < 2:
            return float("inf")
        return self.sample_variance / self.sample_count

    @property
    def total_sampling_time_seconds(self) -> float:
        """Return total time spent on successful samples of this term."""
        return float(sum(self.sample_durations_seconds))

    @property
    def mean_sample_time_seconds(self) -> float:
        """Return mean time per successful sample of this term."""
        if not self.sample_durations_seconds:
            return float("nan")
        return float(np.mean(self.sample_durations_seconds))

    def run_to_mean_variance_target(
        self,
        target_variance: float,
        *,
        minimum_samples: int,
        maximum_samples: int,
        samples_per_iteration: int,
    ) -> bool:
        """Sample until ``Var(Y_l)/N_l`` reaches its assigned target."""
        if not np.isfinite(target_variance) or target_variance <= 0.0:
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

        if self.sample_count < minimum_samples:
            self.add_samples(
                minimum_samples - self.sample_count
            )

        if self.variance_of_mean <= target_variance:
            return True

        maximum_iterations = math.ceil(
            (maximum_samples - self.sample_count)
            / samples_per_iteration
        )

        for _ in range(maximum_iterations):
            remaining = maximum_samples - self.sample_count
            if remaining <= 0:
                return False

            self.add_samples(
                min(samples_per_iteration, remaining)
            )
            if self.variance_of_mean <= target_variance:
                return True

        return False


@dataclass
class SPDEMultilevelMonteCarlo:
    """Manage independently sampled SPDE-driven MLMC correction terms."""

    terms: tuple[SPDEMLMCTerm, ...]

    @classmethod
    def create(
        cls,
        *,
        number_of_levels: int,
        mc_level_factory: MCLevelFactory,
        spde_coarse_maxh: float,
        evaluation_grid_size: int,
        spde_config: SPDEConfig,
        seed: int,
    ) -> "SPDEMultilevelMonteCarlo":
        """Build private Darcy and SPDE state for every correction term."""
        if number_of_levels < 1:
            raise ValueError(
                "number_of_levels must be positive."
            )

        child_seeds = np.random.SeedSequence(seed).spawn(
            number_of_levels
        )
        terms: list[SPDEMLMCTerm] = []

        for level_index in range(number_of_levels):
            upper_level = mc_level_factory(level_index)
            lower_level = (
                None
                if level_index == 0
                else mc_level_factory(level_index - 1)
            )
            sampler = HierarchicalSPDESampler.build(
                level_index=level_index,
                coarse_maxh=spde_coarse_maxh,
                evaluation_grid_size=evaluation_grid_size,
                config=spde_config,
            )

            terms.append(
                SPDEMLMCTerm(
                    level_index=level_index,
                    upper_level=upper_level,
                    lower_level=lower_level,
                    sampler=sampler,
                    rng=np.random.default_rng(
                        child_seeds[level_index]
                    ),
                )
            )

        return cls(terms=tuple(terms))

    def run_fixed(self, samples_per_term: int) -> None:
        """Run the same fixed number of samples for every MLMC term."""
        if samples_per_term < 0:
            raise ValueError(
                "samples_per_term must be nonnegative."
            )
        for term in self.terms:
            term.add_samples(samples_per_term)

    def run_target_variance(
        self,
        target_variance: float,
        *,
        minimum_samples: int = 10,
        maximum_samples_per_term: int = 100_000,
        samples_per_iteration: int = 1,
    ) -> bool:
        """Use equal per-term variance targets, matching multilevel_test.py."""
        if not np.isfinite(target_variance) or target_variance <= 0.0:
            raise ValueError(
                "target_variance must be finite and positive."
            )

        per_term_target = target_variance / len(self.terms)
        converged = [
            term.run_to_mean_variance_target(
                target_variance=per_term_target,
                minimum_samples=minimum_samples,
                maximum_samples=maximum_samples_per_term,
                samples_per_iteration=samples_per_iteration,
            )
            for term in self.terms
        ]

        return (
            all(converged)
            and self.estimator_variance <= target_variance
        )

    @property
    def estimate_qoi(self) -> float:
        """Return the telescoping MLMC estimate ``sum_l mean(Y_l)``."""
        if any(term.sample_count == 0 for term in self.terms):
            raise RuntimeError(
                "Every term requires at least one sample."
            )
        return float(sum(term.mean for term in self.terms))

    @property
    def estimator_variance(self) -> float:
        """Return ``sum_l Var(Y_l)/N_l``."""
        if any(term.sample_count < 2 for term in self.terms):
            raise RuntimeError(
                "At least two samples per term are required."
            )
        return float(
            sum(term.variance_of_mean for term in self.terms)
        )

    @property
    def standard_error(self) -> float:
        """Return the sampling standard deviation of the MLMC estimate."""
        return float(np.sqrt(self.estimator_variance))

    def print_summary(self) -> None:
        """Print correction statistics and measured sample costs."""
        print()
        print("SPDE-driven MLMC summary")
        print(
            f"{'term':>6} {'N':>7} {'mean':>14} {'variance':>14} "
            f"{'V/N':>14} {'time/sample[s]':>15}"
        )
        print("-" * 82)

        for term in self.terms:
            print(
                f"Y_{term.level_index:<4} "
                f"{term.sample_count:>7d} "
                f"{term.mean:>14.6e} "
                f"{term.sample_variance:>14.6e} "
                f"{term.variance_of_mean:>14.6e} "
                f"{term.mean_sample_time_seconds:>15.6e}"
            )

        print()
        print(f"MLMC estimate: {self.estimate_qoi:.8e}")
        print(
            f"Estimator variance: {self.estimator_variance:.8e}"
        )
        print(f"Standard error: {self.standard_error:.8e}")
        print(
            "Total measured sample time: "
            f"{sum(term.total_sampling_time_seconds for term in self.terms):.3f}s"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the standalone validation run."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate hierarchical mixed-SPDE sampling with the existing "
            "multilevel Darcy solver."
        )
    )
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--samples-per-term", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coarse-maxh", type=float, default=0.3)
    parser.add_argument("--evaluation-grid-size", type=int, default=32)
    parser.add_argument(
        "--correlation-length",
        type=float,
        default=0.3,
        help=(
            "Nominal Matern correlation parameter; the prototype uses "
            "inverse_correlation_scale = 1/correlation_length."
        ),
    )
    parser.add_argument("--mean-log-conductivity", type=float, default=0.0)
    parser.add_argument("--standard-deviation", type=float, default=1.0)
    parser.add_argument("--linear-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--maximum-vcycles", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    """Build and execute the standalone SPDE-driven MLMC test."""
    arguments = parse_arguments()

    if arguments.levels < 1:
        raise ValueError("levels must be positive.")
    if arguments.samples_per_term < 2:
        raise ValueError(
            "samples-per-term must be at least 2 for variance estimates."
        )
    if arguments.correlation_length <= 0.0:
        raise ValueError(
            "correlation-length must be positive."
        )

    spde_config = SPDEConfig(
        inverse_correlation_scale=(
            1.0 / arguments.correlation_length
        ),
        mean_log_conductivity=(
            arguments.mean_log_conductivity
        ),
        standard_deviation=arguments.standard_deviation,
    )

    flow_level_factory = MCLevelFactory(
        coarse_maxh=arguments.coarse_maxh,
        relative_solver_tolerance=(
            arguments.linear_tolerance
        ),
        maximum_vcycles=arguments.maximum_vcycles,
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

    print("Building SPDE and Darcy term objects")
    print(f"  levels:                    {arguments.levels}")
    print(f"  samples per term:          {arguments.samples_per_term}")
    print(f"  coarse maxh:               {arguments.coarse_maxh}")
    print(f"  correlation length input:  {arguments.correlation_length}")
    print(
        "  inverse correlation scale: "
        f"{spde_config.inverse_correlation_scale:.6e}"
    )
    print(
        f"  white-noise scale g:       {spde_config.forcing_scale:.6e}"
    )

    build_started = time.perf_counter()
    mlmc = SPDEMultilevelMonteCarlo.create(
        number_of_levels=arguments.levels,
        mc_level_factory=flow_level_factory,
        spde_coarse_maxh=arguments.coarse_maxh,
        evaluation_grid_size=arguments.evaluation_grid_size,
        spde_config=spde_config,
        seed=arguments.seed,
    )
    print(
        f"Build time: {time.perf_counter() - build_started:.3f}s"
    )
    print()
    print(
        f"{'term':>6} {'upper SPDE theta dofs':>24} "
        f"{'lower SPDE theta dofs':>24}"
    )
    print("-" * 58)
    for term in mlmc.terms:
        lower_dofs = (
            "-"
            if term.sampler.lower_level is None
            else str(term.sampler.lower_level.field_space.ndof)
        )
        print(
            f"Y_{term.level_index:<4} "
            f"{term.sampler.upper_level.field_space.ndof:>24d} "
            f"{lower_dofs:>24}"
        )

    run_started = time.perf_counter()
    mlmc.run_fixed(arguments.samples_per_term)
    run_wall_time = time.perf_counter() - run_started

    mlmc.print_summary()
    print(f"Complete fixed-sample wall time: {run_wall_time:.3f}s")

    for term in mlmc.terms:
        if term.sample_count != len(term.corrections):
            raise RuntimeError(
                f"Y_{term.level_index} has inconsistent sample storage."
            )
        if term.sample_count != len(term.sample_durations_seconds):
            raise RuntimeError(
                f"Y_{term.level_index} has inconsistent timing storage."
            )


if __name__ == "__main__":
    main()
