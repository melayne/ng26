"""Hierarchically coupled SPDE conductivity sampling for MLMC.

The Gaussian log-conductivity is sampled with the mixed finite-element SPDE

    (u, v) + (div(v), theta) = 0,
    (div(u), q) - kappa_s**2 (theta, q) = -g (W, q).

Lowest-order ``HDiv`` is used for ``u`` and piecewise-constant ``L2`` for
``theta``. For a correction ``Y_l``, the lower stochastic load is obtained
by restricting the upper load, rather than drawing independent noise. This
is the essential coupling that makes ``Q_l - Q_(l-1)`` small.

This validation implementation uses reusable direct SPDE inverses. It can be
replaced later by a scalable SPDE solver without changing ``mlmc_core.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square

from src.KL_expansion import cartesian_grid_2d, voxel_coefficient_2d

from .mlmc_core import CoupledConductivitySample, QoILevel


@dataclass(frozen=True)
class SPDEConfig:
    """Parameters of the Gaussian reaction-diffusion SPDE.

    ``inverse_correlation_scale`` is the parameter often denoted ``kappa``
    in the SPDE literature. The longer name distinguishes it from the Darcy
    conductivity. In two dimensions, the default white-noise scaling for
    unit marginal variance on an unbounded domain is

        g = 2 sqrt(pi) inverse_correlation_scale.
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
            raise ValueError("mean_log_conductivity must be finite.")
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
            raise ValueError("white_noise_scale must be finite and positive.")

    @property
    def forcing_scale(self) -> float:
        if self.white_noise_scale is not None:
            return float(self.white_noise_scale)
        return float(
            2.0 * np.sqrt(np.pi) * self.inverse_correlation_scale
        )


@dataclass
class SPDELevel:
    """One reusable mixed finite-element discretization of the SPDE."""

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
        """Assemble the mixed system and a reusable direct inverse."""
        flux_space = ng.HDiv(mesh, order=1, dirichlet=".*")
        field_space = ng.L2(mesh, order=0)
        mixed_space = ng.FESpace([flux_space, field_space])

        (u, theta), (v, q) = mixed_space.TnT()
        inverse_scale = config.inverse_correlation_scale
        system = ng.BilinearForm(mixed_space, symmetric=True)
        system += (
            ng.InnerProduct(u, v)
            + ng.div(v) * theta
            + ng.div(u) * q
            - inverse_scale**2 * theta * q
        ) * ng.dx
        system.Assemble()

        field_mass = ng.BilinearForm(field_space, symmetric=True)
        field_trial, field_test = field_space.TnT()
        field_mass += field_trial * field_test * ng.dx
        field_mass.Assemble()
        field_mass_diagonal = (
            field_mass.mat.AsVector().FV().NumPy().copy()
        )

        if field_mass_diagonal.shape != (field_space.ndof,):
            raise RuntimeError(
                "Expected one L2(order=0) mass entry per field DOF."
            )
        if (
            not np.all(np.isfinite(field_mass_diagonal))
            or np.any(field_mass_diagonal <= 0.0)
        ):
            raise RuntimeError("The SPDE field mass must be finite and positive.")

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
        """Draw ``W_h**(1/2) xi`` with ``xi ~ N(0, I)``."""
        xi = rng.standard_normal(self.field_space.ndof)
        return np.sqrt(self.field_mass_diagonal) * xi

    def solve(self, white_noise_load: np.ndarray) -> ng.GridFunction:
        """Solve the SPDE for an already drawn or restricted load."""
        load = np.asarray(white_noise_load, dtype=float)
        expected_shape = (self.field_space.ndof,)
        if load.shape != expected_shape:
            raise ValueError(
                f"white_noise_load must have shape {expected_shape}, "
                f"but received {load.shape}."
            )
        if not np.all(np.isfinite(load)):
            raise ValueError("white_noise_load must contain finite values.")

        right_hand_side = self.solution.vec.CreateVector()
        right_hand_side[:] = 0.0
        field_range = self.mixed_space.Range(1)
        right_hand_side.FV().NumPy()[
            field_range.start:field_range.stop
        ] = -self.config.forcing_scale * load

        self.solution.vec[:] = 0.0
        self.solution.vec.data = self.inverse * right_hand_side
        theta_values = self.log_field.vec.FV().NumPy()
        if not np.all(np.isfinite(theta_values)):
            raise RuntimeError(
                f"SPDE level {self.level_index} produced a nonfinite field."
            )
        return self.solution

    @property
    def log_field(self) -> ng.GridFunction:
        return self.solution.components[1]


def element_parent_indices(
    *,
    coarse_mesh: ng.Mesh,
    fine_mesh: ng.Mesh,
) -> np.ndarray:
    """Map every fine element to its containing coarse element."""
    parents = np.empty(fine_mesh.ne, dtype=int)
    for fine_element in fine_mesh.Elements(ng.VOL):
        vertex_points = [
            fine_mesh.vertices[vertex.nr].point
            for vertex in fine_element.vertices
        ]
        barycenter = np.mean(
            np.asarray(vertex_points, dtype=float),
            axis=0,
        )
        parents[fine_element.nr] = coarse_mesh(
            float(barycenter[0]),
            float(barycenter[1]),
        ).nr

    if np.any(parents < 0) or np.any(parents >= coarse_mesh.ne):
        raise RuntimeError("Failed to identify every fine element's parent.")
    return parents


def restrict_piecewise_constant_load(
    fine_load: np.ndarray,
    *,
    parent_indices: np.ndarray,
    number_of_coarse_elements: int,
) -> np.ndarray:
    """Apply ``P_theta.T`` by summing fine loads over coarse parents."""
    fine_values = np.asarray(fine_load, dtype=float)
    parents = np.asarray(parent_indices, dtype=int)
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
    """Generate a coupled SPDE conductivity pair for one ``Y_l``."""

    level_index: int
    upper_level: SPDELevel
    lower_level: SPDELevel | None
    fine_to_coarse_parent: np.ndarray | None
    evaluation_X: np.ndarray
    evaluation_Y: np.ndarray
    config: SPDEConfig
    last_upper_log_values: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_lower_log_values: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def build(
        cls,
        *,
        level_index: int,
        coarse_maxh: float,
        evaluation_grid_size: int,
        config: SPDEConfig,
    ) -> "HierarchicalSPDESampler":
        """Build only SPDE levels ``l`` and, if needed, ``l-1``."""
        if level_index < 0:
            raise ValueError("level_index must be nonnegative.")
        if not np.isfinite(coarse_maxh) or coarse_maxh <= 0.0:
            raise ValueError("coarse_maxh must be finite and positive.")
        if evaluation_grid_size < 2:
            raise ValueError("evaluation_grid_size must be at least 2.")

        working_mesh = ng.Mesh(
            unit_square.GenerateMesh(maxh=coarse_maxh)
        )
        requested_levels = {level_index}
        if level_index > 0:
            requested_levels.add(level_index - 1)

        built_levels: dict[int, SPDELevel] = {}
        for current_level in range(level_index + 1):
            if current_level in requested_levels:
                snapshot = ng.Mesh(working_mesh.ngmesh.Copy())
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
                raise RuntimeError("A coarse element has no fine children.")

            # For piecewise constants, this checks
            # P_theta.T W_f P_theta = W_c.
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
                    "The stochastic transfer does not satisfy "
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

    def evaluate_log_conductivity(self, level: SPDELevel) -> np.ndarray:
        """Evaluate ``mean + sigma theta_h`` on the common voxel grid."""
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
            + self.config.standard_deviation * theta_values
        )
        if not np.all(np.isfinite(log_values)):
            raise RuntimeError(
                f"SPDE level {level.level_index} produced a nonfinite log field."
            )
        return log_values

    @staticmethod
    def conductivity_from_log_values(
        log_values: np.ndarray,
    ) -> ng.CoefficientFunction:
        """Exponentiate the log field and form a coordinate-based bridge."""
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            conductivity_values = np.exp(log_values)
        if (
            not np.all(np.isfinite(conductivity_values))
            or np.any(conductivity_values <= 0.0)
        ):
            raise RuntimeError("SPDE conductivity must be finite and positive.")
        return voxel_coefficient_2d(conductivity_values, linear=True)

    def draw(
        self,
        rng: np.random.Generator,
    ) -> CoupledConductivitySample:
        """Draw one upper load and restrict it for the coupled lower field."""
        upper_load = self.upper_level.draw_white_noise_load(rng)

        if self.lower_level is None:
            lower_log_values = None
            lower_conductivity = None
        else:
            if self.fine_to_coarse_parent is None:
                raise RuntimeError("A correction sampler needs a parent map.")
            lower_load = restrict_piecewise_constant_load(
                upper_load,
                parent_indices=self.fine_to_coarse_parent,
                number_of_coarse_elements=self.lower_level.field_space.ndof,
            )
            self.lower_level.solve(lower_load)
            lower_log_values = self.evaluate_log_conductivity(
                self.lower_level
            )
            lower_conductivity = self.conductivity_from_log_values(
                lower_log_values
            )

        self.upper_level.solve(upper_load)
        upper_log_values = self.evaluate_log_conductivity(self.upper_level)
        upper_conductivity = self.conductivity_from_log_values(
            upper_log_values
        )

        self.last_upper_log_values = upper_log_values
        self.last_lower_log_values = lower_log_values
        return CoupledConductivitySample(
            upper=upper_conductivity,
            lower=lower_conductivity,
        )


@dataclass(frozen=True)
class SPDESamplerFactory:
    """Build the private hierarchical SPDE sampler for each MLMC term."""

    coarse_maxh: float
    evaluation_grid_size: int
    config: SPDEConfig

    def __call__(
        self,
        *,
        level_index: int,
        upper_level: QoILevel,
        lower_level: QoILevel | None,
    ) -> HierarchicalSPDESampler:
        if upper_level.level_index != level_index:
            raise ValueError("upper_level has the wrong level index.")
        if level_index == 0:
            if lower_level is not None:
                raise ValueError("Y_0 must not have a lower level.")
        elif lower_level is None or lower_level.level_index != level_index - 1:
            raise ValueError("Y_l requires its Q_(l-1) lower level.")

        return HierarchicalSPDESampler.build(
            level_index=level_index,
            coarse_maxh=self.coarse_maxh,
            evaluation_grid_size=self.evaluation_grid_size,
            config=self.config,
        )


__all__ = [
    "HierarchicalSPDESampler",
    "SPDEConfig",
    "SPDELevel",
    "SPDESamplerFactory",
    "element_parent_indices",
    "restrict_piecewise_constant_load",
]
