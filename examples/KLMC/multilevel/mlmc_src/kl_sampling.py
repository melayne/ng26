"""Karhunen--Loeve conductivity sampling adapter for the shared MLMC core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import ngsolve as ng
import numpy as np

from src.KL_expansion import lognormal_transform, voxel_coefficient_2d

from .mlmc_core import (
    CoupledConductivitySample,
    CoupledConductivitySampler,
    QoILevel,
)


KLEvaluator = Callable[
    [np.ndarray | float, np.ndarray | float, np.ndarray],
    np.ndarray,
]
CoefficientFromXi = Callable[[np.ndarray], ng.CoefficientFunction]


def create_voxel_coefficient(
    xi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    evaluate_log_conductivity: KLEvaluator,
) -> ng.CoefficientFunction:
    """Evaluate one KL log field and return its lognormal coefficient."""
    log_kappa = np.asarray(
        evaluate_log_conductivity(X, Y, xi),
        dtype=float,
    )
    if not np.all(np.isfinite(log_kappa)):
        raise RuntimeError("The KL log-conductivity field is nonfinite.")

    with np.errstate(over="raise", under="ignore", invalid="raise"):
        kappa_values = lognormal_transform(log_kappa)

    if (
        not np.all(np.isfinite(kappa_values))
        or np.any(kappa_values <= 0.0)
    ):
        raise RuntimeError("The KL conductivity must be finite and positive.")

    return voxel_coefficient_2d(kappa_values, linear=True)


def make_coefficient_from_xi(
    X: np.ndarray,
    Y: np.ndarray,
    evaluate_log_conductivity: KLEvaluator,
) -> CoefficientFromXi:
    """Bind the fixed evaluation grid and return ``xi -> kappa``."""
    X = np.array(X, dtype=float, copy=True)
    Y = np.array(Y, dtype=float, copy=True)
    if X.shape != Y.shape or X.ndim != 2:
        raise ValueError("X and Y must be two-dimensional arrays of equal shape.")

    def coefficient_from_xi(xi: np.ndarray) -> ng.CoefficientFunction:
        return create_voxel_coefficient(
            xi=xi,
            X=X,
            Y=Y,
            evaluate_log_conductivity=evaluate_log_conductivity,
        )

    return coefficient_from_xi


@dataclass(frozen=True)
class KLCoupledConductivitySampler(CoupledConductivitySampler):
    """Draw one KL vector and reuse its physical field in both PDE solves."""

    level_index: int
    number_of_modes: int
    coefficient_from_xi: CoefficientFromXi

    def __post_init__(self) -> None:
        if self.level_index < 0:
            raise ValueError("level_index must be nonnegative.")
        if self.number_of_modes < 1:
            raise ValueError("number_of_modes must be positive.")

    def draw(
        self,
        rng: np.random.Generator,
    ) -> CoupledConductivitySample:
        """Return the one physical realization required for ``Y_l``."""
        xi = rng.standard_normal(self.number_of_modes)
        kappa = self.coefficient_from_xi(xi)
        return CoupledConductivitySample(
            upper=kappa,
            lower=None if self.level_index == 0 else kappa,
        )


@dataclass(frozen=True)
class KLSamplerFactory:
    """Construct private KL samplers with a common basis/evaluator."""

    number_of_modes: int
    coefficient_from_xi: CoefficientFromXi

    def __post_init__(self) -> None:
        if self.number_of_modes < 1:
            raise ValueError("number_of_modes must be positive.")

    def __call__(
        self,
        *,
        level_index: int,
        upper_level: QoILevel,
        lower_level: QoILevel | None,
    ) -> KLCoupledConductivitySampler:
        if upper_level.level_index != level_index:
            raise ValueError("upper_level has the wrong level index.")
        if level_index == 0:
            if lower_level is not None:
                raise ValueError("Y_0 must not have a lower level.")
        elif lower_level is None or lower_level.level_index != level_index - 1:
            raise ValueError("Y_l requires its Q_(l-1) lower level.")

        return KLCoupledConductivitySampler(
            level_index=level_index,
            number_of_modes=self.number_of_modes,
            coefficient_from_xi=self.coefficient_from_xi,
        )


__all__ = [
    "CoefficientFromXi",
    "KLCoupledConductivitySampler",
    "KLEvaluator",
    "KLSamplerFactory",
    "create_voxel_coefficient",
    "make_coefficient_from_xi",
]
