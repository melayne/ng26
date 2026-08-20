"""Framework-independent interfaces and scaffolding for an MLMC runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np


RandomnessT = TypeVar("RandomnessT")
ModelInputT = TypeVar("ModelInputT")


@dataclass(frozen=True)
class CoupledInputs(Generic[ModelInputT]):
    """Inputs for the current level and its coupled previous level."""

    current: ModelInputT
    previous: ModelInputT | None


class MLMCModel(Protocol[RandomnessT, ModelInputT]):
    """Interface a user model must provide to the MLMC runner."""

    def sample(
        self,
        level: int,
        rng: np.random.Generator,
    ) -> RandomnessT:
        """Draw the latent randomness associated with correction level ``level``."""
        ...

    def couple(
        self,
        level: int,
        randomness: RandomnessT,
    ) -> CoupledInputs[ModelInputT]:
        """Map one draw to coupled inputs for levels ``level`` and ``level - 1``."""
        ...

    def run(
        self,
        level: int,
        model_input: ModelInputT,
    ) -> float:
        """Run one level and return its scalar quantity of interest."""
        ...


@dataclass(frozen=True)
class MLMCRunnerConfig:
    """Configuration shared by all runs performed by an MLMC runner."""

    number_of_levels: int
    seed: int | None = None


@dataclass
class RunningStatistics:
    """Online statistics for the corrections sampled at one level."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0
    total_cost: float = 0.0

    def update(self, correction: float, cost: float) -> None:
        """Add one correction and its measured computational cost."""
        raise NotImplementedError

    @property
    def sample_variance(self) -> float:
        """Return the unbiased sample variance of the corrections."""
        raise NotImplementedError

    @property
    def mean_cost(self) -> float:
        """Return the mean measured cost per correction sample."""
        raise NotImplementedError


@dataclass(frozen=True)
class LevelResult:
    """Final sampling statistics for one MLMC correction level."""

    level: int
    sample_count: int
    mean_correction: float
    correction_variance: float
    mean_cost: float
    total_cost: float


@dataclass(frozen=True)
class MLMCResult:
    """Final MLMC estimate and its per-level diagnostics."""

    estimate: float
    estimator_variance: float
    standard_error: float
    level_results: tuple[LevelResult, ...]
    total_cost: float


@dataclass
class MLMCRunner(Generic[RandomnessT, ModelInputT]):
    """Coordinate sampling, model evaluations, and MLMC statistics."""

    model: MLMCModel[RandomnessT, ModelInputT]
    config: MLMCRunnerConfig

    def run_fixed(self, sample_counts: tuple[int, ...]) -> MLMCResult:
        """Run a fixed number of correction samples at every level."""
        raise NotImplementedError

    def _sample_correction(
        self,
        level: int,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Generate one coupled correction and return its value and cost."""
        raise NotImplementedError

    def _create_level_generators(self) -> tuple[np.random.Generator, ...]:
        """Create one reproducible random-number stream per correction level."""
        raise NotImplementedError

    def _validate_sample_counts(
        self,
        sample_counts: tuple[int, ...],
    ) -> None:
        """Check that fixed sample counts match the configured levels."""
        raise NotImplementedError

    def _build_result(
        self,
        statistics: tuple[RunningStatistics, ...],
    ) -> MLMCResult:
        """Convert accumulated level statistics into an MLMC result."""
        raise NotImplementedError


__all__ = [
    "CoupledInputs",
    "LevelResult",
    "MLMCModel",
    "MLMCResult",
    "MLMCRunner",
    "MLMCRunnerConfig",
    "RunningStatistics",
]
