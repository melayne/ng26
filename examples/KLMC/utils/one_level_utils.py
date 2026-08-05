from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np
from matplotlib.figure import Figure
from netgen.geom2d import unit_square


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_DIR = PROJECT_ROOT / "examples" / "plots"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from KL_expansion import (  # noqa: E402
    cartesian_grid_2d,
    exponential_covariance,
    leading_eigenpairs,
    lognormal_transform,
    sample_discrete_kl,
    voxel_coefficient_2d,
)


@dataclass(frozen=True)
class GridMCResult:
    nx: int
    n_modes: int
    retained_variance: float
    qoi_samples: np.ndarray
    wall_time_s: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.qoi_samples))

    @property
    def variance(self) -> float:
        return float(np.var(self.qoi_samples, ddof=1))

    @property
    def stderr(self) -> float:
        n = len(self.qoi_samples)
        return float(np.sqrt(self.variance / n))


def build_fixed_mesh(*, maxh: float) -> tuple[ng.Mesh, ng.H1]:
    """One triangular mesh / H1 space reused for every KL grid.

    Dirichlet only on left/right (pressure drive). Top/bottom are natural
    no-flow boundaries, matching a Reddy-style pressure Darcy setup.
    """
    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = ng.H1(mesh, order=1, dirichlet="left|right")
    return mesh, fes


def quantity_of_interest(solution: ng.GridFunction, mesh: ng.Mesh, kappa) -> float:
    """Outflow flux ``∫_right (-κ ∂_n p) ds``.

    Volume gradients must be wrapped with ``BoundaryFromVolumeCF`` before
    integrating on facets; otherwise NGSolve returns 0.
    """
    normal = ng.specialcf.normal(mesh.dim)
    flux_density = -kappa * ng.InnerProduct(ng.grad(solution), normal)
    return float(
        ng.Integrate(
            ng.BoundaryFromVolumeCF(flux_density),
            mesh.Boundaries("right"),
        )
    )


def solve_diffusion(fes: ng.H1, kappa) -> ng.GridFunction:
    """Solve ``-div(κ ∇p)=0`` with ``p=-1`` on left and ``p=0`` on right.

    Top/bottom have no Dirichlet condition, so the natural condition is
    no-flow (``∂_n p = 0``).  Dirichlet data are set with one ``BoundaryCF``
    call — consecutive ``Set(..., definedon=...)`` calls overwrite earlier
    boundary values.
    """
    mesh = fes.mesh
    u, v = fes.TnT()
    a = ng.BilinearForm(fes, symmetric=True)
    a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
    a.Assemble()

    f = ng.LinearForm(fes)
    f.Assemble()

    solution = ng.GridFunction(fes)
    # One Set for both Dirichlet sides (do not call Set twice).
    solution.Set(
        mesh.BoundaryCF({"left": -1.0, "right": 0.0}),
        definedon=mesh.Boundaries("left|right"),
    )

    residual = solution.vec.CreateVector()
    residual.data = f.vec - a.mat * solution.vec
    solution.vec.data += a.mat.Inverse(fes.FreeDofs()) * residual
    return solution


def prepare_kl_basis(
    nx: int,
    *,
    sigma: float,
    correlation_length: float,
    num_modes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Eigenpairs on an ``nx x nx`` Cartesian KL grid."""
    _, _, points = cartesian_grid_2d(nx, nx)
    covariance = exponential_covariance(
        points,
        sigma=sigma,
        correlation_length=correlation_length,
    )
    n_modes = min(num_modes, points.shape[0])
    eigenvalues, eigenvectors = leading_eigenpairs(covariance, num_modes=n_modes)
    total_variance = float(np.trace(covariance))
    return eigenvalues, eigenvectors, total_variance

