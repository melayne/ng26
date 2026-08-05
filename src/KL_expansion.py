"""Discrete Karhunen--Loève fields on a Cartesian grid.

This module treats the field values at the grid points as a finite Gaussian
random vector.  Thus the eigenproblem is ``C q = lambda q``; it is not the
finite-element generalized eigenproblem that contains a mass matrix.

The final helper turns a sampled 2D array into an NGSolve
``VoxelCoefficient``.  That coefficient is defined in physical coordinates,
so the same realization can be evaluated on every multigrid level.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from scipy.linalg import eigh


Array: TypeAlias = np.ndarray
Bounds2D: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


def cartesian_grid_2d(
    nx: int,
    ny: int,
    *,
    bounds: Bounds2D = ((0.0, 1.0), (0.0, 1.0)),
) -> tuple[Array, Array, Array]:
    """Return a regular 2D grid and its flattened point coordinates.

    ``X`` and ``Y`` have shape ``(ny, nx)``.  The rows of ``points`` follow
    the same C-order flattening, so the x-coordinate varies fastest.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least 2.")

    (xmin, xmax), (ymin, ymax) = bounds
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("Each bounds pair must be strictly increasing.")

    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    points = np.column_stack((X.ravel(), Y.ravel()))
    return X, Y, points


def exponential_covariance(
    points: Array,
    *,
    sigma: float = 1.0,
    correlation_length: float = 0.3,
) -> Array:
    """Build ``C_ij = sigma**2 exp(-|x_i-x_j|/correlation_length)``."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must have shape (number_of_points, dimension).")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive.")

    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return sigma**2 * np.exp(-distances / correlation_length)


def leading_eigenpairs(covariance: Array, num_modes: int) -> tuple[Array, Array]:
    """Return the largest eigenvalues and eigenvectors of a covariance matrix.

    The eigenvalues are returned in descending order.  Only the requested
    part of the symmetric spectrum is computed.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("covariance must be symmetric.")

    npoints = covariance.shape[0]
    if not 1 <= num_modes <= npoints:
        raise ValueError(f"num_modes must lie between 1 and {npoints}.")

    first = npoints - num_modes
    eigenvalues, eigenvectors = eigh(
        covariance,
        subset_by_index=(first, npoints - 1),
        check_finite=True,
    )

    # eigh returns ascending eigenvalues.  Tiny negative roundoff values are
    # harmless for a positive-semidefinite covariance matrix.
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors


def sample_discrete_kl(
    mean: float | Array,
    eigenvalues: Array,
    eigenvectors: Array,
    *,
    shape: tuple[int, int],
    rng: np.random.Generator | int | None = None,
) -> tuple[Array, Array]:
    """Sample a truncated discrete KL expansion.

    Returns ``(gaussian_values, xi)`` where ``gaussian_values`` has the given
    ``(ny, nx)`` shape and

    ``gaussian = mean + eigenvectors @ (sqrt(eigenvalues) * xi)``.

    ``rng`` may be a NumPy generator, an integer seed, or ``None``.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    ny, nx = shape
    npoints = nx * ny

    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional.")
    if np.any(eigenvalues < 0):
        raise ValueError("eigenvalues must be nonnegative.")
    if eigenvectors.shape != (npoints, len(eigenvalues)):
        raise ValueError(
            "eigenvectors must have shape "
            f"({npoints}, {len(eigenvalues)}), got {eigenvectors.shape}."
        )

    mean_array = np.asarray(mean, dtype=float)
    if mean_array.ndim == 0:
        mean_flat = np.full(npoints, float(mean_array))
    elif mean_array.size == npoints:
        mean_flat = mean_array.reshape(-1)
    else:
        raise ValueError(f"mean must be scalar or contain {npoints} values.")

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    xi = generator.standard_normal(len(eigenvalues))
    gaussian_flat = mean_flat + eigenvectors @ (np.sqrt(eigenvalues) * xi)
    return gaussian_flat.reshape(shape), xi


def lognormal_transform(gaussian_values: Array) -> Array:
    """Exponentiate a Gaussian field to obtain a positive coefficient."""
    return np.exp(np.asarray(gaussian_values, dtype=float))


def voxel_coefficient_2d(
    values: Array,
    *,
    bounds: Bounds2D = ((0.0, 1.0), (0.0, 1.0)),
    linear: bool = True,
):
    """Wrap ``(ny, nx)`` grid values as an NGSolve VoxelCoefficient."""
    import ngsolve as ng

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (ny, nx).")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must all be finite.")

    (xmin, xmax), (ymin, ymax) = bounds
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("Each bounds pair must be strictly increasing.")

    return ng.VoxelCoefficient(
        (xmin, ymin),
        (xmax, ymax),
        values,
        linear=linear,
    )


__all__ = [
    "Bounds2D",
    "cartesian_grid_2d",
    "exponential_covariance",
    "leading_eigenpairs",
    "lognormal_transform",
    "sample_discrete_kl",
    "voxel_coefficient_2d",
]

