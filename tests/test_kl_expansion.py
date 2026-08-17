"""Tests for the standalone Karhunen-Loeve expansion module."""

import numpy as np
import pytest

from src.kl_expansion import (
    KLExpansion,
    TensorProductGrid,
    exponential_covariance,
    matern_covariance,
)


def test_tensor_grid_weights_integrate_unit_square():
    grid = TensorProductGrid.from_bounds(shape=(7, 9))
    x, y = grid.points.T

    assert grid.points.shape == (63, 2)
    assert grid.weights.shape == (63,)
    assert np.isclose(np.sum(grid.weights), 1.0)
    assert np.isclose(np.dot(grid.weights, x + y), 1.0)


def test_matern_half_is_exponential():
    grid = TensorProductGrid.from_bounds(shape=(4, 5))
    exponential = exponential_covariance(
        grid.points, variance=1.7, correlation_length=(0.2, 0.4)
    )
    matern = matern_covariance(
        grid.points,
        variance=1.7,
        correlation_length=(0.2, 0.4),
        smoothness=0.5,
    )

    assert np.allclose(matern, exponential)


def test_weighted_modes_are_orthonormal_and_ordered():
    grid = TensorProductGrid.from_bounds(shape=(6, 5))
    covariance = exponential_covariance(grid.points, correlation_length=0.25)
    kl = KLExpansion.from_covariance(
        covariance,
        num_modes=8,
        quadrature_weights=grid.weights,
        field_shape=grid.shape,
    )

    gram = kl.modes.T @ (grid.weights[:, None] * kl.modes)
    assert np.allclose(gram, np.eye(kl.num_modes), atol=1e-11)
    assert np.all(np.diff(kl.eigenvalues) <= 0.0)
    assert np.all(kl.eigenvalues > 0.0)
    assert 0.0 < kl.captured_variance_fraction < 1.0

    # C(x, x) = 1 on a unit-area domain, so the integrated variance is one.
    assert np.isclose(kl.total_variance, 1.0)


def test_full_weighted_expansion_reconstructs_covariance():
    grid = TensorProductGrid.from_bounds(shape=(4, 3))
    covariance = exponential_covariance(grid.points, correlation_length=0.3)
    kl = KLExpansion.from_covariance(
        covariance,
        num_modes=len(grid.weights),
        quadrature_weights=grid.weights,
    )

    assert np.allclose(kl.covariance_approximation(), covariance, atol=1e-11)
    assert np.isclose(kl.captured_variance_fraction, 1.0)


def test_mass_matrix_modes_use_mass_inner_product():
    covariance = np.array(
        [
            [1.0, 0.4, 0.1],
            [0.4, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ]
    )
    mass = np.array(
        [
            [2.0, 0.2, 0.0],
            [0.2, 1.5, 0.1],
            [0.0, 0.1, 1.0],
        ]
    )
    kl = KLExpansion.from_covariance(
        covariance,
        num_modes=3,
        mass_matrix=mass,
    )

    assert np.allclose(kl.modes.T @ mass @ kl.modes, np.eye(3), atol=1e-11)
    assert np.allclose(kl.covariance_approximation(), covariance, atol=1e-11)


def test_realization_projection_and_sampling_shapes():
    grid = TensorProductGrid.from_bounds(shape=(4, 3))
    covariance = exponential_covariance(grid.points, correlation_length=0.3)
    kl = KLExpansion.from_covariance(
        covariance,
        num_modes=5,
        mean=0.25,
        quadrature_weights=grid.weights,
        field_shape=grid.shape,
    )

    coefficients = np.array([0.2, -0.5, 1.0, 0.7, -0.1])
    field = kl.realization(coefficients)
    recovered = kl.project(field, standardized=True)
    assert field.shape == grid.shape
    assert np.allclose(recovered, coefficients)

    fields, draws = kl.sample_many(4, np.random.default_rng(123))
    assert fields.shape == (4, *grid.shape)
    assert draws.shape == (4, kl.num_modes)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_modes": 0}, "num_modes"),
        ({"num_modes": 3, "quadrature_weights": np.ones(2)}, "quadrature_weights"),
        (
            {
                "num_modes": 3,
                "quadrature_weights": np.ones(3),
                "mass_matrix": np.eye(3),
            },
            "not both",
        ),
    ],
)
def test_invalid_fit_inputs(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        KLExpansion.from_covariance(np.eye(3), **kwargs)
