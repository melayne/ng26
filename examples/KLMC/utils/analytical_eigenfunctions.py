"""Analytical covariance eigenpairs for the unit interval and unit square.

The formulas follow Cliffe et al. (2011) for the exponential covariance
kernel.  One-dimensional eigenpairs are computed first and combined into
two-dimensional tensor-product modes.
"""

import heapq
from typing import Literal, cast
from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ============================================================
# 1. Parameters
# ============================================================

# correlation_length = 0.20
# lambda_corr = correlation_length

# num_modes_1d = 20
# num_modes_2d = 20

# nx = 101
# ny = 101


# ============================================================
# 2. Characteristic equation from the paper
# ============================================================

def characteristic_equation(
    frequency: float,
    correlation_length: float,
) -> float:
    """
    Evaluate the pole-free characteristic equation

        (ell^2*w^2 - 1) sin(w)
        - 2*ell*w*cos(w) = 0.

    This is equivalent to

        tan(w) = 2*ell*w / (ell^2*w^2 - 1).
    """
    ell = correlation_length
    w = frequency

    return (
        (ell**2 * w**2 - 1.0) * np.sin(w)
        - 2.0 * ell * w * np.cos(w)
    )



# ============================================================
# 3. Compute the positive frequencies omega_n
# ============================================================

def calculate_frequencies(
    num_modes: int,
    correlation_length: float,
) -> np.ndarray:
    """
    Calculate the first `num_modes` positive frequencies.

    There is one nonzero frequency in each interval

        ((n-1)*pi, n*pi).

    The zero at w=0 is excluded because it produces the
    identically zero expression in the paper's eigenfunction formula.
    """
    if not isinstance(num_modes, (int, np.integer)):
        raise TypeError("num_modes must be an integer.")

    if num_modes < 1:
        raise ValueError("num_modes must be positive.")

    if correlation_length <= 0.0:
        raise ValueError(
            "correlation_length must be positive."
        )

    frequencies = np.empty(
        num_modes,
        dtype=float,
    )

    for n in range(num_modes):
        left = 1.0e-12 if n == 0 else n * np.pi
        right = (n + 1) * np.pi

        root = cast(
            float,
            brentq(
                characteristic_equation,
                left,
                right,
                args=(correlation_length,),
                xtol=1.0e-14,
                rtol=np.float64(1.0e-13),
                full_output=False,
            ),
        )

        frequencies[n] = root

    return frequencies

def calculate_normalization_constants(
    frequencies: np.ndarray,
    correlation_length: float,
) -> np.ndarray:
    """
    Calculate positive constants A_n such that

        integral_0^1 b_n(x)^2 dx = 1,

    where

        b_n(x) = A_n * (
            sin(omega_n*x)
            + ell*omega_n*cos(omega_n*x)
        ).
    """
    omega = np.asarray(
        frequencies,
        dtype=float,
    )

    ell = correlation_length

    if omega.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if np.any(omega <= 0.0):
        raise ValueError(
            "frequencies must all be positive."
        )

    if ell <= 0.0:
        raise ValueError(
            "correlation_length must be positive."
        )

    norm_squared = (
        0.5 * (1.0 + ell**2 * omega**2)
        + (
            (ell**2 * omega**2 - 1.0)
            * np.sin(2.0 * omega)
            / (4.0 * omega)
        )
        + ell * np.sin(omega) ** 2
    )

    return 1.0 / np.sqrt(norm_squared)


def calculate_1d_eigenvalues(
    frequencies: np.ndarray,
    correlation_length: float,
) -> np.ndarray:
    """
    Calculate the 1D covariance eigenvalues associated with
    the supplied frequencies.
    """

    omega = np.asarray(
        frequencies,
        dtype=float,
    )

    ell = correlation_length

    if omega.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if ell <= 0.0:
        raise ValueError(
            "correlation_length must be positive."
        )

    return (
        2.0 * ell
        / (1.0 + ell**2 * omega**2)
    )


def make_1d_eigenfunction_evaluator(
    frequencies: np.ndarray,
    normalizations: np.ndarray,
    correlation_length: float,
) -> Callable[[np.ndarray | float], np.ndarray]:
    """
    Construct a function that evaluates all normalized 1D eigenfunctions.

    The returned function satisfies

        values[..., n] = b_n(x),

    where

        b_n(x) = A_n * (
            sin(omega_n*x)
            + ell*omega_n*cos(omega_n*x)
        ).
    """
    omega = np.array(
        frequencies,
        dtype=float,
        copy=True,
    )

    A = np.array(
        normalizations,
        dtype=float,
        copy=True,
    )

    ell = float(correlation_length)

    if omega.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if A.shape != omega.shape:
        raise ValueError(
            "normalizations and frequencies must have "
            "the same shape."
        )

    if ell <= 0.0:
        raise ValueError(
            "correlation_length must be positive."
        )

    def evaluate(x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        arguments = x[..., None] * omega

        return A * (
            np.sin(arguments)
            + ell * omega * np.cos(arguments)
        )

    return evaluate

def get_1d_eigenpairs(
    num_modes: int,
    correlation_length: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Callable[[np.ndarray | float], np.ndarray],
]:
    """
    Construct the analytical 1D covariance eigenpairs.

    Returns
    -------
    frequencies:
        Frequencies omega_n.

    normalizations:
        Normalization constants A_n.

    eigenvalues:
        Covariance eigenvalues theta_n.

    evaluator:
        Function that evaluates all b_n(x).
    """

    frequencies = calculate_frequencies(
        num_modes,
        correlation_length,
    )

    normalizations = calculate_normalization_constants(
        frequencies,
        correlation_length,
    )

    eigenvalues = calculate_1d_eigenvalues(
        frequencies,
        correlation_length
    )

    evaluator = make_1d_eigenfunction_evaluator(
        frequencies,
        normalizations,
        correlation_length,
    )

    return (
        frequencies,
        normalizations,
        eigenvalues,
        evaluator,
    )

def leading_2d_eigenvalues_outer(
    eigenvalues_1d: np.ndarray,
    num_modes_2d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the leading 2D tensor-product eigenvalues using
    a dense outer-product matrix.

    Returns
    -------
    eigenvalues_2d:
        Leading 2D eigenvalues in decreasing order.

    mode_indices:
        Corresponding zero-based 1D index pairs (i, j).


    Use `np.argpartition` to identify the leading K tensor-product
    eigenvalues without sorting every product.

    Setting `kth=-K` partitions relative to the K-th position from the
    end, which is equivalent to the ascending-order position N-K, where
    N is the total number of products. After partitioning, the final K
    positions contain the K largest values, but they are not necessarily
    ordered. The slice `[-K:]` extracts their indices; a subsequent
    `argsort` orders the selected values from largest to smallest.

    """
    eigenvalues_1d = np.asarray(
        eigenvalues_1d,
        dtype=float,
    )

    if eigenvalues_1d.ndim != 1:
        raise ValueError(
            "eigenvalues_1d must be one-dimensional."
        )

    if len(eigenvalues_1d) == 0:
        raise ValueError(
            "At least one 1D eigenvalue is required."
        )

    if np.any(eigenvalues_1d < 0.0):
        raise ValueError(
            "The 1D eigenvalues must be nonnegative."
        )

    if np.any(np.diff(eigenvalues_1d) > 0.0):
        raise ValueError(
            "The 1D eigenvalues must be in decreasing order."
        )

    maximum_modes_2d = len(eigenvalues_1d) ** 2

    if not 1 <= num_modes_2d <= maximum_modes_2d:
        raise ValueError(
            "num_modes_2d must be between 1 and "
            f"{maximum_modes_2d}."
        )

    products = np.outer(
        eigenvalues_1d,
        eigenvalues_1d,
    )

    flat_products = products.ravel()

    # Identify the indices of the K largest products.
    leading_flat_indices = np.argpartition(
        flat_products,
        -num_modes_2d,
    )[-num_modes_2d:]

    # Sort only those K products in decreasing order.
    order = np.argsort(
        flat_products[leading_flat_indices]
    )[::-1]

    leading_flat_indices = leading_flat_indices[order]

    eigenvalues_2d = flat_products[leading_flat_indices]

    i, j = np.unravel_index(
        leading_flat_indices,
        products.shape,
    )

    mode_indices = np.column_stack((i, j))

    return eigenvalues_2d, mode_indices

def leading_2d_eigenvalues_heap(
    eigenvalues_1d: np.ndarray,
    num_modes_2d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the leading 2D tensor-product eigenvalues and their indices.

    The 2D eigenvalues are

        theta_ij_2d = theta_i_1d * theta_j_1d.

    Because the 1D eigenvalues are in decreasing order, the
    tensor-product matrix decreases as either i or j increases.

    The heap stores

        (-theta_ij, i, j).

    Python's `heapq` removes the smallest value, so storing the
    negative product causes the largest positive eigenvalue to be
    removed first.

    Parameters
    ----------
    eigenvalues_1d:
        Positive 1D eigenvalues in decreasing order.

    num_modes_2d:
        Number of leading 2D modes to return.

    Returns
    -------
    eigenvalues_2d:
        Leading 2D eigenvalues in decreasing order.

    mode_indices:
        Corresponding zero-based index pairs (i, j). The 2D
        eigenfunction for a pair is

            b_ij(x1, x2) = b_i(x1) * b_j(x2).
    """
    eigenvalues_1d = np.asarray(
        eigenvalues_1d,
        dtype=float,
    )

    if eigenvalues_1d.ndim != 1:
        raise ValueError(
            "eigenvalues_1d must be one-dimensional."
        )

    if len(eigenvalues_1d) == 0:
        raise ValueError(
            "At least one 1D eigenvalue is required."
        )

    if np.any(eigenvalues_1d < 0.0):
        raise ValueError(
            "The 1D eigenvalues must be nonnegative."
        )

    if np.any(np.diff(eigenvalues_1d) > 0.0):
        raise ValueError(
            "The 1D eigenvalues must be in decreasing order."
        )

    maximum_modes_2d = len(eigenvalues_1d) ** 2

    if not 1 <= num_modes_2d <= maximum_modes_2d:
        raise ValueError(
            "num_modes_2d must be between 1 and "
            f"{maximum_modes_2d}."
        )

    number_of_1d_modes = len(eigenvalues_1d)

    heap: list[tuple[float, int, int]] = []

    initial_product = (
        eigenvalues_1d[0]
        * eigenvalues_1d[0]
    )

    heapq.heappush(
        heap,
        (
            -float(initial_product),
            0,
            0,
        ),
    )

    # Pairs are added to this set when they enter the heap.
    # This prevents a pair such as (1,1) from being added twice.
    visited: set[tuple[int, int]] = {
        (0, 0)
    }

    eigenvalues_2d = np.empty(
        num_modes_2d,
        dtype=float,
    )

    mode_indices = np.empty(
        (num_modes_2d, 2),
        dtype=int,
    )

    for mode in range(num_modes_2d):
        # Remove the smallest negative number, which represents
        # the largest remaining positive eigenvalue.
        negative_eigenvalue, i, j = heapq.heappop(heap)

        eigenvalues_2d[mode] = -negative_eigenvalue
        mode_indices[mode] = (i, j)

        # The product matrix decreases as either index increases.
        neighbors = (
            (i + 1, j),
            (i, j + 1),
        )

        for candidate_i, candidate_j in neighbors:
            candidate = (
                candidate_i,
                candidate_j,
            )

            inside_matrix = (
                candidate_i < number_of_1d_modes
                and candidate_j < number_of_1d_modes
            )

            if inside_matrix and candidate not in visited:
                visited.add(candidate)

                product = (
                    eigenvalues_1d[candidate_i]
                    * eigenvalues_1d[candidate_j]
                )

                heapq.heappush(
                    heap,
                    (
                        -float(product),
                        candidate_i,
                        candidate_j,
                    ),
                )

    return eigenvalues_2d, mode_indices

def make_2d_eigenfunction_evaluator(
    frequencies: np.ndarray,
    normalizations: np.ndarray,
    correlation_length: float,
    mode_indices: np.ndarray,
) -> Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]:
    """
    Create an evaluator for selected 2D tensor-product eigenfunctions.

    For mode n with indices (i, j),

        b_n(x, y) = b_i(x) * b_j(y),

    where

        b_i(x) = A_i [
            sin(omega_i*x)
            + ell*omega_i*cos(omega_i*x)
        ].

    The final output dimension is the 2D mode index:

        result[..., n] = b_i(x) * b_j(y).
    """
    omega = np.asarray(
        frequencies,
        dtype=float,
    ).copy()

    A = np.asarray(
        normalizations,
        dtype=float,
    ).copy()

    indices = np.asarray(
        mode_indices,
        dtype=int,
    ).copy()

    ell = float(correlation_length)

    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(
            "mode_indices must have shape (num_modes_2d, 2)."
        )

    if np.any(indices < 0) or np.any(indices >= len(omega)):
        raise ValueError(
            "mode_indices contains an invalid 1D mode index."
        )

    if len(A) != len(omega):
        raise ValueError(
            "frequencies and normalizations must have the same length."
        )

    i = indices[:, 0]
    j = indices[:, 1]

    omega_i = omega[i]
    omega_j = omega[j]
    A_i = A[i]
    A_j = A[j]

    def evaluate(
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)

        # Make x and y broadcast-compatible before adding the mode axis.
        x_array, y_array = np.broadcast_arrays(
            x_array,
            y_array,
        )

        arguments_x = x_array[..., None] * omega_i
        arguments_y = y_array[..., None] * omega_j

        eigenfunctions_x = A_i * (
            np.sin(arguments_x)
            + ell * omega_i * np.cos(arguments_x)
        )

        eigenfunctions_y = A_j * (
            np.sin(arguments_y)
            + ell * omega_j * np.cos(arguments_y)
        )

        return eigenfunctions_x * eigenfunctions_y

    return evaluate

def leading_2d_eigenpairs(
    eigenvalues_1d: np.ndarray,
    frequencies_1d: np.ndarray,
    normalizations_1d: np.ndarray,
    correlation_length: float,
    num_modes_2d: int,
    method: Literal["outer", "heap"] = "outer",
) -> tuple[
    np.ndarray,
    np.ndarray,
    Callable[
        [np.ndarray | float, np.ndarray | float],
        np.ndarray,
    ],
]:
    """
    Construct the leading 2D tensor-product eigenpairs.

    Returns
    -------
    eigenvalues_2d:
        Leading 2D eigenvalues in decreasing order.

    mode_indices:
        Corresponding 1D index pairs (i, j).

    evaluator:
        Function satisfying

            values[..., n] = b_i(x) * b_j(y),

        where (i, j) = mode_indices[n].
    """
    if method == "outer":
        eigenvalues_2d, mode_indices = (
            leading_2d_eigenvalues_outer(
                eigenvalues_1d,
                num_modes_2d,
            )
        )
    elif method == "heap":
        eigenvalues_2d, mode_indices = (
            leading_2d_eigenvalues_heap(
                eigenvalues_1d,
                num_modes_2d,
            )
        )
    else:
        raise ValueError(
            f"Invalid method {method!r}. "
            "Expected 'outer' or 'heap'."
        )

    evaluator = make_2d_eigenfunction_evaluator(
        frequencies_1d,
        normalizations_1d,
        correlation_length,
        mode_indices,
    )

    return (
        eigenvalues_2d,
        mode_indices,
        evaluator,
    )



def make_2d_kl_evaluator(
    eigenvalues_2d: np.ndarray,
    eigenfunction_evaluator: Callable[
        [np.ndarray | float, np.ndarray | float],
        np.ndarray,
    ],
    mean_log_conductivity: float = 0.0,
    variance: float = 1.0
) -> Callable[
    [np.ndarray | float, np.ndarray | float, np.ndarray],
    np.ndarray,
]:
    """
    Construct an evaluator for a truncated 2D KL expansion.

    For standard Gaussian coefficients xi_n, the returned evaluator
    computes

        Z(x, y) = mean_log_conductivity
                  + sum_n sqrt(theta_n) * xi_n * b_n(x, y).

    This is the Gaussian log-conductivity field. The corresponding
    lognormal conductivity is

        k(x, y) = exp(Z(x, y)).
    """
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError(
            "variance must be finite and nonnegative."
        )

    theta = variance * np.array(
        eigenvalues_2d,
        dtype=float,
        copy=True,
    )

    if theta.ndim != 1:
        raise ValueError(
            "eigenvalues_2d must be one-dimensional."
        )

    if np.any(theta < 0.0):
        raise ValueError(
            "eigenvalues_2d must be nonnegative."
        )

    square_root_theta = np.sqrt(theta)
    mean = float(mean_log_conductivity)

    
    # def evaluate(
    #     x: np.ndarray | float,
    #     y: np.ndarray | float,
    #     coefficients: np.ndarray,
    # ) -> np.ndarray:
    #     xi = np.asarray(
    #         coefficients,
    #         dtype=float,
    #     )

    #     if xi.shape != theta.shape:
    #         raise ValueError(
    #             "coefficients must have shape "
    #             f"{theta.shape}, but received {xi.shape}."
    #         )

    #     eigenfunctions = eigenfunction_evaluator(
    #         x,
    #         y,
    #     )

    #     # eigenfunctions[..., n] is b_n(x, y).
    #     # The sum removes the final mode dimension.
    #     return (
    #         mean
    #         + np.sum(
    #             eigenfunctions
    #             * square_root_theta
    #             * xi,
    #             axis=-1,
    #         )
    #     )
    cached_x = None
    cached_y = None
    cached_B = None

    def evaluate(
        x: np.ndarray | float,
        y: np.ndarray | float,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        xi = np.asarray(
            coefficients,
            dtype=float,
        )

        if xi.shape != theta.shape:
            raise ValueError(
                "coefficients must have shape "
                f"{theta.shape}, but received {xi.shape}."
            )
        nonlocal cached_x, cached_y, cached_B
        
        if cached_B is None or x is not cached_x or y is not cached_y:
            cached_B = eigenfunction_evaluator(x, y)
            cached_x = x
            cached_y = y
        return mean + np.sum(
            cached_B * square_root_theta * xi,
            axis=-1,
        )

    return evaluate

    

if __name__ == "__main__":
    correlation_length = 0.2
    variance = 1.0 
    num_modes = 1000

    (
    frequencies_1d,
    normalizations_1d,
    eigenvalues_1d,
    evaluate_1d,
    ) = get_1d_eigenpairs(
        num_modes= num_modes,
        correlation_length=correlation_length,
    )

    (
        eigenvalues_2d,
        mode_indices,
        evaluate_eigenfunctions_2d,
    ) = leading_2d_eigenpairs(
        eigenvalues_1d=eigenvalues_1d,
        frequencies_1d=frequencies_1d,
        normalizations_1d=normalizations_1d,
        correlation_length=correlation_length,
        num_modes_2d=num_modes,
        method="heap",
    )

    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    values_2d = evaluate_eigenfunctions_2d(X, Y)

    print(values_2d.shape)

    evaluate_log_conductivity = make_2d_kl_evaluator(
    eigenvalues_2d,
    evaluate_eigenfunctions_2d,
    mean_log_conductivity=0.0,
    variance=variance,
    )

    rng = np.random.default_rng(seed=1234)

    # One independent N(0, 1) sample for every retained eigenfunction.
    xi = rng.standard_normal(
        len(eigenvalues_2d)
    )

    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    Z = evaluate_log_conductivity(
        X,
        Y,
        xi,
    )

    # Lognormal conductivity used in the PDE.
    conductivity = np.exp(Z)

# %%


# """
# HEAP VISUALIZATION

# """



# from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch
# from matplotlib.widgets import Button

# # In a Jupyter notebook, run `%matplotlib widget` in a cell before executing
# # this file.  IPython magic is intentionally not included here because it is
# # invalid Python syntax in an importable `.py` module.

# # ------------------------------------------------------------
# # 1. Dummy one-dimensional eigenvalues
# # ------------------------------------------------------------

# eigenvalues_1d = np.array([
#     0.60,
#     0.25,
#     0.10,
#     0.04,
# ])

# print(f"eigenvalues_1d: {eigenvalues_1d}")
# number_of_1d_modes = len(eigenvalues_1d)

# # Every matrix entry is theta_i * theta_j.
# product_matrix = np.outer(
#     eigenvalues_1d,
#     eigenvalues_1d,
# )


# # ------------------------------------------------------------
# # 2. Run the heap algorithm and save every step
# # ------------------------------------------------------------

# heap = []
# visited = {(0, 0)}
# removed = set()

# initial_product = (
#     eigenvalues_1d[0]
#     * eigenvalues_1d[0]
# )

# heapq.heappush(
#     heap,
#     (-float(initial_product), 0, 0),
# )

# # Each snapshot stores the state after one loop.
# snapshots = [
#     {
#         "current": None,
#         "added": set(),
#         "heap": {(0, 0)},
#         "removed": set(),
#     }
# ]

# for loop_number in range(number_of_1d_modes**2):
#     # Remove the smallest negative value, corresponding to the
#     # largest positive tensor-product eigenvalue.
#     negative_value, i, j = heapq.heappop(heap)

#     current = (i, j)
#     removed.add(current)

#     neighbors = (
#         (i + 1, j),
#         (i, j + 1),
#     )

#     added_this_loop = set()

#     for candidate_i, candidate_j in neighbors:
#         candidate = (
#             candidate_i,
#             candidate_j,
#         )

#         inside_matrix = (
#             candidate_i < number_of_1d_modes
#             and candidate_j < number_of_1d_modes
#         )

#         if inside_matrix and candidate not in visited:
#             visited.add(candidate)
#             added_this_loop.add(candidate)

#             product = (
#                 eigenvalues_1d[candidate_i]
#                 * eigenvalues_1d[candidate_j]
#             )

#             heapq.heappush(
#                 heap,
#                 (
#                     -float(product),
#                     candidate_i,
#                     candidate_j,
#                 ),
#             )

#     heap_pairs = {
#         (heap_i, heap_j)
#         for _, heap_i, heap_j in heap
#     }

#     snapshots.append(
#         {
#             "current": current,
#             "current_value": -negative_value,
#             "added": added_this_loop,
#             "heap": heap_pairs,
#             "removed": removed.copy(),
#         }
#     )


# # ------------------------------------------------------------
# # 3. Set up the colors
# # ------------------------------------------------------------

# # Matrix state values:
# #
# # 0 = not discovered
# # 1 = removed during an earlier loop
# # 2 = currently in the heap
# # 3 = added during the current loop
# # 4 = removed during the current loop

# colors = [
#     "#f2f2f2",  # not discovered
#     "#bdbdbd",  # removed earlier
#     "#6baed6",  # currently in heap
#     "#74c476",  # newly added
#     "#fb6a4a",  # currently removed
# ]

# color_map = ListedColormap(colors)


# # ------------------------------------------------------------
# # 4. Create the figure
# # ------------------------------------------------------------

# figure, axis = plt.subplots(figsize=(8, 7))

# # Leave room at the bottom for the buttons.
# plt.subplots_adjust(bottom=0.22)

# step_state = {
#     "step": 0,
# }


# def draw_current_step():
#     """Redraw the matrix for the selected snapshot."""
#     axis.clear()

#     step = step_state["step"]
#     snapshot = snapshots[step]

#     matrix_state = np.zeros(
#         (number_of_1d_modes, number_of_1d_modes),
#         dtype=int,
#     )

#     # Mark modes removed during earlier loops.
#     for i, j in snapshot["removed"]:
#         matrix_state[i, j] = 1

#     # Mark modes currently waiting in the heap.
#     for i, j in snapshot["heap"]:
#         matrix_state[i, j] = 2

#     # Newly added modes are also in the heap, but they receive
#     # their own color for this step.
#     for i, j in snapshot["added"]:
#         matrix_state[i, j] = 3

#     # The current mode was removed from the heap this loop.
#     if snapshot["current"] is not None:
#         current_i, current_j = snapshot["current"]
#         matrix_state[current_i, current_j] = 4

#     axis.imshow(
#         matrix_state,
#         cmap=color_map,
#         vmin=0,
#         vmax=4,
#     )

#     # Write the index pair and product inside each matrix cell.
#     for i in range(number_of_1d_modes):
#         for j in range(number_of_1d_modes):
#             value = product_matrix[i, j]

#             axis.text(
#                 j,
#                 i,
#                 f"({i},{j})\n{value:.4f}",
#                 ha="center",
#                 va="center",
#                 fontsize=10,
#             )

#     axis.set_xticks(
#         range(number_of_1d_modes),
#         labels=[
#             f"j={j}"
#             for j in range(number_of_1d_modes)
#         ],
#     )

#     axis.set_yticks(
#         range(number_of_1d_modes),
#         labels=[
#             f"i={i}"
#             for i in range(number_of_1d_modes)
#         ],
#     )

#     axis.set_xlabel("Second 1D eigenvalue index j")
#     axis.set_ylabel("First 1D eigenvalue index i")

#     if step == 0:
#         title = (
#             "Before entering the loop\n"
#             "The heap initially contains (0,0)"
#         )
#     else:
#         current_i, current_j = snapshot["current"]
#         current_value = snapshot["current_value"]

#         added_text = (
#             ", ".join(
#                 str(pair)
#                 for pair in sorted(snapshot["added"])
#             )
#             if snapshot["added"]
#             else "none"
#         )

#         heap_text = (
#             ", ".join(
#                 str(pair)
#                 for pair in sorted(snapshot["heap"])
#             )
#             if snapshot["heap"]
#             else "empty"
#         )

#         title = (
#             f"Loop {step}: removed "
#             f"({current_i},{current_j}) = "
#             f"{current_value:.4f}\n"
#             f"Added: {added_text}    "
#             f"Heap now: {heap_text}"
#         )

#     axis.set_title(title)

#     axis.set_xticks(
#         np.arange(-0.5, number_of_1d_modes, 1.0),
#         minor=True,
#     )

#     axis.set_yticks(
#         np.arange(-0.5, number_of_1d_modes, 1.0),
#         minor=True,
#     )

#     axis.grid(
#         which="minor",
#         color="black",
#         linewidth=1.5,
#     )

#     axis.tick_params(
#         which="minor",
#         bottom=False,
#         left=False,
#     )

#     figure.canvas.draw_idle()


# # ------------------------------------------------------------
# # 5. Add a legend
# # ------------------------------------------------------------

# legend_items = [
#     Patch(
#         facecolor=colors[4],
#         edgecolor="black",
#         label="Removed this loop",
#     ),
#     Patch(
#         facecolor=colors[3],
#         edgecolor="black",
#         label="Added this loop",
#     ),
#     Patch(
#         facecolor=colors[2],
#         edgecolor="black",
#         label="Currently in heap",
#     ),
#     Patch(
#         facecolor=colors[1],
#         edgecolor="black",
#         label="Removed earlier",
#     ),
#     Patch(
#         facecolor=colors[0],
#         edgecolor="black",
#         label="Not discovered",
#     ),
# ]

# figure.legend(
#     handles=legend_items,
#     loc="lower center",
#     ncol=3,
#     bbox_to_anchor=(0.5, 0.09),
# )


# # ------------------------------------------------------------
# # 6. Add Previous, Next, and Reset buttons
# # ------------------------------------------------------------

# previous_button_axis = plt.axes([
#     0.20,
#     0.02,
#     0.18,
#     0.055,
# ])

# next_button_axis = plt.axes([
#     0.41,
#     0.02,
#     0.18,
#     0.055,
# ])

# reset_button_axis = plt.axes([
#     0.62,
#     0.02,
#     0.18,
#     0.055,
# ])

# previous_button = Button(
#     previous_button_axis,
#     "Previous",
# )

# next_button = Button(
#     next_button_axis,
#     "Next loop",
# )

# reset_button = Button(
#     reset_button_axis,
#     "Reset",
# )


# def show_previous_step(event):
#     if step_state["step"] > 0:
#         step_state["step"] -= 1
#         draw_current_step()


# def show_next_step(event):
#     if step_state["step"] < len(snapshots) - 1:
#         step_state["step"] += 1
#         draw_current_step()


# def reset_steps(event):
#     step_state["step"] = 0
#     draw_current_step()


# previous_button.on_clicked(show_previous_step)
# next_button.on_clicked(show_next_step)
# reset_button.on_clicked(reset_steps)


# # ------------------------------------------------------------
# # 7. Draw the initial state
# # ------------------------------------------------------------

# draw_current_step()
# plt.show()
