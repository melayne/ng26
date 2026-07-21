#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ============================================================
# 1. Build a 2D grid on [0, 1] x [0, 1]
# ============================================================

def make_2d_grid(nx, ny):
    """
    Creates a 2D grid on [0,1]^2.

    Returns
    -------
    X, Y : 2D arrays of shape (ny, nx)
        Meshgrid coordinates for plotting.

    points : array of shape (nx*ny, 2)
        Flattened list of spatial coordinates.
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    X, Y = np.meshgrid(x, y, indexing="xy")

    points = np.column_stack([X.ravel(), Y.ravel()])

    return X, Y, points


# ============================================================
# 2. Exponential covariance kernel in 2D
# ============================================================

def exponential_covariance_2d(points, sigma=1.0, ell=0.2):
    """
    Builds the covariance matrix

        C_ij = sigma^2 exp(-||x_i - x_j|| / ell)

    Parameters
    ----------
    points : array of shape (N, 2)
        Grid points.

    sigma : float
        Standard deviation of the random field.

    ell : float
        Correlation length.

    Returns
    -------
    C : array of shape (N, N)
        Covariance matrix.
    """
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # distances = np.sum(np.abs(diff), axis=2)
    C = sigma**2 * np.exp(-distances / ell)

    return C


# ============================================================
# 3. Compute KL eigenpairs
# ============================================================

def compute_kl_modes(C, num_modes):
    """
    Computes the leading KL eigenvalues and eigenvectors.

    For the discrete covariance matrix,

        C q_i = lambda_i q_i

    Parameters
    ----------
    C : array of shape (N, N)
        Covariance matrix.

    num_modes : int
        Number of KL modes to keep.

    Returns
    -------
    eigenvalues : array of shape (num_modes,)
        Leading eigenvalues.

    eigenvectors : array of shape (N, num_modes)
        Leading eigenvectors.
    """
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = eigh(C)

    # Reverse to descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Keep only the leading modes
    eigenvalues = eigenvalues[:num_modes]
    eigenvectors = eigenvectors[:, :num_modes]

    return eigenvalues, eigenvectors


# ============================================================
# 4. Sample from the truncated KL expansion
# ============================================================

def sample_kl_expansion(mean, eigenvalues, eigenvectors, nx, ny, rng=None):
    """
    Samples a Gaussian random field using the truncated KL expansion

        Z(x) ≈ mean(x) + sum_{i=1}^m sqrt(lambda_i) xi_i q_i(x)

    where xi_i ~ N(0, 1).

    Parameters
    ----------
    mean : array of shape (N,)
        Mean function evaluated at grid points.

    eigenvalues : array of shape (m,)
        KL eigenvalues.

    eigenvectors : array of shape (N, m)
        KL eigenvectors.

    nx, ny : int
        Grid dimensions.

    rng : numpy random generator
        Optional random number generator.

    Returns
    -------
    field : array of shape (ny, nx)
        Sampled random field on the 2D grid.

    xi : array of shape (m,)
        Random standard normal KL coefficients.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_modes = len(eigenvalues)

    xi = rng.standard_normal(num_modes)

    sample_flat = mean + eigenvectors @ (np.sqrt(eigenvalues) * xi)

    field = sample_flat.reshape(ny, nx)
    exp_field = np.exp(field)

    return exp_field, xi


# ============================================================
# 5. Main example
# ============================================================

if __name__ == "__main__":

    # Grid resolution
    nx = 40
    ny = 40
    h = 1.0 / (nx - 1)
    # Covariance parameters
    sigma = 1.0
    ell = 0.1

    # Number of KL modes to keep
    num_modes = 200

    # Random seed for reproducibility
    rng = np.random.default_rng(123)

    # Build grid
    X, Y, points = make_2d_grid(nx, ny)
    N = points.shape[0]

    # Mean function
    mean = np.zeros(N)

    # Build covariance matrix
    C = exponential_covariance_2d(points, sigma=sigma, ell=ell)

    # Compute KL modes
    eigenvalues, eigenvectors = compute_kl_modes(C, num_modes=num_modes)

    # Sample random field
    field, xi = sample_kl_expansion(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        nx=nx,
        ny=ny,
        rng=rng,
    )

    # ========================================================
    # Plot eigenvalue decay
    # ========================================================

    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(eigenvalues)), eigenvalues, marker="o")
    plt.xlabel("Mode index")
    plt.ylabel("Eigenvalue")
    plt.title("KL Eigenvalue Decay")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # Plot one sampled field
    # ========================================================

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, field, levels=40, cmap="jet")
    plt.colorbar(label="Random field value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D KL Sample: Exponential Covariance, {num_modes} Modes")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # ========================================================
    # Plot several samples
    # ========================================================

    n_samples = 4

    fig, axes = plt.subplots(1, n_samples, figsize=(14, 3.5))

    for k in range(n_samples):
        field_k, _ = sample_kl_expansion(
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            nx=nx,
            ny=ny,
            rng=rng,
        )

        ax = axes[k]
        im = ax.contourf(X, Y, field_k, levels=40, cmap="jet")
        ax.set_title(f"Sample {k + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle(f"2D KL Samples with {num_modes} Modes")
    plt.tight_layout()
    plt.show()
# %%
