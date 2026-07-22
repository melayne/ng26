# Discrete KL Expansion with a Two-Level NGSolve Solver

This guide explains the complete workflow implemented by:
 
- `src/KL_expansion.py`
- `examples/kl_two_level.py`
- the hierarchy and V-cycle routines in `src/multigrid_cycles.py`

The example generates one random diffusion coefficient with a discrete
Karhunen--Loève (KL) expansion, uses that same coefficient on a coarse and a
fine finite-element mesh, solves the resulting elliptic problem with repeated
two-level V-cycles, and plots the stochastic field, meshes, convergence, and
solution.

## 1. The mathematical problem

The deterministic PDE for one realization is

$$
-\nabla\cdot\left(\kappa(x,\omega)\nabla u(x,\omega)\right)=1 \qquad \text{in }\Omega=[0,1]^2,
$$

with homogeneous Dirichlet boundary conditions

$$
u=0 \qquad \text{on }\partial\Omega.
$$

The diffusion coefficient is lognormal:

$$
\kappa(x,\omega)=\exp\bigl(Z(x,\omega)\bigr),
$$

where the Gaussian field is approximated by a truncated discrete KL
expansion:

$$
Z(x,\omega) \approx \mu(x)+\sum_{j=1}^{m}\sqrt{\lambda_j}q_j(x)\xi_j(\omega), \qquad \xi_j\sim N(0,1).
$$

Exponentiation guarantees

$$
\kappa(x,\omega)>0,
$$

which is important for an elliptic diffusion problem.

## 2. Complete data flow

```text
Cartesian KL grid
       |
       v
Covariance matrix C_ij = Cov[Z(x_i), Z(x_j)]
       |
       v
Leading eigenpairs C q_j = lambda_j q_j
       |
       v
One Gaussian sample Z = mean + Q sqrt(Lambda) xi
       |
       v
Positive coefficient kappa = exp(Z)
       |
       v
NGSolve VoxelCoefficient kappa(x,y)
       |
       +------------------------------+
       |                              |
       v                              v
Coarse stiffness matrix A_H     Fine stiffness matrix A_h
       |                              |
       +---------- two-level V-cycle -+
                                      |
                                      v
                              Fine-grid solution u_h
```

The KL coefficient is sampled once. It is not resampled when the hierarchy
moves between levels.

## 3. Step 1: build the Cartesian KL grid

The example starts with:

```python
nx = ny = 16
X, Y, points = cartesian_grid_2d(nx, ny)
```

The outputs are:

| Name     | Shape        | Meaning                               |
| -------- | ------------ | ------------------------------------- |
| `X`      | `(ny, nx)`   | x-coordinate at every KL grid point   |
| `Y`      | `(ny, nx)`   | y-coordinate at every KL grid point   |
| `points` | `(nx*ny, 2)` | flattened list of `(x,y)` coordinates |

`X` and `Y` are convenient for Matplotlib. `points` is convenient for
building the covariance matrix.

With `indexing="xy"`, the x-coordinate varies fastest when the arrays are
flattened. This convention is preserved when a sampled vector is reshaped
back to `(ny, nx)`.

## 4. Step 2: build the discrete covariance matrix

The example uses an isotropic exponential covariance kernel:

$$
C(x_i,x_j) = \sigma^2 \exp\left(-\frac{\lVert x_i-x_j\rVert_2}{\ell}\right).
$$

In code:

```python
covariance = exponential_covariance(
    points,
    sigma=1.0,
    correlation_length=0.30,
)
```

The parameters have different effects:

- `sigma` controls the standard deviation of the Gaussian field `Z`.
- `correlation_length` controls the spatial smoothness and feature size.
- Larger correlation length gives broad, smooth structures.
- Smaller correlation length gives shorter-scale variation and slower KL
eigenvalue decay.

If there are `N = nx*ny` grid points, the covariance matrix has shape
`(N,N)`. It is dense, so increasing the KL grid resolution can become
expensive in memory and eigensolver time.

## 5. Step 3: compute the leading discrete KL modes

The module solves

$$
Cq_j=\lambda_jq_j.
$$

In code:

```python
eigenvalues, eigenvectors = leading_eigenpairs(
    covariance,
    num_modes=12,
)
```

If `N=nx*ny` and `m=num_modes`, then:

| Name           | Shape   | Meaning                                  |
| -------------- | ------- | ---------------------------------------- |
| `eigenvalues`  | `(m,)`  | retained eigenvalues in descending order |
| `eigenvectors` | `(N,m)` | one discrete mode in each column         |

Large eigenvalues identify directions containing more variance. The retained
discrete variance fraction is

$$
\frac{\sum_{j=1}^{m}\lambda_j}{\operatorname{trace}(C)}.
$$

`trace(C)` is the total variance of the full discrete Gaussian vector.

### Why this eigenproblem has no mass matrix

This module treats

$$
(Z(x_1),\ldots,Z(x_N))
$$

as a finite Gaussian random vector with covariance matrix `C`. Therefore the
ordinary matrix eigendecomposition is appropriate.

This is different from discretizing the continuous covariance integral
operator with finite elements. A finite-element KL method would normally
produce a generalized eigenproblem such as

$$
Kq=\lambda Mq,
$$

where `M` is the finite-element mass matrix. That is a possible later
extension, but it is not what the present discrete Cartesian-grid module
computes.

## 6. Step 4: sample one truncated KL realization

The example calls:

```python
log_kappa_values, xi = sample_discrete_kl(
    mean=0.0,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    shape=(ny, nx),
    rng=7,
)
```

Internally, the module generates

```python
xi = generator.standard_normal(num_modes)
```

and evaluates

```python
gaussian_flat = (
    mean_flat
    + eigenvectors @ (np.sqrt(eigenvalues) * xi)
)
```

This is the matrix form of the truncated KL expansion:

$$
Z=\mu+Q\Lambda^{1/2}\xi.
$$

The integer seed `rng=7` makes the realization reproducible. Changing the
seed changes `xi` and therefore changes the random field, while leaving the
covariance model and KL basis unchanged.

The returned `log_kappa_values` array has shape `(ny,nx)`.

## 7. Step 5: transform to a positive diffusion coefficient

The Gaussian field is exponentiated:

```python
kappa_values = lognormal_transform(log_kappa_values)
```

or equivalently

```python
kappa_values = np.exp(log_kappa_values)
```

The names are important:

- `log_kappa_values` contains the Gaussian field `Z`.
- `kappa_values` contains the positive lognormal field `exp(Z)`.

Because the exponential is nonlinear, the mean of `kappa` is not simply the
exponential of the mean of `Z`.

## 8. Step 6: convert the array into an NGSolve coefficient

The KL calculation produces a NumPy array, but NGSolve's weak form requires a
function that can be evaluated at physical coordinates. The bridge is:

```python
kappa = voxel_coefficient_2d(kappa_values, linear=True)
```

This creates an NGSolve `VoxelCoefficient` over `[0,1]^2`.

In two dimensions, it is helpful to think of a `VoxelCoefficient` as a pixel
grid plus spatial interpolation. With `linear=True`, NGSolve interpolates the
stored values when it asks for `kappa(x,y)` at a point between KL grid nodes.

The finite-element nodes do not need to coincide with the Cartesian KL grid.
NGSolve usually evaluates `kappa` at quadrature points inside each element,
not merely at mesh nodes.

## 9. Step 7: insert the random coefficient into the weak form

For one realization, the weak form is: find `u` in the finite-element space
such that

$$
\int_\Omega \kappa(x,\omega)\nabla u\cdot\nabla v\,dx = \int_\Omega v\,dx \qquad \text{for every test function }v.
$$

The example expresses this as:

```python
def diffusion_form(a, u, v):
    a += kappa * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx

def load_form(f, u, v):
    f += v * ng.dx
```

The closure over `kappa` is significant: both hierarchy levels use the exact
same Python `kappa` object and therefore the same random realization.

Do not generate `xi` or create a new random field inside `diffusion_form`.
Doing that could assemble a different PDE on each level.

## 10. Step 8: construct exactly two finite-element levels

The initial mesh is generated by:

```python
coarse_mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.35))
```

The hierarchy is built with:

```python
hierarchy = build_hierarchy(
    coarse_mesh,
    form_setup,
    n_refines=1,
    order=1,
    dirichlet="left|right|top|bottom",
    dirichlet_value=0.0,
)
```

`n_refines=1` means:

```text
level 0: original coarse mesh
level 1: one uniform refinement of the coarse mesh
```

For each level, `build_hierarchy`:

1. Creates an `H1` finite-element space.
2. Calls `form_setup(fes)`.
3. Assembles that level's bilinear and linear forms.
4. Stores the stiffness matrix, load vector, and solution `GridFunction`.
5. Constructs prolongation `P` and its transpose `PT` between the levels.

The two stiffness matrices are rediscretizations of the same random PDE:

$$
(A_\ell)_{ij} = \int_\Omega \kappa(x,\omega) \nabla\phi_j^\ell\cdot\nabla\phi_i^\ell\,dx.
$$

The basis functions change with the level, but `kappa(x,omega)` does not.

## 11. Why the KL field is not projected between levels

`kappa` is not stored as a `GridFunction` belonging to the coarse or fine
finite-element space. It is an external coordinate-based
`VoxelCoefficient`.

Therefore the workflow is not:

```text
coarse KL coefficients -> project -> fine KL coefficients
```

Instead it is:

```text
one physical function kappa(x,y)
      |                         |
      v                         v
evaluated by coarse mesh   evaluated by fine mesh
```

The example verifies this at one physical point:

```python
probe = (0.37, 0.42)
probe_values = np.array([
    float(kappa(level.mesh(*probe)))
    for level in hierarchy.levels
])

np.testing.assert_allclose(probe_values, probe_values[0])
```

Both levels must return the same value because both evaluate the same
physical field at the same physical coordinate.

### When an L2 projection would be needed

If a KL mode or coefficient were stored as a fine-mesh finite-element
`GridFunction` and a coarse finite-element representation were required, then
an L2 projection would be appropriate:

$$
u_H=M_H^{-1}P^TM_hu_h.
$$

That is not required for the current `VoxelCoefficient` design.

## 12. Step 9: run a two-level V-cycle

The solver is configured by:

```python
solver = MultigridSolver(
    hierarchy,
    VCycleConfig(
        pre_sweeps=2,
        post_sweeps=2,
        coarse_direct=True,
    ),
)
```

One two-level V-cycle performs:

1. Pre-smooth the fine-grid iterate.
2. Compute the fine residual `r_h = b_h - A_h x_h`.
3. Restrict the residual with `r_H = P.T @ r_h`.
4. Solve the coarse correction equation `A_H e_H = r_H`.
5. Prolong the correction with `e_h = P @ e_H`.
6. Update `x_h <- x_h + e_h`.
7. Post-smooth the fine-grid iterate.

The two different transfers have different meanings:

| Quantity          | Transfer        | Reason                                    |
| ----------------- | --------------- | ----------------------------------------- |
| coarse correction | `e_h = P e_H`   | corrections are primal FE vectors         |
| fine residual     | `r_H = P.T r_h` | residuals represent linear functionals    |
| KL coefficient    | no transfer     | it is reevaluated in physical coordinates |

The residual restriction is not an L2 projection. An algebraic residual is a
dual object, so the variational restriction is `P.T`.

## 13. Step 10: repeat cycles until convergence

The solve call is:

```python
history, _ = solver.solve(
    max_cycles=12,
    tol=1.0e-8,
    norms=("l2",),
    verbose=True,
)
```

The stopping rule is relative:

$$
\lVert r_k\rVert_2 \leq \text{tol}\lVert r_0\rVert_2.
$$

The example stores the initial residual together with the post-cycle history:

```python
residuals = np.concatenate(
    ([initial_residual], history["l2"])
)
```

This makes cycle zero appear in the convergence plot.

## 14. Plotting workflow

The plotting functions do not participate in the KL calculation, form
assembly, or V-cycle. They only convert existing results into Matplotlib
figures.

### `mesh_triangulation`

```python
def mesh_triangulation(mesh):
    points = np.array([
        vertex.point[:2]
        for vertex in mesh.vertices
    ])

    triangles = np.array([
        [vertex.nr for vertex in element.vertices]
        for element in mesh.Elements(ng.VOL)
    ])

    return mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )
```

This converts NGSolve mesh geometry into the vertex and triangle arrays that
Matplotlib's `triplot` understands. It assumes a two-dimensional triangular
mesh.

### `overlay_mesh`

```python
ax.triplot(mesh_triangulation(mesh), ...)
```

This draws element edges over a contour plot. It does not interpolate or
alter the random field.

### `evaluate_on_grid`

```python
values = np.asarray(
    coefficient(mesh(X.ravel(), Y.ravel()))
)
return values.reshape(X.shape)
```

Matplotlib's `contourf` wants rectangular arrays. An NGSolve `GridFunction`
is instead defined through FE degrees of freedom on an unstructured mesh.
This helper:

1. Flattens the plotting coordinates.
2. Lets NGSolve locate those points in its mesh.
3. Evaluates the coefficient or `GridFunction` at the points.
4. Reshapes the values for `contourf`.

It is used to display the final FE solution. It is a visualization sample,
not a multigrid transfer.

### `make_field_figure`

This creates a 2-by-2 figure:

```text
+----------------------------+----------------------------+
| Gaussian field Z           | kappa = exp(Z)             |
+----------------------------+----------------------------+
| kappa + coarse mesh        | kappa + fine mesh          |
+----------------------------+----------------------------+
```

The same contour levels are used in all three `kappa` panels, so equal colors
represent equal coefficient values. The dots on the Gaussian panel show the
Cartesian KL sampling grid.

### `make_diagnostics_figure`

This creates three panels:

1. KL eigenvalues and cumulative retained variance.
2. Fine-grid residual versus V-cycle number.
3. Computed fine-grid solution with the fine mesh overlay.

## 15. Function reference

### Discrete KL module

| Function                 | Input                               | Output                    | Purpose                        |
| ------------------------ | ----------------------------------- | ------------------------- | ------------------------------ |
| `cartesian_grid_2d`      | grid dimensions, bounds             | `X`, `Y`, `points`        | Create the KL grid             |
| `exponential_covariance` | points, `sigma`, correlation length | dense `C`                 | Define spatial covariance      |
| `leading_eigenpairs`     | `C`, number of modes                | eigenvalues, eigenvectors | Compute dominant KL directions |
| `sample_discrete_kl`     | mean, eigenpairs, shape, RNG        | Gaussian grid, `xi`       | Draw one truncated sample      |
| `lognormal_transform`    | Gaussian values                     | positive values           | Construct `kappa=exp(Z)`       |
| `voxel_coefficient_2d`   | regular-grid values                 | NGSolve coefficient       | Bridge NumPy and NGSolve       |

### Example and plotting helpers

| Function                  | Purpose                                      |
| ------------------------- | -------------------------------------------- |
| `mesh_triangulation`      | Convert NGSolve mesh geometry for Matplotlib |
| `overlay_mesh`            | Draw FE edges over a field                   |
| `evaluate_on_grid`        | Sample an NGSolve function for `contourf`    |
| `finish_spatial_axis`     | Apply common axis labels and aspect ratio    |
| `make_field_figure`       | Plot `Z`, `kappa`, and both meshes           |
| `make_diagnostics_figure` | Plot spectrum, convergence, and solution     |
| `main`                    | Run the complete workflow                    |

## 16. Running the example

From the `ng26` project directory:

```bash
source .venv/bin/activate
python examples/kl_two_level.py
```

This runs the solver, saves the PNG files, and opens the plots.

To save without opening graphical windows:

```bash
python examples/kl_two_level.py --no-show
```

By default, the figures are saved as:

```text
examples/plots/kl_two_level_fields.png
examples/plots/kl_two_level_diagnostics.png
```

## 17. How to read the output

The hierarchy table reports:

- the number of degrees of freedom on each level;
- the stiffness-matrix dimensions;
- the prolongation-matrix dimensions;
- the free and Dirichlet degrees of freedom.

The cycle output reports:

```text
cycle k  l2 = residual  (rate residual_k/residual_{k-1})
```

A nearly constant rate below one indicates stable geometric convergence.

The remaining printed values mean:

- `KL coefficients xi`: the standard-normal random coordinates for this
realization;
- `retained discrete variance`: the fraction represented by the retained
modes;
- `kappa range`: the minimum and maximum on the Cartesian KL grid;
- `kappa(probe)`: confirmation that both levels see the same field;
- `final residual`: the algebraic residual after the last V-cycle.

## 18. Useful experiments

Change only one parameter at a time and compare the figures.

### Change the random realization

```python
rng=8
```

This changes the random coefficients `xi` without changing the covariance or
KL modes.

### Change the variability

```python
sigma=0.3
```

Smaller `sigma` keeps `kappa` closer to one. Large values can produce strong
coefficient contrast after exponentiation.

### Change the spatial scale

```python
correlation_length=0.10
```

Smaller correlation length creates shorter-scale variation. The KL grid and
FE mesh must be fine enough to resolve it.

### Retain more KL modes

```python
num_modes=30
```

This increases retained variance and permits more detailed samples, at the
cost of more stochastic dimensions.

### Refine the background KL grid

```python
nx = ny = 32
```

This represents the covariance on more points, but the dense covariance
matrix grows from `N by N` with `N=nx*ny`. Memory and eigensolver costs grow
quickly.

### Add another FE refinement

```python
n_refines=2
```

This produces three levels rather than two. The same `VoxelCoefficient`
continues to be used by every level.

## 19. Current limitations

1. The covariance matrix is dense.
2. The KL expansion is a point-grid discretization, not an FE generalized
  eigenproblem.
3. `VoxelCoefficient` uses a rectangular background grid.
4. The mesh-overlay helper assumes triangular two-dimensional elements.
5. The example solves one realization; Monte Carlo would repeat the sample,
  assembly, and solve stages.

For multiple realizations, compute the covariance eigenpairs once, draw new
`xi` values repeatedly, rebuild `kappa`, and reassemble the hierarchy because
the stiffness matrices depend on the new coefficient.

## 20. Central idea to remember

There are three separate discretizations or representations:

1. The Cartesian grid used to construct and sample the discrete KL field.
2. The NGSolve coarse finite-element mesh.
3. The NGSolve fine finite-element mesh.

`VoxelCoefficient` connects the first representation to both FE meshes by
physical-coordinate evaluation. `P` and `P.T` connect the coarse and fine FE
spaces during the V-cycle. Keeping these two roles separate is what makes the
workflow consistent.