# MLMC Runner Outline

This document describes the high-level interface between the MLMC runner and
a user-defined computational model. Level $0$ is the coarsest level, and
larger level indices represent increasingly accurate or expensive models.
Each level ultimately represents a numerical linear system

$$
A_\ell x_\ell = b_\ell.
$$

The system may come from a finite-element mesh, a graph, a time-stepping
model, a finite-difference grid, a reduced basis, or any other construction.
The MLMC runner must not assume that a level has a mesh.

## Separation of responsibilities

The MLMC runner knows:

- the number and indices of the levels;
- that $Y_0 = Q_0$;
- that $Y_\ell = Q_\ell - Q_{\ell-1}$ for $\ell > 0$;
- how many correction samples to request at each level;
- how to manage reproducible random-number streams;
- how to measure costs and accumulate sample statistics.

The user model knows:

- what each level represents;
- how much latent randomness is required at a level;
- how to couple the inputs for levels $\ell$ and $\ell-1$;
- how to construct and run the linear system at one level;
- how to compute the quantity of interest.

The MLMC runner cannot verify that the user's coupling is mathematically
correct. It can only verify that the required current and previous inputs are
present.

## User level representation

The runner itself only requires integer level indices. However, the user model
may use a `UserLevel` class to organize the data and operations associated with
each fidelity level. This is a user-side model object, not something the MLMC
runner needs to understand internally.

A conceptual level interface is:

```python
@dataclass
class UserLevel:
    """One fidelity level of the user's linear-system model."""

    index: int
    shape: tuple[int, int] | None = None

    def generate_sample(self, xi):
        """Map level-appropriate latent randomness to this level's model input."""
        ...
```

The attributes have the following meaning:

- `index` identifies the level, with `0` representing the coarsest level;
- `shape` may describe the shape of the level's linear system, but it is
  optional because a model may not know the shape until assembly;
- `shape` describes the linear system, not necessarily the size of `xi`.

The user model may internally store one object per level containing:

- the level index;
- a matrix dimension or other fidelity information;
- an optional mesh, graph, time step, grid spacing, or basis description;
- cached matrices, assemblies, or solver state;
- level-specific solver configuration.

Randomness should not be generated independently by each level because the
two sides of an MLMC correction must be coupled.

`generate_sample(xi)` does not draw unrelated randomness. It transforms the
latent randomness supplied by the user model into the input needed by that
level. For a correction above level `0`, the user model remains responsible for
constructing compatible current-level and previous-level inputs.

## Coupled inputs

`CoupledInputs` contains the two model inputs required for one correction.
The inputs may have different dimensions or even different concrete types.

```python
class CoupledInputs:
    """Inputs for the current level and its coupled previous level."""

    current: object
    previous: object | None
```

At level $0$, `previous` must be `None`. At level $\ell > 0$, `previous`
must contain the input used to evaluate $Q_{\ell-1}$.

## User model interface

The user supplies one model object with the following methods.

Typical user-model attributes may include:

- `levels`: the ordered `UserLevel` objects or equivalent level data;
- model parameters shared by every level;
- cached matrices, factorizations, or assembly information;
- configuration needed to construct a linear system and evaluate its quantity
  of interest.

The runner should not reach into these attributes. It interacts with the model
through the methods below.

### Mapping the original names to the runner interface

The initial design used the names `get_xi()` and `get_samples()`. They map to
the current interface as follows:

| Original design name | Current interface name | Responsibility |
| --- | --- | --- |
| `get_xi(level, rng)` | `sample(level, rng)` | Draw one latent random object for a correction level. |
| `get_samples(level, xi)` | `couple(level, xi)` | Produce the current and previous level inputs from that draw. |
| `UserLevel.generate_sample(xi)` | Optional user-model helper | Convert level-appropriate randomness into one level's model input. |

Only one naming convention should appear in the eventual public API. The
current scaffold uses `sample()` and `couple()`, while the table preserves how
they correspond to the original design.

Conceptually, the original user-model methods are:

```python
def get_xi(level: int, rng: np.random.Generator):
    """Draw the latent randomness used for one correction at this level."""
    ...


def get_samples(level: int, xi) -> CoupledInputs:
    """Return one level input at level 0 or coupled inputs above level 0."""
    ...
```

The random object returned by `get_xi()` does not have to be a NumPy vector,
and its size does not have to equal the number of unknowns at the requested
level. Those details belong to the user model and its coupling construction.

### `sample()`

```python
def sample(level: int, rng: np.random.Generator):
    """Draw the latent randomness associated with correction level l."""
    ...
```

Inputs:

- `level`: the current correction level $\ell$;
- `rng`: the random-number generator owned by the runner.

Output:

- a model-specific latent random object, often written as `xi`;
- its size does not have to equal the number of model unknowns;
- for discretization-dependent noise, it may have a size associated with the
  current linear system;
- for KL or scalar parameters, its size may be independent of the level.

### `couple()`

```python
def couple(level: int, randomness) -> CoupledInputs:
    """Map one random draw to coupled inputs for levels l and l-1."""
    ...
```

Inputs:

- `level`: the current correction level $\ell$;
- `randomness`: the single latent draw returned by `sample()`.

Output:

- `current`: the input used to evaluate $Q_\ell$;
- `previous`: the coupled input used to evaluate $Q_{\ell-1}$, or `None`
  at level $0$.

Examples of coupling include:

- evaluating the same KL coefficients on two discretizations;
- restricting or projecting a fine random vector into a smaller vector space;
- aggregating graph-based random inputs onto a coarser graph;
- using one stochastic forcing realization with two time-step sizes;
- passing the same scalar random parameter to both levels.

### `run()`

```python
def run(level: int, model_input) -> float:
    """Run one level and return its scalar quantity of interest."""
    ...
```

Inputs:

- `level`: the level to evaluate;
- `model_input`: the level-specific input produced by `couple()`.

Output:

- the scalar quantity of interest $Q_\ell$.

Internally, `run()` may assemble a linear system, call
`solve_linear_system()`, and evaluate the quantity of interest. Regardless of
whether the system originated from a mesh, graph, time discretization, or
another model, it must cross the solver boundary in a supported NumPy/SciPy
form. The low-level linear solver still knows nothing about MLMC levels or
random samples.

## MLMC runner configuration

The runner initially requires:

- a user model;
- `number_of_levels`;
- a reproducibility seed;
- fixed sample counts for levels $0, \ldots, L$.

The runner should create one independent random-number stream per correction
level. Current and previous-level inputs within one correction remain coupled
because they come from the same call to `model.sample()`.

## Three-stage runner view

At the highest level, the MLMC runner performs the following workflow for each
requested correction sample:

1. Generate one latent random object by asking the user model for `xi`:

   ```python
   xi = model.sample(level, rng)  # original name: model.get_xi(...)
   ```

2. Ask the user model to construct the level input or coupled pair of inputs:

   ```python
   inputs = model.couple(level, xi)  # original name: model.get_samples(...)
   ```

   At level `0`, the model returns only the current input. Above level `0`, it
   returns inputs for levels `level` and `level - 1`. These inputs may have
   different shapes.

3. Solve for the correction. The runner calls `model.run()` once at level `0`
   or twice above level `0`. Each call may assemble and solve a linear system.
   The runner then forms either $Q_0$ or $Q_\ell-Q_{\ell-1}$.

The runner controls when these operations occur, but the user model controls
the random representation, the coupling rule, construction of each linear
system, and evaluation of the quantity of interest.

## One correction sample

```python
def sample_correction(level: int, rng: np.random.Generator) -> float:
    """Generate and evaluate one MLMC correction at the requested level."""
    ...
```

The runner performs these steps:

1. Draw one latent random object:

   ```python
   xi = model.sample(level, rng)
   ```

2. Construct the coupled model inputs:

   ```python
   inputs = model.couple(level, xi)
   ```

3. Evaluate the current level:

   ```python
   q_current = model.run(level, inputs.current)
   ```

4. Form the correction.

   At level $0$:

   ```python
   correction = q_current
   ```

   At level $\ell > 0$:

   ```python
   q_previous = model.run(level - 1, inputs.previous)
   correction = q_current - q_previous
   ```

5. Measure the total cost of generating the correction.

6. Validate that the quantities of interest and correction are finite.

7. Update the running statistics for the correction level.

## Fixed-sample run

```python
def run_fixed(sample_counts: tuple[int, ...]) -> MLMCResult:
    """Run a fixed number of correction samples at every level."""
    ...
```

For each level, the runner:

- validates the requested sample count;
- repeatedly calls `sample_correction()`;
- updates the correction mean, variance, and measured cost.

After all levels finish, the runner computes

$$
\widehat Q_{\mathrm{ML}}
=
\sum_{\ell=0}^{L} \overline{Y}_\ell,
$$

with estimated sampling variance

$$
\operatorname{Var}(\widehat Q_{\mathrm{ML}})
\approx
\sum_{\ell=0}^{L}\frac{\widehat V_\ell}{N_\ell}.
$$

## Results

Each level result should contain:

- level index;
- sample count $N_\ell$;
- mean correction $\overline{Y}_\ell$;
- correction variance $\widehat V_\ell$;
- mean cost per correction;
- total cost.

The final MLMC result should contain:

- the MLMC estimate;
- estimated sampling variance;
- standard error;
- per-level results;
- total measured cost.

## Initial implementation order

1. Define the user-model protocol and result data classes.
2. Implement online running statistics.
3. Implement one correction sample.
4. Implement a serial fixed-sample runner.
5. Test with a small synthetic model.
6. Add adaptive sample allocation using estimated variance and cost.
7. Add bias estimation and automatic level selection.
8. Add parallel execution, persistence, and restart support.
