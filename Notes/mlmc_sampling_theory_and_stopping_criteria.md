# Monte Carlo and Multilevel Monte Carlo Sampling Theory

## Purpose of this note

This note explains the sampling theory and stopping criteria used in the
multilevel Monte Carlo (MLMC) method. It is written to support the implementation
in `examples/KLMC/multilevel/multilevel_test.py`, but the theory is independent
of Python, NGSolve, and the particular elliptic PDE.

The most important distinction in the entire note is this:

> The variance of a random quantity and the variance of an estimated mean are
> different quantities.

For an MLMC correction (Y_\ell), these are

\[
V_\ell = \operatorname{Var}(Y_\ell)
\]

and

\[
\operatorname{Var}(\overline{Y}_\ell)
= \frac{V_\ell}{N_\ell}.
\]

Taking more samples does not normally change (V_\ell). It reduces
(V_\ell/N_\ell), which is the uncertainty in the estimated expected value.
The MLMC sampling stopping criterion is therefore based on the variance of the
estimated mean, not directly on the raw variance of the correction samples.

The presentation follows Section 2 of Cliffe, Giles, Scheichl, and Teckentrup
(2011), especially equations (2.5)-(2.9) and the practical algorithm described
after Theorem 2.1.

---

## 1. The quantity we want to estimate

Let (Q) be a scalar random quantity of interest (QoI). In the groundwater-flow
problem, (Q) is computed from the solution of a PDE whose conductivity is
random. Examples include:

- the pressure at a selected point;
- an average pressure over part of the domain;
- the flux through part of the boundary;
- a norm or another functional of the PDE solution.

The goal is not to compute one realization of (Q). The goal is to compute its
expected value

\[
\mathbb{E}[Q].
\]

The exact continuum PDE cannot be solved numerically, so let (Q_\ell) denote
the QoI computed on spatial level (\ell). Here:

- (Q_0) is computed on the coarsest mesh;
- (Q_1) is computed on the next finer mesh;
- (Q_L) is computed on the active finest mesh.

As the mesh is refined, we want (Q_\ell) to approach (Q).

There are therefore two separate approximations:

1. **Spatial approximation:** replace (Q) by (Q_L).
2. **Sampling approximation:** replace an exact expectation by a finite sample
   average.

These produce two different sources of error: discretization bias and sampling
error.

---

## 2. Ordinary Monte Carlo first

Suppose we draw (N) independent random-field samples and solve the level-(L)
problem for each one. This gives

\[
Q_L^{(1)}, Q_L^{(2)}, \ldots, Q_L^{(N)}.
\]

The ordinary Monte Carlo estimator is the sample mean

\[
\widehat{Q}^{\mathrm{MC}}_{L,N}
= \frac{1}{N}\sum_{i=1}^{N}Q_L^{(i)}.
\]

Because the samples are independent and identically distributed,

\[
\mathbb{E}\left[\widehat{Q}^{\mathrm{MC}}_{L,N}\right]
= \mathbb{E}[Q_L].
\]

Thus, ordinary Monte Carlo gives an unbiased estimator of the *discrete*
expectation (\mathbb{E}[Q_L]). It is not generally an unbiased estimator of
the continuum expectation (\mathbb{E}[Q]), because (Q_L\neq Q).

### 2.1 Raw variance versus variance of the sample mean

Let

\[
V_L=\operatorname{Var}(Q_L).
\]

This describes how much individual realizations (Q_L^{(i)}) vary. Taking more
samples does not make the physical random variable (Q_L) less variable.
Instead, it makes the average more stable:

\[
\operatorname{Var}\left(
\widehat{Q}^{\mathrm{MC}}_{L,N}
\right)
= \frac{V_L}{N}.
\]

For example, suppose individual QoIs have variance (V_L=4).

- With (N=1), the estimator variance is (4).
- With (N=100), the estimator variance is (4/100=0.04).
- With (N=10{,}000), the estimator variance is (4/10{,}000=0.0004).

The raw variance remains (4), but the estimated mean becomes increasingly
precise.

### 2.2 Standard deviation and standard error

The standard deviation of individual QoI samples is

\[
\sigma_L=\sqrt{V_L}.
\]

The standard deviation of the Monte Carlo estimator is

\[
\operatorname{SE}
= \sqrt{\frac{V_L}{N}}
= \frac{\sigma_L}{\sqrt{N}}.
\]

This second quantity is called the **standard error**. It measures the sampling
uncertainty in the estimated mean.

There is no additional division by (N) after taking this square root. The
division has already occurred inside the variance:

\[
\operatorname{SE}
= \sqrt{V_L/N},
\]

not

\[
\sqrt{V_L/N}/N.
\]

### 2.3 Estimating variance from samples

The true variance is not known, so it is estimated from the samples. If

\[
\overline{Q}_L=\frac{1}{N}\sum_{i=1}^{N}Q_L^{(i)},
\]

then the unbiased sample variance is

\[
\widehat{V}_L
= \frac{1}{N-1}
\sum_{i=1}^{N}
\left(Q_L^{(i)}-\overline{Q}_L\right)^2.
\]

The (N-1) denominator is used because the sample mean was estimated from the
same data. In NumPy, this corresponds to `np.var(samples, ddof=1)`.

---

## 3. The MLMC telescoping identity

MLMC does not estimate (\mathbb{E}[Q_L]) by repeatedly solving only the finest
problem. It defines correction random variables

\[
Y_0=Q_0
\]

and, for (\ell\geq 1),

\[
Y_\ell=Q_\ell-Q_{\ell-1}.
\]

Adding these quantities causes all intermediate terms to cancel:

\[
\begin{aligned}
Q_0
&+(Q_1-Q_0)
 +(Q_2-Q_1)
 +\cdots
 +(Q_L-Q_{L-1}) \\
&=Q_L.
\end{aligned}
\]

Taking expectations gives the telescoping identity

\[
\mathbb{E}[Q_L]
= \sum_{\ell=0}^{L}\mathbb{E}[Y_\ell].
\]

This is an exact algebraic identity. MLMC changes how the individual
expectations on the right are estimated; it does not change the target
(\mathbb{E}[Q_L]).

### 3.1 The MLMC estimator

Let (N_\ell) be the number of samples used for correction (Y_\ell). Its
sample mean is

\[
\overline{Y}_\ell
= \frac{1}{N_\ell}
\sum_{i=1}^{N_\ell}Y_\ell^{(i)}.
\]

The MLMC estimator is

\[
\widehat{Q}^{\mathrm{ML}}
=\sum_{\ell=0}^{L}\overline{Y}_\ell.
\]

Different terms may use different numbers of samples. Usually (N_0) is large
and (N_L) is much smaller because coarse solves are cheap while fine solves
are expensive.

---

## 4. Coupling and independence: two rules that look contradictory

Correct MLMC sampling uses both coupling and independence, but in different
places.

### 4.1 Within one correction sample, use the same random input

For one sample of (Y_\ell), use one Gaussian vector (\xi^{(i)}) to construct
one physical conductivity field (\kappa^{(i)}), and solve both levels with that
same field:

\[
Y_\ell^{(i)}
=Q_\ell\left(\xi^{(i)}\right)
-Q_{\ell-1}\left(\xi^{(i)}\right).
\]

Do not draw one (\xi) for (Q_\ell) and another for (Q_{\ell-1}).

The point of the shared input is that the two QoIs are then strongly correlated.
They contain almost the same physical response, so subtracting them cancels
most of the random variation. Consequently,

\[
V_\ell=\operatorname{Var}(Y_\ell)
=\operatorname{Var}(Q_\ell-Q_{\ell-1})
\]

is often much smaller than either
(\operatorname{Var}(Q_\ell)) or
(\operatorname{Var}(Q_{\ell-1})).

This small correction variance is the central variance-reduction mechanism in
MLMC.

### 4.2 Between correction estimators, use independent sample streams

The sample collection used to estimate (\mathbb{E}[Y_0]) should be independent
of the collection used for (\mathbb{E}[Y_1]), and similarly for the other
terms. With independent term estimators,

\[
\operatorname{Cov}(\overline{Y}_\ell,\overline{Y}_k)=0
\qquad \text{for }\ell\neq k.
\]

This gives the simple sum-of-variances formula used below. Separate child random
number streams created with `SeedSequence.spawn` are a reproducible way to
implement this independence.

The two rules are therefore:

- **same (\xi) inside one difference** (Q_\ell-Q_{\ell-1});
- **independent sequences of (\xi)** for different MLMC terms.

If term estimators are not independent, the estimator variance also contains
covariance terms. The code and formulas in this note assume independent terms.

---

## 5. The three variances that must not be confused

For each correction level (\ell), define

\[
V_\ell=\operatorname{Var}(Y_\ell).
\]

This is the raw correction variance. It describes the distribution of
individual (Y_\ell) samples.

The variance of the estimated correction mean is

\[
\operatorname{Var}(\overline{Y}_\ell)
=\frac{V_\ell}{N_\ell}.
\]

Finally, because the term estimators are independent, the variance of the full
MLMC estimator is

\[
\boxed{
\operatorname{Var}\left(\widehat{Q}^{\mathrm{ML}}\right)
=\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}
}
\]

and its standard error is

\[
\boxed{
\operatorname{SE}\left(\widehat{Q}^{\mathrm{ML}}\right)
=\sqrt{
\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}
}
}.
\]

In practice, the unknown (V_\ell) values are replaced by unbiased sample
variance estimates (\widehat{V}_\ell).

### 5.1 What adding samples changes

For a fixed level (\ell):

- increasing (N_\ell) does **not** force (V_\ell) toward zero;
- increasing (N_\ell) **does** force (V_\ell/N_\ell) toward zero;
- refining the mesh may make (V_\ell) decrease with (\ell), because two
  consecutive approximations become more similar.

The first and third statements concern different operations. Sampling more at
the same level improves the estimate of a mean. Refining to a higher level may
change the distribution of the correction itself.

### 5.2 Why raw correction variance is still important

Although (V_\ell) is not the final stopping quantity, it determines how many
samples the term needs. A term with a large (V_\ell) needs more samples to
estimate its expectation accurately. A term with small (V_\ell) needs fewer.

Thus:

- (V_\ell) is used for **sample allocation**;
- (V_\ell/N_\ell) is used for **sampling uncertainty**;
- the sum (\sum_\ell V_\ell/N_\ell) is used for the **global sampling stopping
  test**.

---

## 6. Mean square error and the two stopping tests

The final goal is to approximate (\mathbb{E}[Q]), not merely
(\mathbb{E}[Q_L]). Define the root mean square error (RMSE) by

\[
\operatorname{RMSE}
=\sqrt{
\mathbb{E}\left[
\left(
\widehat{Q}^{\mathrm{ML}}-\mathbb{E}[Q]
\right)^2
\right]
}.
\]

The mean square error is the square of the RMSE. For the MLMC estimator it
decomposes as

\[
\boxed{
\operatorname{MSE}
=
\underbrace{
\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}
}_{\text{sampling variance}}
+
\underbrace{
\left(
\mathbb{E}[Q_L-Q]
\right)^2
}_{\text{squared discretization bias}}
}.
\]

This is equation (2.8) in Cliffe et al. The decomposition explains why one
stopping test is not enough.

### 6.1 Sampling error

Even at a fixed spatial level (L), a finite sample mean is random. Its
uncertainty is measured by

\[
\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}.
\]

This error is reduced by adding samples.

### 6.2 Discretization bias

Even with infinitely many samples, the MLMC estimator at finest level (L)
converges to (\mathbb{E}[Q_L]), not automatically to (\mathbb{E}[Q]). The
remaining bias is

\[
B_L=\mathbb{E}[Q_L-Q].
\]

This error is reduced by adding finer spatial levels, not by repeatedly
sampling the current levels.

This leads to an important statement:

> Infinite sampling cannot remove spatial-discretization bias.

---

## 7. Turning an RMSE tolerance into stopping criteria

Let the requested RMSE tolerance be (\varepsilon>0). A standard sufficient
condition is to split the squared error budget equally between sampling and
bias:

\[
\text{sampling variance}
\leq \frac{\varepsilon^2}{2}
\]

and

\[
\text{squared bias}
\leq \frac{\varepsilon^2}{2}.
\]

Equivalently, the two practical targets are

\[
\boxed{
\sum_{\ell=0}^{L}
\frac{\widehat{V}_\ell}{N_\ell}
\leq \frac{\varepsilon^2}{2}
}
\]

and

\[
\boxed{
|\widehat{B}_L|
\leq \frac{\varepsilon}{\sqrt{2}}
}.
\]

If both hold, the estimated MSE is at most approximately (\varepsilon^2), so
the estimated RMSE is at most approximately (\varepsilon).

### 7.1 The equivalent standard-error test

The sampling test may also be written as

\[
\operatorname{SE}\left(\widehat{Q}^{\mathrm{ML}}\right)
\leq \frac{\varepsilon}{\sqrt{2}}.
\]

These two tests are mathematically identical because standard error is the
square root of estimator variance.

### 7.2 Should the variance of the expected value go to zero?

In an asymptotic analysis where (\varepsilon\to 0), yes: the sampling variance
of the estimator must also approach zero.

For one computation with a fixed requested tolerance, it does not need to
become literally zero. It only needs to become smaller than the chosen sampling
variance budget (\varepsilon^2/2). Requiring zero variance would require
infinitely many samples.

### 7.3 Why the tolerance is squared in the variance test

An RMSE tolerance (\varepsilon) has the same physical units as the QoI. A
variance has squared units. Therefore a variance must be compared with a
quantity involving (\varepsilon^2), not (\varepsilon).

This dimensional check catches a common mistake:

\[
\sum_\ell \widehat{V}_\ell/N_\ell \leq \varepsilon
\qquad \text{is generally not the RMSE criterion.}
\]

The correct equal-budget criterion is

\[
\sum_\ell \widehat{V}_\ell/N_\ell
\leq \varepsilon^2/2.
\]

---

## 8. Equal per-level tolerances versus optimal allocation

One simple valid strategy is to split the sampling variance budget equally
among the (L+1) terms:

\[
\frac{\widehat{V}_\ell}{N_\ell}
\leq
\frac{\varepsilon^2}{2(L+1)}
\qquad \text{for every }\ell.
\]

If every term passes, summing the inequalities guarantees the global sampling
criterion. This is easy to understand, but it is usually not the cheapest
choice because the levels have different costs and variances.

It is not correct to use (\varepsilon/(L+1)) as a variance target. If an equal
split is used, the object being divided is the variance budget
(\varepsilon^2/2).

The paper instead uses an allocation that accounts for both:

- the correction variance (V_\ell);
- the average cost (C_\ell) of one (Y_\ell) sample.

For (\ell>0), (C_\ell) includes both the (Q_\ell) and (Q_{\ell-1}) solves
needed to produce one coupled difference.

---

## 9. Deriving the optimal number of samples

The expected total sampling cost is

\[
C_{\mathrm{total}}
=\sum_{\ell=0}^{L}N_\ell C_\ell.
\]

Let the allowed sampling variance be

\[
T_{\mathrm{samp}}=\frac{\varepsilon^2}{2}.
\]

We want to minimize total cost while satisfying

\[
\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}
=T_{\mathrm{samp}}.
\]

For the derivation, temporarily treat each (N_\ell) as a positive continuous
number. Introduce the Lagrangian

\[
\mathcal{L}
=\sum_{\ell=0}^{L}N_\ell C_\ell
+\lambda\left(
\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}
-T_{\mathrm{samp}}
\right).
\]

Differentiate with respect to (N_\ell):

\[
\frac{\partial\mathcal{L}}{\partial N_\ell}
=C_\ell-\lambda\frac{V_\ell}{N_\ell^2}.
\]

At the optimum this derivative is zero, so

\[
C_\ell
=\lambda\frac{V_\ell}{N_\ell^2}
\]

and therefore

\[
N_\ell
=\sqrt{\lambda}
\sqrt{\frac{V_\ell}{C_\ell}}.
\]

This is the proportionality stated in equation (2.9) of the paper:

\[
\boxed{
N_\ell\propto\sqrt{\frac{V_\ell}{C_\ell}}
}.
\]

It has an intuitive interpretation:

- a larger variance (V_\ell) increases the required sample count;
- a larger cost (C_\ell) discourages placing samples on that level;
- the square roots balance those two effects.

To find the constant of proportionality, define

\[
S=\sum_{j=0}^{L}\sqrt{V_jC_j}.
\]

Substituting the allocation into the variance constraint gives

\[
\sqrt{\lambda}=\frac{S}{T_{\mathrm{samp}}}.
\]

Consequently,

\[
\boxed{
N_\ell
=
\frac{S}{T_{\mathrm{samp}}}
\sqrt{\frac{V_\ell}{C_\ell}}
}
\]

or, using (T_{\mathrm{samp}}=\varepsilon^2/2),

\[
\boxed{
N_\ell
=
\frac{2}{\varepsilon^2}
\left(
\sum_{j=0}^{L}\sqrt{V_jC_j}
\right)
\sqrt{\frac{V_\ell}{C_\ell}}.
}
\]

In code, replace (V_\ell) and (C_\ell) by sample estimates, round upward to
an integer, and enforce a minimum pilot sample count.

---

## 10. A complete numerical allocation example

Suppose there are three terms and the estimated variances and costs are

| Term | (V_\ell) | (C_\ell) |
|---|---:|---:|
| (Y_0=Q_0) | (1.00) | (1) |
| (Y_1=Q_1-Q_0) | (0.16) | (4) |
| (Y_2=Q_2-Q_1) | (0.04) | (16) |

Let the requested RMSE tolerance be

\[
\varepsilon=0.1.
\]

The sampling variance target is

\[
T_{\mathrm{samp}}
=\frac{0.1^2}{2}
=0.005.
\]

First calculate

\[
S
=\sqrt{1.00\cdot 1}
+\sqrt{0.16\cdot 4}
+\sqrt{0.04\cdot 16}
=1+0.8+0.8
=2.6.
\]

Next calculate the variance-to-cost factors:

\[
\sqrt{V_0/C_0}=1,
\qquad
\sqrt{V_1/C_1}=0.2,
\qquad
\sqrt{V_2/C_2}=0.05.
\]

The target counts are therefore

\[
N_0=\frac{2.6}{0.005}(1)=520,
\]

\[
N_1=\frac{2.6}{0.005}(0.2)=104,
\]

and

\[
N_2=\frac{2.6}{0.005}(0.05)=26.
\]

Check the resulting estimator variance:

\[
\frac{1.00}{520}
+\frac{0.16}{104}
+\frac{0.04}{26}
=0.005.
\]

The standard error is

\[
\sqrt{0.005}
=0.07071
=\frac{0.1}{\sqrt{2}}.
\]

Notice the pattern: there are many cheap coarse samples and only 26 expensive
fine correction samples. The fine term does not need 520 samples because its
correction variance is already small and its samples are expensive.

---

## 11. Estimating and testing the remaining bias

The exact bias

\[
B_L=\mathbb{E}[Q_L-Q]
\]

cannot be computed directly because (Q) is inaccessible. The finest
correction mean is used to estimate its size.

Suppose the leading weak error behaves like

\[
\mathbb{E}[Q_\ell-Q]
\approx cM_\ell^{-\alpha},
\]

where (M_\ell) is a measure of spatial resolution and

\[
M_\ell=sM_{\ell-1},
\qquad s>1.
\]

Then

\[
\begin{aligned}
\mathbb{E}[Y_\ell]
&=\mathbb{E}[Q_\ell-Q_{\ell-1}] \\
&\approx cM_\ell^{-\alpha}
-cM_{\ell-1}^{-\alpha} \\
&=cM_\ell^{-\alpha}(1-s^\alpha).
\end{aligned}
\]

The magnitude of the remaining bias can therefore be estimated by

\[
\boxed{
|\widehat{B}_L|
\approx
\frac{|\overline{Y}_L|}{s^\alpha-1}.
}
\]

If the convergence law is written using mesh size instead, with
(h_{\ell-1}=rh_\ell) and bias (O(h_\ell^\alpha)), the corresponding formula
is

\[
|\widehat{B}_L|
\approx
\frac{|\overline{Y}_L|}{r^\alpha-1}.
\]

The bias stopping test is then

\[
\frac{|\overline{Y}_L|}{s^\alpha-1}
\leq \frac{\varepsilon}{\sqrt{2}}.
\]

If this fails, merely adding more samples at the existing levels does not
remove the estimated bias. A new, finer level should be added.

### 11.1 Why this is an estimate rather than an identity

The formula assumes that the computation is in the asymptotic convergence
regime and that a single leading-order error term dominates. On coarse meshes,
the model may not yet be accurate. Also, if (\overline{Y}_L) is small relative
to its sampling noise, the bias estimate may be unstable.

Practical safeguards include:

- using a known theoretical weak rate (\alpha), when available;
- estimating (\alpha) from several recent correction means rather than one
  ratio;
- requiring enough finest-level samples before trusting
  (\overline{Y}_L);
- checking that the latest correction means decay approximately
  geometrically;
- using a conservative estimate based on more than one of the latest levels.

The paper states the convergence test asymptotically as

\[
|\widehat{Y}_L|\asymp M_L^{-\alpha}
\]

and uses it to decide whether another level is needed.

---

## 12. The adaptive MLMC algorithm

A practical adaptive algorithm proceeds as follows.

### Step 1: start with a coarse active hierarchy

Start with (L=0), or with a small number of levels if at least two levels are
needed to form an initial bias estimate.

### Step 2: take pilot samples

Take at least two samples per active term so a sample variance exists. In
practice, use more than two because a variance estimated from two observations
is extremely noisy.

For every active term, estimate:

\[
\widehat{V}_\ell
=\text{sample variance of }Y_\ell
\]

and

\[
\widehat{C}_\ell
=\text{average time or work for one }Y_\ell\text{ sample}.
\]

### Step 3: compute optimal target counts

Using the current (\widehat{V}_\ell) and (\widehat{C}_\ell), compute

\[
N_\ell^{\mathrm{target}}
=\left\lceil
\frac{
\sum_j\sqrt{\widehat{V}_j\widehat{C}_j}
}{
\varepsilon^2/2
}
\sqrt{
\frac{\widehat{V}_\ell}{\widehat{C}_\ell}
}
\right\rceil.
\]

### Step 4: add only the missing samples

If the term already has (N_\ell) samples, draw

\[
\max\left(0,
N_\ell^{\mathrm{target}}-N_\ell
\right)
\]

additional samples. Existing samples remain valid and should not be discarded.

### Step 5: recompute estimates and check sampling convergence

Because the variance and cost estimates change as data are added, recompute the
targets. Continue until

\[
\sum_{\ell=0}^{L}
\frac{\widehat{V}_\ell}{N_\ell}
\leq \frac{\varepsilon^2}{2}.
\]

It can take several allocation iterations for the estimates and counts to
settle.

### Step 6: check bias convergence

When (L\geq 1), compute a bias estimate from the finest correction mean and
test

\[
|\widehat{B}_L|
\leq \frac{\varepsilon}{\sqrt{2}}.
\]

### Step 7: add a finer level if necessary

If sampling has converged but bias has not, set (L\leftarrow L+1), take pilot
samples of the new correction, and return to the allocation step. Adding a new
level can also change the optimal target counts on the old levels, so all active
targets should be recomputed.

### Step 8: stop only when both errors pass

The final stopping condition is

\[
\boxed{
\text{sampling converged}
\quad\text{and}\quad
\text{bias converged}.
}
\]

Passing only the sampling test means that (\mathbb{E}[Q_L]) has been estimated
accurately, but (Q_L) might still be a poor approximation of (Q).

---

## 13. Welford's running variance update

The code does not need to recompute the mean and variance from the full sample
list after every new sample. Welford's algorithm updates them in one pass.

Suppose the old sample count is (n-1), the old mean is
(\overline{Y}_{n-1}), and a new correction (y_n) arrives.

First update the count:

\[
n\leftarrow n+1.
\]

Define the difference from the old mean:

\[
\delta=y_n-\overline{Y}_{n-1}.
\]

Update the mean:

\[
\overline{Y}_n
=\overline{Y}_{n-1}+\frac{\delta}{n}.
\]

Then define the difference from the new mean:

\[
\delta_2=y_n-\overline{Y}_n.
\]

Finally update

\[
M_{2,n}=M_{2,n-1}+\delta\delta_2.
\]

After all (n) samples,

\[
M_{2,n}
=\sum_{i=1}^{n}
\left(Y^{(i)}-\overline{Y}_n\right)^2.
\]

Therefore the unbiased sample variance is

\[
\widehat{V}=\frac{M_{2,n}}{n-1}.
\]

This explains the code:

```python
self.sample_count += 1

difference = correction - self.mean
self.mean += difference / self.sample_count

difference_from_new_mean = correction - self.mean

self.sum_squared_deviations += (
    difference * difference_from_new_mean
)
```

`sum_squared_deviations` is mathematically accurate: after the update it stores
the sum of squared deviations from the current mean. Common shorter names are
`m2`, `correction_m2`, or `sum_squared_deviations_from_mean`.

The product in the update is not visibly a square because the mean moves when a
new observation is added. Nevertheless, after the update it produces exactly
the required final sum of squared deviations, up to floating-point roundoff.

Keeping `upper_qois` and `lower_qois` in lists is optional for the statistics,
but useful for plotting and diagnostics. Welford's algorithm is still valuable
because it provides stable online statistics and does not require traversing
the lists after every sample.

---

## 14. Mapping the mathematics to `multilevel_test.py`

The current code quantities correspond to the theory as follows:

| Code name | Mathematical meaning |
|---|---|
| `MLMCTerm.sample_count` | (N_\ell) |
| `MLMCTerm.mean` | (\overline{Y}_\ell) |
| `MLMCTerm.sum_squared_deviations` | (M_{2,\ell}=\sum_i(Y_\ell^{(i)}-\overline{Y}_\ell)^2) |
| `MLMCTerm.sample_variance` | (\widehat{V}_\ell) |
| `MLMCTerm.variance_of_mean` | (\widehat{V}_\ell/N_\ell) |
| `MLMCTerm.upper_qois` | stored (Q_\ell^{(i)}) values |
| `MLMCTerm.lower_qois` | stored (Q_{\ell-1}^{(i)}) values in the same coupled pairs |
| `MLMCTerm.corrections` | stored (Y_\ell^{(i)}) values |
| `MultilevelMonteCarlo.estimate_qoi` | (\sum_\ell\overline{Y}_\ell) |
| `MultilevelMonteCarlo.estimator_variance` | (\sum_\ell\widehat{V}_\ell/N_\ell) |
| `MultilevelMonteCarlo.standard_error` | (\sqrt{\sum_\ell\widehat{V}_\ell/N_\ell}) |

The current `standard_error` implementation is therefore conceptually correct:

```python
return float(
    np.sqrt(
        self.estimator_variance
    )
)
```

It must not divide by another sample count. `estimator_variance` has already
added the independently scaled terms
(\widehat{V}_\ell/N_\ell), each of which can have a different (N_\ell).
There is no single additional (N) by which the full MLMC estimator should be
divided.

### 14.1 Optional more explicit names

The existing names are valid. If names that emphasize the MLMC roles are
preferred, possible alternatives are:

| Current name | More explicit alternative |
|---|---|
| `mean` | `correction_mean` |
| `sum_squared_deviations` | `correction_m2` |
| `sample_variance` | `correction_sample_variance` |
| `variance_of_mean` | `term_estimator_variance` |
| `estimator_variance` | `mlmc_sampling_variance` |
| `standard_error` | `mlmc_sampling_standard_error` |

These are naming choices, not mathematical changes.

---

## 15. A compact implementation outline

The following pseudocode shows the division of responsibilities. It is not a
replacement for the current classes; it records the theory that an adaptive
driver should implement.

```python
sampling_variance_target = tolerance**2 / 2.0
bias_target = tolerance / np.sqrt(2.0)

# 1. Make sure every active term has pilot samples.
for term in active_terms:
    if term.sample_count < pilot_count:
        term.add_samples(pilot_count - term.sample_count)

while True:
    # 2. Estimate raw correction variances and costs.
    variances = np.array([
        term.sample_variance
        for term in active_terms
    ])
    costs = np.array([
        term.average_cost_per_sample
        for term in active_terms
    ])

    # 3. Compute the cost-optimal counts.
    weighted_sum = np.sum(np.sqrt(variances * costs))
    targets = np.ceil(
        (weighted_sum / sampling_variance_target)
        * np.sqrt(variances / costs)
    ).astype(int)

    # 4. Add only missing samples.
    for term, target in zip(active_terms, targets):
        number_to_add = max(0, target - term.sample_count)
        term.add_samples(number_to_add)

    # 5. Check the actual estimated sampling variance.
    sampling_variance = sum(
        term.sample_variance / term.sample_count
        for term in active_terms
    )

    if sampling_variance <= sampling_variance_target:
        break

# 6. Estimate the bias after sampling is sufficiently stable.
bias_estimate = (
    abs(active_terms[-1].mean)
    / (refinement_factor**weak_rate - 1.0)
)

if bias_estimate <= bias_target:
    converged = True
else:
    add_one_finer_level()
    converged = False
```

Real adaptive code also needs safeguards for noisy pilot variance estimates,
zero or nearly zero estimated variances, timing noise, maximum levels, and
maximum sample or cost budgets.

---

## 16. Confidence intervals are related but not identical to RMSE stopping

Under a central limit approximation, a rough 95% sampling confidence interval
is

\[
\widehat{Q}^{\mathrm{ML}}
\pm 1.96\,
\operatorname{SE}(\widehat{Q}^{\mathrm{ML}}).
\]

This interval describes sampling uncertainty around
(\mathbb{E}[Q_L]). It does not automatically include the unknown
discretization bias between (\mathbb{E}[Q_L]) and (\mathbb{E}[Q]).

Also, the paper's RMSE tolerance is not the same as requesting a 95% confidence
interval with half-width (\varepsilon). For a confidence-interval requirement,
the sampling target would need to incorporate the chosen normal quantile. The
paper instead allocates an RMSE budget directly through MSE.

---

## 17. Common mistakes and why they are mistakes

### Mistake 1: stopping when every raw (V_\ell) is small

Raw correction variance does not decrease merely because more samples are
taken. It is the estimator variance (V_\ell/N_\ell) that sampling controls.

### Mistake 2: comparing a variance with (\varepsilon)

Variance has squared units. For an RMSE tolerance (\varepsilon), compare
sampling variance with a multiple of (\varepsilon^2).

### Mistake 3: dividing the full standard error by another (N)

Each term already contributes (V_\ell/N_\ell). The MLMC terms generally have
different counts, so there is no additional global sample count.

### Mistake 4: using different random fields in (Q_\ell-Q_{\ell-1})

This destroys the strong coupling and usually makes the correction variance
much larger.

### Mistake 5: assuming more samples remove the finest-level bias

More samples estimate (\mathbb{E}[Q_L]) more accurately. Only spatial
refinement moves (\mathbb{E}[Q_L]) toward (\mathbb{E}[Q]).

### Mistake 6: checking sampling convergence but not bias convergence

A very precise estimate of the wrong discrete expectation can still have large
total error.

### Mistake 7: using one sample count for every level

This ignores the reason MLMC is efficient. Cheap, noisy terms should generally
receive more samples than expensive, low-variance corrections.

### Mistake 8: assuming the bias estimate is exact

The finest correction mean is a proxy based on an asymptotic convergence model.
It can be noisy or misleading before the meshes enter that regime.

---

## 18. What should be printed in an adaptive run

For each term, a useful summary contains:

- level and correction name;
- current (N_\ell);
- target (N_\ell);
- correction mean (\overline{Y}_\ell);
- raw correction variance (\widehat{V}_\ell);
- term estimator variance (\widehat{V}_\ell/N_\ell);
- average cost per sample (\widehat{C}_\ell).

The global summary should contain:

- MLMC estimate (\sum_\ell\overline{Y}_\ell);
- estimator variance
  (\sum_\ell\widehat{V}_\ell/N_\ell);
- estimator standard error;
- sampling variance target (\varepsilon^2/2);
- bias estimate;
- bias target (\varepsilon/\sqrt{2});
- separate Boolean results for sampling and bias convergence.

For example:

```text
sampling variance = 4.8e-05, target = 5.0e-05: PASS
estimated bias    = 4.0e-02, target = 7.1e-03: FAIL
```

This output means the expected value of the active discrete QoI has been
sampled precisely enough, but the spatial hierarchy is not yet fine enough.
The appropriate next action is to add a finer level, not simply to continue
sampling the same hierarchy.

---

## 19. Final mental model

It is useful to think of MLMC as solving two nested design problems.

### Spatial design problem

Choose a finest level (L) so that

\[
|\mathbb{E}[Q_L-Q]|
\]

is acceptably small. This is the bias problem.

### Sampling design problem

For that active hierarchy, choose (N_0,\ldots,N_L) so that

\[
\sum_{\ell=0}^{L}\frac{V_\ell}{N_\ell}
\]

is acceptably small at minimum cost. This is the sampling problem.

The raw correction variances tell the algorithm where sampling effort is
needed. The divided variances tell the algorithm how uncertain the estimated
expectation still is. The finest correction mean tells the algorithm whether
the spatial hierarchy is likely fine enough.

In one line:

\[
\boxed{
\text{raw variance allocates samples,}
\quad
\text{variance of means stops sampling,}
\quad
\text{finest correction estimates bias.}
}
\]

---

## Reference in the Cliffe et al. paper

The relevant discussion is in Section 2:

- equation (2.3): ordinary Monte Carlo MSE;
- equations (2.5)-(2.7): the MLMC telescoping identity and estimator;
- equation (2.8): MLMC MSE as sampling variance plus squared bias;
- equation (2.9): cost-optimal allocation
  (N_\ell\propto\sqrt{V_\ell/C_\ell});
- the practical six-step adaptive algorithm immediately after Theorem 2.1.

Source: Cliffe, K. A., Giles, M. B., Scheichl, R., and Teckentrup, A. L. (2011),
*Multilevel Monte Carlo methods and applications to elliptic PDEs with random
coefficients*, Computing and Visualization in Science, 14, 3-15.
