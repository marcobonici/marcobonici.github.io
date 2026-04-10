@def title = "From MCMC Draws to PSIS-LOO and LOO-PIT"
@def author = "Marco Bonici"
@def hasmath = true

# From MCMC Draws to PSIS-LOO and LOO-PIT

This note explains how we go from a fitted Bayesian model to two important diagnostics:

- **PSIS-LOO**, which estimates out-of-sample predictive performance without refitting the model $N$ times.
- **LOO-PIT**, which checks whether the model is well calibrated.

The central idea is simple: after fitting the model once to the full dataset, we reuse the posterior draws in a clever way to approximate what would have happened if we had left each observation out.

## Why we need this

Suppose we observe data

\[
y = (y_1, y_2, \dots, y_N).
\]

After fitting a Bayesian model to all $N$ observations using MCMC, we obtain $S$ draws from the posterior distribution:

\[
\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(S)} \sim p(\theta \mid y).
\]

These draws are the starting point for everything that follows.

A good way to think about them is this: each draw $\theta^{(s)}$ represents one plausible version of the world, given the data. The posterior is therefore not a single best-fit answer, but a collection of plausible parameter values.

Now imagine we want to answer the following question:

> If observation $y_i$ had not been included in the fit, how well would the model have predicted it?

The most direct strategy would be:

1. Remove $y_i$.
2. Refit the model using the remaining $N-1$ observations.
3. Evaluate how well that refitted model predicts $y_i$.
4. Repeat this for every $i = 1, \dots, N$.

That is exact leave-one-out cross-validation. It is conceptually simple, but often computationally prohibitive, because it requires fitting the model $N$ separate times.

PSIS-LOO gives us a much cheaper approximation. Instead of refitting the model $N$ times, it reuses the draws from the full-data posterior and adjusts their importance so that they behave approximately like draws from the leave-one-out posterior.

## Phase 1: The starting point

Before any leave-one-out approximation happens, we fit the model once using the full dataset.

The result is a collection of posterior draws:

\[
\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(S)}.
\]

These draws summarize our uncertainty about the model parameters after observing all the data.

Here is the key intuition:

- Some posterior draws fit observation $y_i$ extremely well.
- Other draws fit $y_i$ less well.
- If we now pretend that $y_i$ had never been observed, then draws that were strongly influenced by $y_i$ should count less.

This is exactly the role of importance sampling: it lets us reweight the existing posterior draws so they better represent the posterior we would have obtained without $y_i$.

## Phase 2: Reweighting instead of refitting

The leave-one-out posterior for observation $i$ is

\[
p(\theta \mid y_{-i}),
\]

where $y_{-i}$ means all observations except $y_i$.

We do not want to sample from this distribution by rerunning MCMC. Instead, we approximate it using the full posterior draws $p(\theta \mid y)$ together with importance weights.

### Step 2a: Evaluate the relevant likelihood

For each posterior draw $s$ and each observation $i$, we evaluate how compatible $y_i$ is with that draw.

In models with conditionally independent observations, this is simply

\[
p(y_i \mid \theta^{(s)}).
\]

In non-factorizable models, where observations are not conditionally independent, we must instead use the exact conditional density

\[
p(y_i \mid y_{-i}, \theta^{(s)}).
\]

To keep the notation general, define

\[
\ell_i^{(s)} = \log p(y_i \mid y_{-i}, \theta^{(s)}).
\]

### Step 2b: Compute the raw importance weights

To approximate the leave-one-out posterior, the raw importance weight for draw $s$ is

\[
w_i^{(s)} = \frac{1}{p(y_i \mid y_{-i}, \theta^{(s)})}.
\]

Equivalently, in log form,

\[
\log w_i^{(s)} = -\log p(y_i \mid y_{-i}, \theta^{(s)}).
\]

Why does this make sense? If a particular draw makes $y_i$ extremely likely, then that draw may have been heavily influenced by having seen $y_i$ during fitting. When we pretend $y_i$ was left out, such draws should be downweighted.

### Step 2c: Stabilize the weights with PSIS

Raw importance weights are often unstable. A few of them can become extremely large, and then the approximation gets dominated by only a small number of posterior draws.

Pareto Smoothed Importance Sampling, or **PSIS**, fixes this by smoothing the upper tail of the weight distribution:

1. Sort the raw weights for observation $i$.
2. Identify the largest weights.
3. Fit a generalized Pareto distribution to that upper tail.
4. Replace the most extreme raw weights with smoothed values from the fitted Pareto tail.
5. Normalize the smoothed weights so they sum to $1$.

The final normalized smoothed weights are denoted

\[
\tilde{w}_i^{(s)}.
\]

These are the weights used in both the predictive performance calculation and the calibration diagnostic.

## Phase 3: Predictive performance through ELPD

Now we want to quantify how well the model predicts left-out data.

The main quantity is the **expected log predictive density**, or **ELPD**. Larger values indicate better out-of-sample predictive performance.

### Step 3a: Pointwise leave-one-out predictive density

For each observation $y_i$, we approximate its leave-one-out predictive density by averaging across posterior draws using the smoothed weights:

\[
\widehat{p}(y_i \mid y_{-i}) = \sum_{s=1}^S \tilde{w}_i^{(s)} \, p(y_i \mid y_{-i}, \theta^{(s)}).
\]

The pointwise leave-one-out contribution is then

\[
\widehat{\mathrm{elpd}}_{\mathrm{LOO}, i}
=
\log \widehat{p}(y_i \mid y_{-i}).
\]

In practice, this is usually computed using the LogSumExp trick for numerical stability:

\[
\widehat{\mathrm{elpd}}_{\mathrm{LOO}, i}
=
\operatorname{logSumExp}_{s=1}^S
\left(
\log \tilde{w}_i^{(s)}
+
\log p(y_i \mid y_{-i}, \theta^{(s)})
\right).
\]

### Step 3b: Total leave-one-out ELPD

We sum these pointwise contributions over all observations:

\[
\widehat{\mathrm{ELPD}}_{\mathrm{LOO}}
=
\sum_{i=1}^N
\widehat{\mathrm{elpd}}_{\mathrm{LOO}, i}.
\]

This gives a single score for the model’s leave-one-out predictive accuracy.

If two models are being compared, the model with the larger $\widehat{\mathrm{ELPD}}_{\mathrm{LOO}}$ is generally the better out-of-sample predictor.

## Phase 4: Calibration through LOO-PIT

Predictive accuracy is only part of the story. A model can score well and still misrepresent uncertainty.

Calibration asks a different question:

> Does the model assign realistic uncertainty to new observations?

This is what **LOO-PIT** is designed to check. PIT stands for **Probability Integral Transform**.

The basic idea is to compare each observed value $y_i$ to its leave-one-out predictive distribution. If the model is well calibrated, the observation should look like a typical draw from that predictive distribution.

### Step 4a: Generate posterior predictive draws

For each posterior draw $\theta^{(s)}$, generate a synthetic observation

\[
\tilde{y}_i^{(s)}.
\]

Conceptually, this is a draw from the predictive distribution associated with $\theta^{(s)}$. In non-factorizable models, the relevant quantity is the conditional predictive distribution given $y_{-i}$.

### Step 4b: Compare synthetic data to the observed data

For each draw $s$, define the indicator

\[
I(\tilde{y}_i^{(s)} \le y_i),
\]

which equals:

- $1$ if the simulated value is less than or equal to the observed value,
- $0$ otherwise.

This records whether the simulated draw falls below the actual observation.

### Step 4c: Average with the same LOO weights

We then average those indicators using the same smoothed importance weights from PSIS:

\[
\widehat{\mathrm{PIT}}_i
=
\sum_{s=1}^S
\tilde{w}_i^{(s)}
I(\tilde{y}_i^{(s)} \le y_i).
\]

This gives one number between $0$ and $1$ for each observation $i$. It is an estimate of the leave-one-out predictive CDF evaluated at the observed data point.

## How to interpret LOO-PIT

After computing $\widehat{\mathrm{PIT}}_i$ for all $i = 1, \dots, N$, we examine their distribution.

If the model is well calibrated, these values should be approximately uniformly distributed on $[0,1]$.

Common patterns are:

- **Flat histogram**: calibration is good.
- **U-shaped histogram**: the predictive distributions are too narrow, so the model is under-dispersed or overconfident.
- **Hump-shaped histogram**: the predictive distributions are too wide, so the model is over-dispersed or underconfident.
- **Left- or right-skewed histogram**: the model shows systematic bias.

In short:

- **ELPD** tells us how well the model predicts unseen data.
- **LOO-PIT** tells us whether the model’s uncertainty statements are trustworthy.

## Practical note

PSIS-LOO works well only when the importance weights are sufficiently stable. In practice, software usually reports the Pareto shape diagnostic $k$:

- Small $k$: the approximation is reliable.
- Large $k$: PSIS-LOO may be unstable, and exact refitting or $K$-fold cross-validation may be safer.

This is why tools such as ArviZ or the `loo` package report Pareto $k$ values alongside the LOO estimates.
