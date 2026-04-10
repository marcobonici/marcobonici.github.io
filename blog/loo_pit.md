@def title = "From MCMC Draws to PSIS-LOO and LOO-PIT"
@def author = "Marco Bonici"
@def hasmath = true

# From MCMC Draws to PSIS-LOO and LOO-PIT

This note explains how we go from a fitted Bayesian model to two important diagnostics:

- **PSIS-LOO**, which estimates out-of-sample predictive performance without refitting the model $N$ times.
- **LOO-PIT**, which checks whether the model is well calibrated.

The central idea is simple: after fitting the model once to the full dataset, we reuse the posterior draws in a clever way to approximate what would have happened if we had left each observation out.

## Why we need this

After fitting a Bayesian model, we obtain a posterior distribution for its parameters. If our observed data are

\[
y = (y_1, y_2, \dots, y_N),
\]

then MCMC gives us $S$ draws of the form

\[
\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(S)} \sim p(\theta \mid y).
\]

These draws summarize what parameter values are plausible **under the assumptions of the model**.

This point is important: a posterior distribution tells us what we should believe about the parameters *if we take the model seriously*. But in practice, that is not the end of the story. Once a model has been fitted, we usually want to ask broader questions about the model itself.

In particular, there are two natural questions:

1. **Is the model fit good enough?**  
   That is, does the model describe the observed data in a reasonable way? Are its predictions and uncertainty statements compatible with what we actually see?

2. **If we have several models, which one is better?**  
   That is, which model is expected to predict new, unseen data more accurately?

These are related questions, but they are not identical.

A model can fit the observed data reasonably well and still be worse than a competing model for prediction. Conversely, two models may have similar predictive accuracy, but one may be better calibrated or easier to interpret. So after fitting a Bayesian model, we usually want tools that help us assess both:

- **absolute adequacy**, meaning whether the model is broadly consistent with the data;
- **relative predictive performance**, meaning how well it predicts compared with alternatives.

This is where leave-one-out methods become useful.

In some areas, the motivation for out-of-sample testing is very natural. For example, in a time-series problem such as stock-market forecasting, we can fit a model using data up to a given day and then test it on data from the following week. In that setting, the distinction between training data and future data is built directly into the problem.

In many scientific applications, however, we do not have that luxury. In cosmology, for instance, we usually have one dataset in hand and no stream of future observations arriving in the same sequential way. We still want to know whether the model has learned something general about the underlying phenomenon, rather than merely adapting itself to the particular data sample we observed, but we cannot test this by waiting for “next week’s universe”.

That is exactly why leave-one-out cross-validation is so useful. It gives us a principled way to approximate out-of-sample predictive assessment even when no genuinely future dataset is available.

The idea is simple. For each observation $y_i$, we temporarily pretend that it was not observed, fit the model using the remaining data $y_{-i}$, and then ask how well that refitted model predicts the held-out point. Repeating this for every $i = 1, \dots, N$ gives a systematic picture of how the model performs on data points that were not used in the corresponding fit.

This leave-one-out perspective helps us answer two different questions.

- **How accurate are the predictions?**  
  This leads to quantities such as the leave-one-out expected log predictive density, or LOO-ELPD, which is especially useful when comparing several models.

- **Are the predictive distributions calibrated?**  
  This leads to diagnostics such as LOO-PIT, which help us assess whether the model’s uncertainty is realistic, too narrow, too wide, or systematically biased.

The challenge is computational. Exact leave-one-out cross-validation would require refitting the Bayesian model $N$ separate times, which is often far too expensive when each fit already requires a full MCMC run.

PSIS-LOO provides a practical shortcut. Instead of refitting the model over and over, it reuses the posterior draws from the full-data fit and reweights them so that they approximately behave like draws from the leave-one-out posterior. In this way, we can approximate out-of-sample predictive performance and calibration without paying the full cost of exact leave-one-out refitting.

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

We begin with the most general situation, namely the case in which the likelihood cannot be factorized into independent contributions from each observation.

In that setting, when we leave out observation $y_i$, the relevant quantity is not an isolated term such as $p(y_i \mid \theta^{(s)})$, but the exact conditional density of the left-out datum given all the others:

$$
p(y_i \mid y_{-i}, \theta^{(s)}).
$$

This is the quantity we must evaluate for each posterior draw $\theta^{(s)}$ and each observation $i$.

This general case is especially important in cosmology. In many cosmological analyses, the likelihood is not factorizable because the data are modeled jointly through a multivariate distribution, often with a dense covariance matrix. In such cases, the contribution of one data point cannot be treated as independent of the others, so leave-one-out calculations must be based on the full conditional density rather than on a simple pointwise likelihood term.

There is, of course, a simpler mathematical expression in models with conditionally independent observations. In those cases, the likelihood factorizes and the relevant term reduces to

$$
p(y_i \mid \theta^{(s)}).
$$

However, to keep the discussion focused on the setting we usually care about, we will work only with the general non-factorizable case. It automatically includes the more familiar factorizable setting as a simpler special case.

To simplify notation, define

$$
\ell_i^{(s)} = \log p(y_i \mid y_{-i}, \theta^{(s)}).
$$

This log-conditional density is the basic building block for the importance weights and all the leave-one-out quantities that follow. We will later work out the math for the most common case of a multivariate normal distribution with a dense covariance matrix.

### Step 2b: Compute the raw importance weights

We now want to use the posterior based on the full dataset as a proposal distribution to approximate the leave-one-out posterior.

The target distribution is

$$
p(\theta \mid y_{-i}),
$$

while the proposal distribution is

$$
p(\theta \mid y).
$$

In importance sampling, the ideal weight for a draw $\theta^{(s)}$ is proportional to the ratio

$$
r_i(\theta^{(s)}) = \frac{p(\theta^{(s)} \mid y_{-i})}{p(\theta^{(s)} \mid y)}.
$$

Let us now expand this ratio using Bayes' theorem.

For the leave-one-out posterior, we have

$$
p(\theta \mid y_{-i}) = \frac{p(y_{-i} \mid \theta)\,p(\theta)}{p(y_{-i})}.
$$

For the full-data posterior, we have

$$
p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)}.
$$

Taking the ratio gives

$$
\frac{p(\theta \mid y_{-i})}{p(\theta \mid y)}
=
\frac{p(y_{-i} \mid \theta)\,p(\theta)}{p(y_{-i})}
\cdot
\frac{p(y)}{p(y \mid \theta)\,p(\theta)}.
$$

At this point, the prior $p(\theta)$ cancels exactly, because it is the same in both the leave-one-out posterior and the full-data posterior. This is an important point: the reweighting is not driven by a change in the prior, but by the fact that one observation has been removed from the likelihood.

Now use the decomposition

$$
p(y \mid \theta) = p(y_i \mid y_{-i}, \theta)\,p(y_{-i} \mid \theta).
$$

Substituting this into the ratio gives

$$
\frac{p(\theta \mid y_{-i})}{p(\theta \mid y)}
=
\frac{p(y)}{p(y_{-i})}
\cdot
\frac{1}{p(y_i \mid y_{-i}, \theta)}.
$$

So, strictly speaking, the importance ratio is

$$
r_i(\theta^{(s)}) =
\frac{p(y)}{p(y_{-i})}
\cdot
\frac{1}{p(y_i \mid y_{-i}, \theta^{(s)})}.
$$

The first factor,

$$
\frac{p(y)}{p(y_{-i})},
$$

involves the marginal likelihoods, or evidences. This factor does **not** cancel algebraically. However, it does not depend on $\theta$, so it is the same for every posterior draw $s$.

That is why, in practice, we do not need to keep it. Importance sampling weights are normalized at the end of the procedure, so multiplying every weight by the same overall constant has no effect on the final normalized weights.

For this reason, we work with weights only up to proportionality:

$$
w_i^{(s)} \propto \frac{1}{p(y_i \mid y_{-i}, \theta^{(s)})}.
$$

Thus, the raw importance weight for draw $s$ is taken to be

$$
w_i^{(s)} = \frac{1}{p(y_i \mid y_{-i}, \theta^{(s)})},
$$

understanding that this is the unnormalized weight, with the evidence ratio omitted because it is common to all draws.

Equivalently, in log form,

$$
\log w_i^{(s)} = -\log p(y_i \mid y_{-i}, \theta^{(s)}).
$$

This expression also gives useful intuition. If a particular draw makes the left-out observation $y_i$ extremely likely, then that draw was especially compatible with a posterior that had already seen $y_i$. When we try to approximate the posterior in which $y_i$ was *not* included, such draws should receive less relative weight. Conversely, draws for which $y_i$ was less influential receive more weight in the leave-one-out approximation.

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
