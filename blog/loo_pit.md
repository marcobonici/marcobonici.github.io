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

$$
\tilde{y}_i^{(s)}.
$$

This is a **stochastic realization** from the predictive distribution associated with $\theta^{(s)}$. In other words, once we fix the parameter draw $\theta^{(s)}$, we do not simply compute a deterministic summary such as the mean of the likelihood; instead, we actually draw a random value from the likelihood itself.

This point is worth stressing because it is a common source of confusion in posterior predictive checks. People sometimes look at the collection of conditional means, for example the various $\mu^{(s)}$ values obtained from the posterior draws, and think that this collection already represents the posterior predictive distribution. That is not correct.

The set of means $\mu^{(s)}$ only describes how the **center** of the predictive distribution changes across posterior draws. It does not include the intrinsic randomness of the likelihood around that center. The posterior predictive distribution must include both sources of uncertainty:

1. uncertainty in the parameters, represented by the posterior draws $\theta^{(s)}$;
2. randomness in the data-generating process, represented by a fresh random draw from the likelihood for each $\theta^{(s)}$.

So the correct construction is:

- first draw $\theta^{(s)}$ from the posterior;
- then draw $\tilde{y}_i^{(s)}$ from the sampling distribution implied by that parameter draw.

Symbolically, this means

$$
\tilde{y}_i^{(s)} \sim p(\tilde y_i \mid y_{-i}, \theta^{(s)})
$$

in the general leave-one-out setting.

In non-factorizable models, this predictive distribution is again a conditional one, because the left-out observation must be generated given the remaining data $y_{-i}$. Thus, the relevant object is not just a mean vector or best-fit prediction, but a full conditional predictive distribution from which we generate random realizations.

That is exactly what makes posterior predictive checks meaningful: we are comparing the observed data to what the model would actually generate, not merely to a smoothed average prediction.

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

## Part II: LOO-PIT in practice

### 1. From theory to computation
In the first part, we saw that LOO-PIT is based on leave-one-out posterior predictive distributions. For each observation, we want to assess where the observed value falls relative to what the model would have predicted if it had never seen that specific observation during training. Because refitting the model for every single data point is too computationally expensive, we approximate this by using the draws from our full posterior and reweighting them using Pareto-Smoothed Importance Sampling (PSIS).

Let's see how this works in Julia. We will start by importing the necessary packages, setting a random seed for reproducibility, defining a simple Bayesian model using Turing.jl, and generating some synthetic data.

```julia
using Random
using LinearAlgebra
using Turing
using PosteriorStats
using Distributions
using PDMats
using LogExpFunctions
using Plots

# Define the Turing model
@model function myfoo(SL, x)
    a ~ Uniform(-10, 10)
    b ~ Uniform(-10, 10)
    μ = a .* x .+ b
    S = Symmetric(SL * SL')
    # Multivariate Normal observation
    y ~ MvNormal(μ, PDMat(Cholesky(SL)))
    return y
end

# Setup synthetic data
Random.seed!(1234)
n = 30
R = rand(LKJCholesky(n, 1))
s = rand(filldist(truncated(TDist(7); lower=0), n))
SL = LowerTriangular(Diagonal(s) * R.L) + 0.2I
S = Symmetric(SL * SL')

x = Array(LinRange(-7, 7, n))
a_true = 1.0
b_true = -3.0
y = rand(MvNormal(a_true .* x .+ b_true, S))
```
In this snippet, we have defined a model with a multivariate normal likelihood. The parameters `a` and `b` dictate the mean of the distribution, while the covariance `S` is constructed from a lower-triangular matrix `SL`.

### 2. Fitting the model
Before we can perform any leave-one-out approximations, we need to fit the model to the full dataset. We use Turing to sample from the posterior and then generate full-posterior predictive draws.

```julia
# Sample from the model using 4 parallel chains
chns = sample(myfoo(SL, x) | (; y), NUTS(), MCMCThreads(), 200, 4)

# Generate posterior predictions
y_pred = predict(myfoo(SL, x), chns)
```
Here, we use the No-U-Turn Sampler (NUTS) to obtain 200 draws across 4 chains. We also generate the predictive draws `y_pred` which simulate new data from the fitted model.

### 3. Pointwise log-likelihood
To reweight our posterior draws, we need to know how much each draw "liked" each observation. This is captured by the pointwise log-likelihood.

```julia
# Extract parameter samples to build the pointwise log-likelihood representation
a_samples = chns[:a].data
b_samples = chns[:b].data

# Compute the deterministic mean array for all draws and chains
# shape: (draws, chains, n)
μ_array = a_samples .* reshape(x, 1, 1, n) .+ b_samples
S_pd = PDMat(Cholesky(SL))

# Construct pointwise conditional log-likelihoods
# We instantiate the MvNormal objects for each slice (each posterior draw)
dists = map(μ -> MvNormal(μ, S_pd), eachslice(μ_array; dims=(1, 2)))

# Calculate log-likelihood array of shape (draws, chains, n) 
log_like = PosteriorStats.pointwise_conditional_loglikelihoods(y, dists)
```
This code manually calculates the pointwise conditional log-likelihood for every posterior draw and every data point. The resulting `log_like` array tells us the log-probability of each observed data point $y_i$ under each posterior sample.

### 4. PSIS-based leave-one-out reweighting
Exact refitting for every left-out point would be incredibly slow. Instead, we approximate the leave-one-out posterior using PSIS. We will treat the PSIS smoothing algorithm itself as a black box and simply extract the smoothed log-weights it produces.

```julia
# Compute LOO using PosteriorStats.jl to get the PSIS weights
loo_res = loo(log_like)

# Extract the smoothed log-weights
log_weights = loo_res.psis_result.log_weights
```
The `log_weights` array now contains the importance weights for each draw and each observation. Draws that fit an observation $y_i$ too well will be down-weighted for that observation, simulating what the posterior would look like if $y_i$ had been left out.

![](https://github.com/user-attachments/assets/3cb5db55-4ae6-458b-b849-5d728e019e3a)

### 5. Building leave-one-out posterior predictive draws
With our PSIS log-weights in hand, we can transform our full-posterior predictive draws into leave-one-out predictive draws. First, let's extract our simulated predictive draws into a clean array format.

```julia
# Extract predictions for y into an array of shape (draws, chains, n)
y_pred_data = y_pred[[Symbol("y[$i]") for i in 1:n]].value.data
y_pred_array = permutedims(y_pred_data, (1, 3, 2))

ndraws, nchains, nobs = size(y_pred_array)
```
For any observation $i$, `y_pred_array[:, :, i]` gives us the Monte Carlo draws of the prediction for $y_i$ under the full posterior, and `log_weights[:, :, i]` gives the corresponding PSIS weights needed to shift those predictions to the leave-one-out distribution.

### 6. Two ways to evaluate the predictive CDF
To compute the LOO-PIT values, we need to evaluate the Cumulative Distribution Function (CDF) of the leave-one-out predictive distribution at the observed value $y_i$. We can do this in two ways:

1. **Analytically:** For this specific model, the observation model is a Gaussian. This means the predictive distribution conditional on the parameters is tractable, and we can evaluate the CDF exactly.
2. **By Monte Carlo:** We can approximate the CDF by counting the proportion of predictive draws that fall below the observed value. This method is much more general and works for any model, even when the CDF is not analytically tractable.

Let's look at both approaches.

#### 6a. Analytical CDF for this example
Because our likelihood is a multivariate normal, the marginal distribution for observation $y_i$ under a specific parameter draw is simply a univariate normal $\mathcal{N}(\mu_i^{(s)}, \sigma_i)$. We can calculate the exact Gaussian CDF for each draw, and then take the weighted average using our PSIS weights.

```julia
# The standard deviations are on the diagonal of the covariance matrix S
σ_array = sqrt.(diag(S))
pitvals_exact = zeros(nobs)

for i in 1:nobs
    log_w_i = log_weights[:, :, i]
    
    # Extract μ and σ for observation i
    μ_i = μ_array[:, :, i]
    σ_i = σ_array[i]
    
    # Evaluate the exact Gaussian CDF at the observed value y[i] for each draw
    cdf_draws = cdf.(Normal.(μ_i, σ_i), y[i])
    
    # The exact LOO-PIT is the weighted sum of these exact CDFs
    # Because weights are logged, we use logsumexp
    exact_cdf = exp(logsumexp(log_w_i .+ log.(cdf_draws)))
    
    pitvals_exact[i] = exact_cdf
end
```
By evaluating the exact CDF, we eliminate the sampling noise that comes from drawing random predictive variables. This "semi-analytic" method provides a very smooth and accurate estimate.

#### 6b. Monte Carlo approximation to the CDF
What if our likelihood wasn't so nicely tractable? Instead of using a closed-form formula, we can estimate the CDF empirically. We simply look at our generated predictive draws $\tilde{y}_i$ and check which ones satisfy the condition $\tilde{y}_i \leq y_i$. We then sum the PSIS weights of only those matching draws.

```julia
pitvals_manual = zeros(nobs)

for i in 1:nobs
    # 1. Grab the full-posterior predictive draws for this observation
    y_pred_i = y_pred_array[:, :, i] 
    
    # 2. Grab the PSIS log-weights for this observation
    log_w_i = log_weights[:, :, i]
    
    # 3. Find which predictive draws are less than or equal to the actual observed value
    indicator_mask = y_pred_i .<= y[i]
    
    if any(indicator_mask)
        # Sum the log-weights of the matching draws using logsumexp
        cdf_estimate = exp(logsumexp(log_w_i[indicator_mask]))
    else
        cdf_estimate = 0.0
    end
    
    pitvals_manual[i] = cdf_estimate
end
```
Because the total PSIS weights for a given observation sum to 1, summing the weights of the draws that fall below $y_i$ directly yields the estimated empirical CDF fraction. This is the broadly applicable Monte Carlo route you should remember.

![](https://github.com/user-attachments/assets/b766befa-ece2-4550-98ba-0eb25fba6f6c)

### 7. Analytical versus Monte Carlo
Let's compare the results of the two methods for the first few observations:

![](https://github.com/user-attachments/assets/79fd5161-0623-4f16-8bfc-493c1b92b501)

The exact analytical CDF is mathematically superior, but the Monte Carlo method closely approximates it. The small differences (around ~0.05 to ~0.10) are entirely normal and are due to the finite number of posterior predictive draws (200 draws per chain). In practice, the Monte Carlo approximation is accurate enough for diagnostic purposes and is far easier to generalize to complex models where the analytical CDF is unknown.



### 8. Visualizing calibration with KDEs
By running the loop above, we have collected a LOO-PIT value for every single observation in our dataset. If the model is well-calibrated, the distribution of these values should look roughly uniform between 0 and 1. 

A powerful way to visualize this is by comparing the Kernel Density Estimate (KDE) of our actual `pitvals_manual` against an ensemble of KDEs generated from perfectly uniform simulated data of the same sample size. To ensure a fair comparison and avoid plotting densities outside the valid $[0, 1]$ interval, we use a boundary-corrected KDE approach.

```julia
kde_bw = 0.15 # Tuned bandwidth for smoothness on N=30 samples

p = plot(title="LOO-PIT KDE vs Uniform Reference",
         xlabel="LOO-PIT value", ylabel="Density",
         xlims=(0, 1), ylims=(0, 2.5),
         legend=:topright)

# Helper function to compute bounded KDE over [0, 1]
function get_bounded_kde(data, bw)
    k = PosteriorStats.kde_reflected(data, bounds=(0, 1), bandwidth=bw)
    # Only keep points inside [0, 1]
    mask = 0.0 .<= k.x .<= 1.0
    return k.x[mask], k.density[mask]
end

# Plot 100 uniform reference KDEs in the background
for i in 1:100
    ref_sample = rand(Uniform(0, 1), nobs)
    rx, ry = get_bounded_kde(ref_sample, kde_bw)
    plot!(p, rx, ry, color=:lightblue, alpha=0.3, linewidth=1, 
          label=i==1 ? "Uniform Reference" : "")
end

# Plot actual LOO-PIT KDE in the foreground
ox, oy = get_bounded_kde(pitvals_manual, kde_bw)
plot!(p, ox, oy, color=:darkblue, linewidth=3, label="Observed LOO-PIT")
```

![](https://github.com/user-attachments/assets/afbf3240-02d4-4b8a-a192-07f003c0ee35)

In this plot, the thin light-blue lines represent the expected sampling variance for a truly uniform distribution at our finite sample size. If our highlighted dark-blue curve stays broadly within that background envelope, we can be confident the model is well-calibrated. Deviations outside that reference envelope would indicate a lack of calibration (e.g., under-dispersion or over-dispersion). 

### 9. Comparing with PosteriorStats.jl
Finally, we can compare our pedagogical manual implementation with the robust, production-quality implementation provided by `PosteriorStats.jl`. The package automates the Monte Carlo workflow we just built.

```julia
# Use the automated PosteriorStats.jl function
pitvals_automated = loo_pit(y, y_pred_array, log_weights)

println("Automated LOO-PIT values:           ", round.(pitvals_automated[1:5], digits=4))

max_diff_mc = maximum(abs.(pitvals_manual .- pitvals_automated))
println("Max difference (Automated vs MC):   ", round(max_diff_mc, digits=8))
```
Output:
```
Automated LOO-PIT values:           [0.6800, 0.1002, 0.5069, 0.5040, 0.0262]
Max difference (Automated vs MC):   0.0
```
The maximum difference is exactly zero! Our manual Monte Carlo implementation perfectly matches the package. 

While our manual code unpacks the underlying logic for educational purposes, `PosteriorStats.jl` handles edge cases, shape checks, and optimizations under the hood. In your real Bayesian analyses, the package implementation is the one you should trust and use.
