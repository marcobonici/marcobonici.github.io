@def title = "From MCMC Draws to PSIS-LOO and LOO-PIT"
@def author = "Marco Bonici"
@def hasmath = true

# From MCMC Draws to PSIS-LOO and LOO-PIT

This note provides a guided walkthrough from a fitted Bayesian model to its production-quality diagnostics. We will interleave the mathematical theory with a concrete Julia implementation using `Turing.jl` and `PosteriorStats.jl`, exploring exactly how we go from raw MCMC draws to:

- **PSIS-LOO**, which estimates out-of-sample predictive performance without refitting the model $N$ times.
- **LOO-PIT**, which checks whether the model is well calibrated.

The central idea is simple: after fitting the model once to the full dataset, we reuse the posterior draws in a clever way to approximate what would have happened if we had left each observation out.

## 1. The Setup: Model and Data

In many scientific applications, such as cosmology, we have one dataset in hand and no stream of "future" observations to test against. We want to know if the model has learned something general or merely overfitted the noise.

Let's define a non-trivial multivariate normal model in `Turing.jl` and generate some synthetic data to work with.

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

## 2. Fitting the Model

First, we obtain a posterior distribution for the model parameters. MCMC gives us $S$ draws summarizing what values are plausible under our assumptions.

```julia
# Sample from the model using 8 parallel chains
chns = sample(myfoo(SL, x) | (; y), NUTS(), MCMCThreads(), 2000, 8)

# Generate posterior predictions: stochastic realizations from the likelihood
y_pred = predict(myfoo(SL, x), chns)
```

We stress that `y_pred` contains **stochastic realizations**. It is not enough to look at the conditional means $\mu^{(s)}$; we must actually draw random values from the likelihood to account for the intrinsic randomness of the data-generating process.

## 3. Pointwise Likelihood

Leave-one-out (LOO) cross-validation requires us to evaluate how well each individual data point $y_i$ is predicted when it is excluded from the training. To approximate this, we first need the **pointwise log-likelihood**: the probability of each observed datum $y_i$ under each posterior draw $\theta^{(s)}$.

In non-factorizable models (like those with a dense covariance matrix), the relevant quantity is the conditional density:
\[
\ell_i^{(s)} = \log p(y_i \mid y_{-i}, \theta^{(s)}).
\]

```julia
# Extract parameter samples
a_samples = chns[:a].data
b_samples = chns[:b].data

# Compute the mean array for all draws and chains (draws, chains, n)
μ_array = a_samples .* reshape(x, 1, 1, n) .+ b_samples
S_pd = PDMat(Cholesky(SL))

# Instantiate distributions for each posterior slice
dists = map(μ -> MvNormal(μ, S_pd), eachslice(μ_array; dims=(1, 2)))

# Calculate pointwise conditional log-likelihoods
log_like = PosteriorStats.pointwise_conditional_loglikelihoods(y, dists)
```

## 4. PSIS Reweighting instead of Refitting

The leave-one-out posterior for observation $i$ is $p(\theta \mid y_{-i})$. Instead of refitting $N$ times, we use **Pareto Smoothed Importance Sampling (PSIS)** to reweight our existing draws. Draws that made $y_i$ extremely likely receive less relative weight in the LOO approximation, because they were "too influenced" by $y_i$ during the full fit.

```julia
# Compute LOO to obtain smoothed importance weights
loo_res = loo(log_like)
log_weights = loo_res.psis_result.log_weights
```

PSIS stabilizes these weights by fitting a generalized Pareto distribution to the upper tail. We can check the reliability of this approximation using the Pareto $k$ diagnostic.

![](https://github.com/user-attachments/assets/d963f905-1680-4dca-8d79-835f576c9037)

If $k < 0.7$, the importance weights are stable and our LOO approximation is trustworthy.

## 5. From Weights to Predictive Accuracy (ELPD)

The **Expected Log Predictive Density (ELPD)** quantifies out-of-sample performance. For each point $y_i$, we average the conditional likelihoods using our smoothed weights:
\[
\widehat{p}(y_i \mid y_{-i}) = \sum_{s=1}^S \tilde{w}_i^{(s)} \, p(y_i \mid y_{-i}, \theta^{(s)}).
\]
Summing these log-probabilities across all observations gives the total $\widehat{\mathrm{ELPD}}_{\mathrm{LOO}}$. This is calculated automatically inside the `loo_res` object we just created.

## 6. Calibration and the LOO-PIT

While ELPD tells us about accuracy, **LOO-PIT (Probability Integral Transform)** tells us about **calibration**: does the model assign realistic uncertainty?

We compare each observed value $y_i$ to its leave-one-out predictive distribution $p(\tilde{y}_i \mid y_{-i})$. We want to evaluate the Cumulative Distribution Function (CDF) at the observed point:
\[
\widehat{\mathrm{PIT}}_i = P(\tilde{y}_i \le y_i \mid y_{-i}).
\]

There are two ways to compute this CDF in practice.

### 6a. The Monte Carlo Approach (General)

The most general method is to count what fraction of our simulated predictive draws fall below the observed value, reweighted by our PSIS weights.

```julia
# Clean predictions into (draws, chains, n)
y_pred_array = permutedims(y_pred[[Symbol("y[$i]") for i in 1:n]].value.data, (1, 3, 2))

pitvals_manual = zeros(n)
for i in 1:n
    y_pred_i = y_pred_array[:, :, i] 
    log_w_i = log_weights[:, :, i]
    
    # Identify which draws fall below the observation
    mask = y_pred_i .<= y[i]
    
    # Estimate the CDF as the sum of the weights of those draws
    pitvals_manual[i] = any(mask) ? exp(logsumexp(log_w_i[mask])) : 0.0
end
```

This counts the "mass" of the predictive distribution to the left of our observation.

![](https://github.com/user-attachments/assets/b766befa-ece2-4550-98ba-0eb25fba6f6c)

### 6b. The Analytical Approach (Exact)

If the likelihood is tractable (like our Gaussian model), we can evaluate the exact CDF for each draw instead of drawing random $\tilde{y}_i$ values. This "semi-analytic" approach is more precise as it removes sampling noise.

```julia
σ_array = sqrt.(diag(S))
pitvals_exact = zeros(n)

for i in 1:n
    log_w_i = log_weights[:, :, i]
    μ_i = μ_array[:, :, i]
    σ_i = σ_array[i]
    
    # Weighted average of exact Gaussian CDFs
    cdf_draws = cdf.(Normal.(μ_i, σ_i), y[i])
    pitvals_exact[i] = exp(logsumexp(log_w_i .+ log.(cdf_draws)))
end
```

## 7. Comparing the Methods

Let's see how our Monte Carlo approximation holds up against the exact analytic benchmark.

```julia
println("Exact Semi-Analytic LOO-PIT (first 5): ", round.(pitvals_exact[1:5], digits=4))
println("Manual Monte Carlo LOO-PIT (first 5): ", round.(pitvals_manual[1:5], digits=4))
```

The difference between these two is typically small (~0.05) and scales with the number of predictive draws. The Monte Carlo method is the one to remember, as it works even when no closed-form CDF exists.

![](https://github.com/user-attachments/assets/79fd5161-0623-4f16-8bfc-493c1b92b501)

## 8. Visualizing Calibration with KDEs

If the model is well-calibrated, the LOO-PIT values should be uniformly distributed. We can visualize this by plotting the Kernel Density Estimate (KDE) of our values against an ensemble of KDEs generated from truly uniform data. To ensure a fair comparison and avoid plotting densities outside the valid $[0, 1]$ interval, we use a boundary-corrected KDE approach.

```julia
kde_bw = 0.15 # Tuned bandwidth for smoothness on N=30 samples

p = plot(title="LOO-PIT KDE vs uniform reference",
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
          label=i==1 ? "Uniform reference" : "")
end

# Plot actual LOO-PIT KDE in the foreground
ox, oy = get_bounded_kde(pitvals_manual, kde_bw)
plot!(p, ox, oy, color=:darkblue, linewidth=3, label="Observed LOO-PIT")
```

![](https://github.com/user-attachments/assets/afbf3240-02d4-4b8a-a192-07f003c0ee35)

If our solid dark-blue curve stays within the light-blue "cloud" of uniform reference samples, our model's uncertainty statements are broadly consistent with the data.

## 9. Quantifying Uniformity: The Kolmogorov-Smirnov Test

Visual inspection is often the most intuitive diagnostic, but we can also quantify the departure from uniformity using a statistical test. The **Kolmogorov-Smirnov (KS) test** is a standard choice for this purpose.

The null hypothesis ($H_0$) is that the LOO-PIT values are drawn from a $\text{Uniform}(0, 1)$ distribution. A small p-value (typically $< 0.05$) suggests that the observed values are inconsistent with perfect calibration, indicating that the model may be over-confident or under-confident.

```julia
using HypothesisTests

# Perform the KS test against the Uniform(0, 1) distribution
ks_test = ExactOneSampleKSTest(pitvals_manual, Uniform(0, 1))

println("LOO-PIT KS test p-value: ", pvalue(ks_test))
```

It is important to remember that a large p-value does not *prove* the model is perfectly calibrated—it only means we haven't found strong evidence of miscalibration at our current sample size. Conversely, with very large datasets, even tiny, practically irrelevant deviations from uniformity might trigger a small p-value. Therefore, the KS test should always be interpreted as a quantitative complement to the visual KDE diagnostic.

## 10. Validation with PosteriorStats.jl

While we built these diagnostics manually for educational purposes, `PosteriorStats.jl` provides a robust, optimized implementation that handles these steps automatically.

```julia
# Automated package call
pitvals_automated = loo_pit(y, y_pred_array, log_weights)

# Validation: our manual MC implementation matches perfectly!
max_diff = maximum(abs.(pitvals_manual .- pitvals_automated))
println("Max difference (Automated vs Manual MC): ", max_diff)
```

In practice, the package implementation is the one you should trust for production work, while the manual steps we've explored provide the intuition for what is happening under the hood.
