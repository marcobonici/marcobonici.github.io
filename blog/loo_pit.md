@def title = "From MCMC Draws to PSIS-LOO and LOO-PIT"
@def author = "Marco Bonici"
@def hasmath = true

# From MCMC Draws to PSIS-LOO and LOO-PIT

After fitting a Bayesian model, we usually want to answer two different questions.

- How well does the model predict unseen data?
- Does the model assign uncertainty in a realistic way?

These are related, but they are not the same. A model can have reasonable predictive accuracy and still be poorly calibrated, for example by being too confident or too conservative. This is why it is useful to study both **PSIS-LOO**, which focuses on predictive performance, and **LOO-PIT**, which focuses on calibration.

The conceptual starting point is leave-one-out cross-validation. For each observation $y_i$, we imagine removing it from the dataset, fitting the model on the remaining data $y_{-i}$, and then asking how well the model predicts the held-out point. In principle this is straightforward, but in practice it is computationally expensive because it would require refitting the model $N$ times.

The purpose of **Pareto Smoothed Importance Sampling Leave-One-Out**, or **PSIS-LOO**, is to avoid those repeated refits. Instead of sampling separately from each leave-one-out posterior, we reuse the posterior draws from the model fitted to the full dataset and reweight them so they behave approximately like draws from the corresponding leave-one-out posterior.

This note is written with two goals in mind. First, I want to explain the main ideas at a level that is accessible to students seeing these diagnostics for the first time. Second, I want to connect those ideas directly to a concrete Julia implementation using `Turing.jl` and `PosteriorStats.jl`.

## 1. The general picture

Suppose we observe data

$$
y = (y_1, y_2, \dots, y_N),
$$

and after fitting a Bayesian model we obtain posterior draws

$$
\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(S)} \sim p(\theta \mid y).
$$

These posterior draws tell us which parameter values are plausible after seeing the full dataset. But they do not by themselves answer the question of how well the model predicts unseen data. For that, we need a leave-one-out point of view.

For each observation $y_i$, leave-one-out asks us to consider the posterior

$$
p(\theta \mid y_{-i}),
$$

where $y_{-i}$ denotes all observations except the $i$th one. Using that leave-one-out posterior, we can ask two kinds of questions:

- **Predictive accuracy:** how much probability mass does the model assign to the held-out observation?
- **Calibration:** where does the held-out observation fall inside its leave-one-out predictive distribution?

The first question leads to **ELPD**, the expected log predictive density. The second leads to **LOO-PIT**, the leave-one-out probability integral transform.

## 2. The model and the data

To make the discussion concrete, let us work with a multivariate normal model in `Turing.jl`. This is a useful example because it is not factorizable into independent likelihood contributions: the observations are modeled jointly through a dense covariance structure. That makes it a good setting for understanding the more general leave-one-out logic.

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

At this stage we have only specified a model and generated one synthetic dataset from it. Nothing leave-one-out has happened yet. But we now have the ingredients we need: a model, data, and a joint covariance structure that makes the observations dependent.

## 3. Fitting the model once

The next step is to fit the model to the full dataset. This gives us posterior draws for the parameters, and it also allows us to generate posterior predictive draws.

```julia
# Sample from the model using 8 parallel chains
chns = sample(myfoo(SL, x) | (; y), NUTS(), MCMCThreads(), 2000, 8)

# Generate posterior predictions: stochastic realizations from the likelihood
y_pred = predict(myfoo(SL, x), chns)
```

This distinction is important:

- `chns` contains posterior draws for the parameters;
- `y_pred` contains posterior predictive draws for the data.

Students often confuse these two objects, so it is worth stating this explicitly. The posterior tells us which parameter values are plausible. The posterior predictive distribution tells us what new data the model would generate after accounting for both parameter uncertainty and the intrinsic randomness of the likelihood.

That is why `y_pred` must contain **stochastic realizations**. It is not enough to look only at the conditional means $\mu^{(s)}$. A predictive check must compare the observed data to actual draws from the predictive distribution, not just to its center.

## 4. What we need for leave-one-out

To approximate leave-one-out behavior, we need to know how each posterior draw relates to each observation. In practice, this means computing a **pointwise log-likelihood** object.

In factorized models, one often writes something like

$$
\log p(y_i \mid \theta^{(s)}).
$$

In our non-factorizable multivariate setting, the relevant quantity is instead the conditional log-density

$$
\ell_i^{(s)} = \log p(y_i \mid y_{-i}, \theta^{(s)}).
$$

This is the basic building block for everything that follows: PSIS reweighting, ELPD, and LOO-PIT.

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

At this point, the heavy conceptual lifting has already happened. Once we have `log_like`, we have a quantitative measure of how compatible each observation is with each posterior draw.

## 5. PSIS reweighting instead of refitting

Exact leave-one-out would require sampling from each posterior $p(\theta \mid y_{-i})$ separately. PSIS avoids that by reweighting the draws from the full posterior $p(\theta \mid y)$.

The intuition is simple. If a particular posterior draw makes $y_i$ extremely likely, then that draw may have been strongly influenced by having seen $y_i$ during fitting. If we want to mimic the posterior that would have arisen without $y_i$, such a draw should receive less relative weight.

This is what importance sampling does, and **PSIS** stabilizes those raw weights by smoothing the upper tail.

```julia
# Compute LOO to obtain smoothed importance weights
loo_res = loo(log_like)
log_weights = loo_res.psis_result.log_weights
```

The object `log_weights` is the key output of this step. For each observation, it tells us how to reweight the full-posterior draws so that they approximate the leave-one-out posterior.

PSIS also comes with an internal reliability diagnostic, the **Pareto $k$** value. Large values mean the reweighting is unstable; small values mean the approximation is much more trustworthy.

![](https://github.com/user-attachments/assets/d963f905-1680-4dca-8d79-835f576c9037)

A useful rule of thumb is that values below about $0.7$ are usually acceptable. When the Pareto $k$ values are too large, one should be more cautious and consider alternatives such as exact leave-one-out refits or $K$-fold cross-validation.

## 6. Predictive accuracy: ELPD

Once we have leave-one-out weights, we can quantify predictive performance.

For each held-out observation $y_i$, the leave-one-out predictive density is approximated by a weighted average over posterior draws:

$$
\widehat{p}(y_i \mid y_{-i}) = \sum_{s=1}^S \tilde{w}_i^{(s)} \, p(y_i \mid y_{-i}, \theta^{(s)}).
$$

Taking the log of this quantity gives the pointwise leave-one-out contribution. Summing over all observations gives the total leave-one-out expected log predictive density:

$$
\widehat{\mathrm{ELPD}}_{\mathrm{LOO}}.
$$

This is the quantity used for model comparison and predictive scoring. A larger ELPD means better expected out-of-sample predictive performance.

In our code, we do not manually reconstruct this calculation term by term, because `PosteriorStats.jl` already performs it inside the `loo_res` object. But conceptually it is important to see that the same PSIS weights we just computed are now being used to approximate predictive performance without repeated refits.

## 7. Calibration: the idea behind LOO-PIT

Predictive accuracy is not the whole story. A model can predict reasonably well on average and still misrepresent uncertainty. For example, it may produce predictive distributions that are systematically too narrow or too wide.

This is what **LOO-PIT** is designed to diagnose.

For each observation $y_i$, we compare the actual observed value to its leave-one-out predictive distribution. More precisely, we compute the leave-one-out predictive CDF at the observed point:

$$
\widehat{\mathrm{PIT}}_i = P(\tilde{y}_i \le y_i \mid y_{-i}).
$$

If the model is well calibrated, then across many observations these PIT values should look approximately like draws from a $\mathrm{Uniform}(0,1)$ distribution.

That statement is the central interpretation rule for LOO-PIT:

- values near $0$ mean the observation lies far in the left tail of its predictive distribution;
- values near $1$ mean it lies far in the right tail;
- if the model is calibrated, the collection of these values should not systematically bunch up at one end or in the middle.

There are two ways to compute this CDF in our example.

## 8. The Monte Carlo route

The most general way to compute LOO-PIT is by Monte Carlo.

We already have posterior predictive draws in `y_pred`. For each observation $i$, we check which simulated draws satisfy $\tilde{y}_i \le y_i$, and then sum the corresponding PSIS weights. In other words, we estimate the predictive CDF by a weighted empirical fraction.

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

This is the method students should remember, because it is the most general one. It does not rely on a closed-form CDF. As long as we can generate predictive draws and compute PSIS weights, we can approximate the leave-one-out predictive CDF in this way.

![](https://github.com/user-attachments/assets/b766befa-ece2-4550-98ba-0eb25fba6f6c)

This figure is useful for building intuition. It shows how the observed value is located relative to a leave-one-out predictive distribution for one observation. The LOO-PIT value is precisely the amount of predictive mass lying to the left of that observed point.

## 9. The analytical route

In our Gaussian example, we can also compute the CDF more directly. Since the predictive distribution is normal, we can evaluate the exact Gaussian CDF for each posterior draw and then average those CDF values with the PSIS weights.

That gives a cleaner, less noisy estimate of LOO-PIT.

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

This route is valuable pedagogically because it provides a benchmark. It shows that when the predictive CDF is tractable, we can bypass the extra Monte Carlo noise from drawing $\tilde y_i$ values explicitly.

But it is also important not to overlearn this special case. In many realistic models, there will be no convenient closed-form CDF. In those cases, the Monte Carlo route is the one that survives.

## 10. Comparing the two approaches

Now that we have both versions, we can compare them directly.

```julia
println("Exact Semi-Analytic LOO-PIT (first 5): ", round.(pitvals_exact[1:5], digits=4))
println("Manual Monte Carlo LOO-PIT (first 5): ", round.(pitvals_manual[1:5], digits=4))
```

Output:
```julia
Exact Semi-Analytic LOO-PIT (first 5): [0.6592, 0.0974, 0.5047, 0.5123, 0.0175]
Manual Monte Carlo LOO-PIT (first 5): [0.6559, 0.0996, 0.5048, 0.5155, 0.0184]
```

In this run, the two answers are extremely close. The remaining difference comes from Monte Carlo error, which decreases as the number of predictive draws increases.

![](https://github.com/user-attachments/assets/79fd5161-0623-4f16-8bfc-493c1b92b501)

This comparison matters because it reassures us that the general Monte Carlo method is not just a vague approximation. In a case where the exact answer is available, it tracks it very well.

## 11. Looking at the full collection of PIT values

A single PIT value is not very informative. The real diagnostic appears only after we compute one for every observation and look at their distribution.

If the model is well calibrated, the LOO-PIT values should be approximately uniform on the interval $[0,1]$. Different patterns of deviation from uniformity have different interpretations:

- a **U-shape** suggests under-dispersion or overconfidence;
- a **hump in the center** suggests over-dispersion;
- a **left or right skew** suggests systematic bias.

A convenient way to visualize this is to compare the KDE of the observed LOO-PIT values with KDEs obtained from truly uniform samples of the same size.

```julia
kde_bw = 0.15 # Tuned bandwidth for smoothness on N=30 samples

p = plot(title="LOO-PIT KDE vs uniform reference",
         xlabel="LOO-PIT value", ylabel="Density",
         xlims=(0, 1), ylims=(0, 2.5),
         legend=:topright)

# Helper function to compute bounded KDE over[2]
function get_bounded_kde(data, bw)
    k = PosteriorStats.kde_reflected(data, bounds=(0, 1), bandwidth=bw)
    # Only keep points inside[2]
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

I find this plot especially useful for students because it emphasizes that “uniform” does not mean “perfectly flat” at finite sample size. Even truly uniform samples fluctuate. So the real question is not whether our curve is perfectly flat, but whether it behaves like something plausibly produced by a uniform sample of the same size.

## 12. A numerical complement: the KS test

Visual diagnostics are often the most informative, but it is also natural to ask for a numerical summary of departure from uniformity.

A standard choice is the **Kolmogorov-Smirnov test**. Its null hypothesis is that the LOO-PIT values are drawn from a $\mathrm{Uniform}(0,1)$ distribution.

```julia
using HypothesisTests

# Perform the KS test for the Monte Carlo version
ks_test_mc = ExactOneSampleKSTest(pitvals_manual, Uniform(0, 1))

# Perform the KS test for the Exact Analytic version
ks_test_exact = ExactOneSampleKSTest(pitvals_exact, Uniform(0, 1))

println("MC LOO-PIT KS test p-value:    ", pvalue(ks_test_mc))
println("Exact LOO-PIT KS test p-value: ", pvalue(ks_test_exact))
```

Output:
```julia
MC LOO-PIT KS test p-value:    0.1919
Exact LOO-PIT KS test p-value: 0.1860
```

In this example, both p-values are comfortably above the common threshold of $0.05$, and they are also very close to each other. That is consistent with the picture we already saw visually: there is no strong evidence here against uniformity.

Still, this test should be interpreted with care. A large p-value does not prove perfect calibration; it only means that we have not found strong evidence against it at the current sample size. Conversely, with very large datasets, even very small deviations from uniformity can produce a small p-value. For that reason, I would treat the KS test as a useful complement to the plots rather than as a replacement for them.

## 13. Validation with PosteriorStats.jl

So far I have unpacked the logic manually to make the ideas transparent. But in real work we usually want the robust package implementation.

This is where `PosteriorStats.jl` comes back in. The package provides a direct `loo_pit` function, and we can compare its result to the manual Monte Carlo calculation.

```julia
# Automated package call
pitvals_automated = loo_pit(y, y_pred_array, log_weights)

# Validation: our manual MC implementation matches perfectly!
max_diff = maximum(abs.(pitvals_manual .- pitvals_automated))
println("Max difference (Automated vs Manual MC): ", max_diff)
```

Output:
```julia
Max difference (Automated vs Manual MC): 0.0
```

This exact agreement is reassuring. It tells us that the manual construction we walked through step by step is not merely illustrative: it really reproduces the same quantity that `PosteriorStats.jl` computes automatically.

## 14. Final takeaway

The main lesson of this note is that a single set of posterior draws can be reused in a surprisingly rich way.

Starting from a model fit to the full dataset, we:

1. compute the pointwise conditional log-likelihood;
2. use PSIS to approximate the leave-one-out posterior through importance reweighting;
3. use those same weights to study predictive performance through ELPD;
4. and use them again to study calibration through LOO-PIT.

That is the conceptual thread linking all the pieces together.

In practice, the package implementation is the one you should rely on in production. But for students, I think it is extremely valuable to work through the manual construction at least once. Once you see how posterior draws, predictive draws, pointwise log-likelihoods, and PSIS weights fit together, the high-level diagnostics become much less mysterious.
