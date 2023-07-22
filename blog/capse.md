@def title = "Capse.jl"

# Capse.jl: a one-second-CMB analysis tool

On of the recent hot topics in Cosmology is represented by the development of _emulators_, surrogate models that are meant to replace a computationally expensive function with an approximation that is way cheaper. Focusing on CMB emulators, two of the last emulators built are ClassNet[^classnet] and Cosmopower[^cosmopower]. Following different strategies and approaches, these two frameworks are able to compute the CMB angular spectrum in a fraction of the time required by an Einstein-Boltzmann solver; Cosmopower in particular is around 1,000 times faster, computing the CMB power spectrum in a few milliseconds.

So, the interested reader might ask us a simple question: *why are you working on yet another CMB emulator? Can't you just use Cosmopower?*

I think that, although Cosmopower has reached an incredibile performance and versatility, we can do better, reducing the amount of resources required to train the emulators and reaching a higher computational performance. Furthermore, I wanna leverage one of the emulators main characteristics, their _differentiability_. This open up to the possibility of using gradient-based algorithms for bayesian inference and minimization.

In order to reach that goal, together with Federico Bianchini[^moustache] and Jaime Ruiz-Zapatero[^phylosophy], I have developed Capse.jl, a CMB Angular Power Spectrum emulator written using the Julia language.

So, what are the main characteristics of Capse.jl and why should you consider giving it a try?

- ⚡ Capse.jl is blazing fast. It builds on SimpleChains.jl[^simplechains], a Julia Neural Network library designed with a limited scope: develop fast neural networks running on the CPU.
- ⚗️ In order to emulate only the essential components, Capse.jl performs a physics-based preprocess of the output features[^preprocess].
- 🔋 Capse.jl is cheap to train: as a consequence of the previous two points, it can be trained in around one hour using a standard CPU.
- 🎯 Capse.jl can be coupled with gradient-based algorithms, in order to perform bayesian analysis and/or minimizations.

Furthermore, Capse.jl is written, _ça va sans dire_, in pure Julia. This means that it can be easily deployed on different kind of machines (installing JAX and TensorFlow on Mac devices can be troublesome), even on Android smartphones. Yes, I have used Capse.jl on my Android smartphone and it was incredibly performant also there.

Let us now give a quick summary of the main points of the Capse.jl emulator.


\toc

## Installation & Usage

Installing Capse.jl is quite easy. After moving to the package environment, you have to type

`Pkg` comes with a REPL. Enter the Pkg REPL by pressing `]` from the Julia REPL. To get back to the Julia REPL, press `Ctrl+C` or backspace (when the REPL cursor is at the beginning of the input).

Upon entering the Pkg REPL, you should see the following prompt:

```
(@v1.9) pkg>
```
Now you can install AbstractEmulator.jl, the higher level package I use to develop my emulators, and Capse.jl
```
(@v1.9) pkg> add https://github.com/CosmologicalEmulators/AbstractEmulator.jl
(@v1.9) pkg> add https://github.com/CosmologicalEmulators/Capse.jl
```

This is something that should work with no particol effort. After installation, can move back to the standard `Julia` REPL, load the following packages
```julia
using SimpleChains
using Static
using NPZ
using Capse
```
Now you we have to load the emulator, which requires three elements:
- the $\ell$-grid the emulator has been trained on
- the trained weights
- the input and output features minimum and maximum ranges, used the perform the min-max normalization

```julia
ℓ = npzread("l.npy")

weights_TT = npzread("weights_TT_lcdm.npy")
trained_emu_TT = Capse.SimpleChainsEmulator(Architecture= mlpd, Weights = weights_TT)
CℓTT_emu = Capse.CℓEmulator(TrainedEmulator = trained_emu_TT, ℓgrid = ℓ,
                             InMinMax = npzread("inMinMax_lcdm.npy"),
                             OutMinMax = npzread("outMinMaxCℓTT_lcdm.npy"))
```
\warning{Up to now `Capse.jl` takes the input with an hardcoded order, being them $\ln 10^{10}A_s$, $n_s$, $H_0$, $\omega_b$, $\omega_c$ and $\tau$. In a future release we are going to add a more flexible way to use it!}

```julia
@benchmark Capse.get_Cℓ($input_test, $CℓTT_emu)
```
That's it! The mean execution time of `Capse.jl` is of $45\,\mu s$! This is almost 3 orders of magnitudes lower than `Cosmopower`!

## Precision tests: residual curves & chains

`Capse.jl` is the fastest CMB emulator available. But is it precise? This is something we need to check, because as Mr. Pirelli said...
![Pirelli](https://github.com/marcobonici/marcobonici.github.io/assets/58727599/da73ded5-0ad9-4855-aa06-e8696d96e6d1)

Let us start plotting the distribution of the ratio between the emulation error and the expected variance from the forthcoming S4-observatory, for a validation dataset with $20,000$ cosmologies covering the entire training space.

![Capse error distribution](https://github.com/marcobonici/marcobonici.github.io/assets/58727599/0dff8c09-604c-4713-9c26-68ee68904531)

This means that the emulation error is $10$ times smaller than expected measurements variance for almost all scales[^scales] for $99\%$ of the validation points!

These are not the only tests we performed in order to assess the accuracy of our emulators. We analyzed the Planck and ACT DR4 datasets, using CAMB and Cobaya, then we used `Turing.jl` to write a likelihood using `Capse.jl`. Although this can be different from what is usually done in Cosmology, the following code snippet should highlight how to write a likelihood

```julia
@model function CMB_planck(data, covariance)
    #prior on model parameters
    ln10As ~ Uniform(0.25, 0.35)
    ns     ~ Uniform(0.88, 1.06)
    h0     ~ Uniform(0.60, 0.80)
    ωb     ~ Uniform(0.1985, 0.25)
    ωc     ~ Uniform(0.08, 0.20)
    τ      ~ Normal(0.0506, 0.0086)
    yₚ     ~ Normal(1.0, 0.0025)

    θ = [10*ln10As, ns, 100*h0, ωb/10, ωc, τ]
    #compute theoretical prediction
    pred = theory_planck(θ) ./(yₚ^2)
    #compute likelihood
    data ~ MvNormal(pred, covariance)

    return nothing
end
```
Here comes one of the `Julia` strenghts: the high interoperability of its ecosystem. Given our `Turing.jl` likelihood, we can use different analysis algorithms:
- NUTS, the No-U-Turn-Sampler. This is the state-of-the-art gradient-based sampler
- MicroCanonical Hamiltonian MonteCarlo, MCHMC, a recently developed Hamiltonian MonteCarlo Sampler
- Pathfinder, a variational inference algorithm

The `Julia` implementation of these algorithms is interfaced with `Turing.jl`, so we can easily test these algorithms with our likelihoods!
What is the result of this comparison?


![Contour Planck](https://github.com/marcobonici/marcobonici.github.io/assets/58727599/6ad77fa6-5df7-4edd-b629-35aef48f312b)

The chains are basically the same, with differences of about $0.1\sigma$ for the MCMC methods, and $0.2\sigma$ for Pathfinder. But what about the computational performance?

- CAMB analyzed Planck with $80$ CPUhours, with an Effective Sample Size per second (ESS/s) of $0.004$
- `Capse.jl` + NUTS employed 1 CPUhour, with an ESS/s of 1.6
- `Capse.jl` + MCHMC required less than 1 CPUhour, with an ESS/s of 3.7
- Pathfinder computes the posterior in ~ $15$ seconds

Although these results are impressive, there is still room for improvements...here comes the Chebyshev-based emulator!


[^classnet]: [CosmicNet II: Emulating extended cosmologies with efficient and accurate neural networks (2022)](https://arxiv.org/abs/2207.05707)
[^cosmopower]: [COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys](https://arxiv.org/abs/2106.03846)
[^moustache]: The man with the most beautiful pair of moustaches of the East Coast. Here you can find his [website](https://fbianchini.github.io/).
[^phylosophy]: I love discussing with Jaime about no-physics related topics. The problem is that he has a way deeper knowledge of phylosophy than me, and every single time I have to admit (at least to myself) that he is right. Here you can find his [website](https://jaimeruizzapatero.net/).
[^simplechains]: Here you can find a link to [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) repository.
[^preprocess]: Although we already reached a nice performance, we wanna improve the preprocessing in a future work.
[^scales]: The only exception is the $EE$ 2-pt correlation function for $\ell<10$. We are working to improve the precision of `Capse.jl` also on these scales.
