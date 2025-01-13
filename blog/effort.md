@def title = "EFfort.jl"

# The Julia EFTofLSS emulator: here comes Effort!

One of the recent hot topics in cosmology is the development of emulators—surrogate models designed to approximate computationally expensive functions with far more efficient alternatives. While many advancements have focused on emulating Cosmic Microwave Background (CMB) calculations, such as ClassNet[^classnet] and Cosmopower[^cosmopower], the spotlight has recently shifted to emulators for Large-Scale Structure (LSS) analyses, after the release of new datasets from the DESI collaboration.

Effort.jl focuses on emulating the galaxy power spectrum in the context of the Effective Field Theory of Large-Scale Structure (EFTofLSS), a cornerstone framework for interpreting LSS surveys. But why yet another emulator? What makes Effort.jl stand out from the many existing tools?

While other frameworks such as EmulateLSS[^emulatelss], Matryoshka[^matryoshka], COMET[^comet], COBRA[^cobra] have demonstrated exceptional performance, Effort.jl aims to take the game a step further, achieving new heights in computational efficiency and flexibility. Here’s how:

⚡ Blazing Performance: Effort.jl can compute the galaxy power spectrum multipoles in approximately 15 µs—orders of magnitude faster than traditional Boltzmann solvers. This is thanks to its integration with the high-performance Julia ecosystem.

🧠 Physics-Based Preprocessing: To reduce training resources, Effort.jl incorporates analytical rescaling of input and output features based on the underlying physics of the problem. This preprocessing minimizes the emulator’s complexity while maintaining precision.

🔋 Effortless Training: Unlike many resource-heavy frameworks, Effort.jl is designed to be trained efficiently using standard hardware, completing training in about one hour on a standard CPU.

🎯 Gradient-Based Optimization: Effort.jl is fully differentiable, enabling seamless integration with gradient-based algorithms for Bayesian inference and parameter minimization. This opens up possibilities for faster and more accurate cosmological analyses.

These characteristics make Effort.jl an appealing choice for cosmologists working on next-generation surveys, such as DESI and Euclid, where precision and computational efficiency are paramount.

Effort.jl is part of a broader vision to revolutionize cosmological emulators by combining speed, accessibility, and adaptability. Whether you’re exploring the intricacies of galaxy clustering or aiming for cutting-edge inference, Effort.jl promises to be a powerful tool in your arsenal.

Furthermore, Effort.jl is written, _ça va sans dire_, in pure Julia. This means that it can be easily deployed on different kind of machines, even on Android smartphones. Yes, I have used Capse.jl on my Android smartphone and it was incredibly performant also there[^girlfriend].

Let us now give a quick summary of the main points of the Capse.jl emulator.


\toc

## Installation & Usage

Installing Capse.jl is quite easy. After moving to the package environment, you can enter the Pkg REPL by pressing `]` from the Julia REPL. To get back to the Julia REPL, press ``Ctrl+C`` or backspace (when the REPL cursor is at the beginning of the input).

Upon entering the Pkg REPL, you should see the following prompt:

```
(@v1.10) pkg>
```
Now you can install Capse.jl
```
(@v1.10) pkg> add https://github.com/CosmologicalEmulators/Effort.jl
```

Now load the following packages

```julia
using Effort
```

Now you have to load the trained emulator. Assuming rhe trained emulator is in a specific folder,

1. You can define a `SimpleChains.jl` emulator

```julia
emu = Effort.load_multipole_emulator("/path/to/folder")
```

Easy, isn't it?

We can now use the loaded emulator to compute some $P_\ell(k)$'s!

```julia
Capse.get_Pk(input_test, emu)
```

That's it! The mean execution time of `Effort.jl` is of $15\,\mu s$!

## Precision tests: residual curves & chains

`Effort.jl` is the fastest EFTofLSS emulator available. But is it precise?

Let us start plotting the distribution of the percentage residuals

![Effort error distribution](https://github.com/user-attachments/assets/8c32f463-51bc-4da2-821c-64319ce52679)

As can be seen, the impact of the preprocessing is more important than that of the training dataset size!

These are not the only tests we performed in order to assess the accuracy of our emulators. We analyzed the PT-challenge and BOSS datasets.

Here comes one of the `Julia` strenghts: the high interoperability of its ecosystem. Given our `Turing.jl` likelihood, we can use different analysis algorithms:
- NUTS, the No-U-Turn-Sampler. This is the state-of-the-art gradient-based sampler
- MicroCanonical Hamiltonian MonteCarlo, MCHMC, a recently developed Hamiltonian MonteCarlo Sampler
- Pathfinder, a variational inference algorithm

The `Julia` implementation of these algorithms is interfaced with `Turing.jl`, so we can easily test these algorithms with our likelihoods!
What is the result of this comparison?

![Contour Planck](https://github.com/marcobonici/marcobonici.github.io/assets/58727599/6ad77fa6-5df7-4edd-b629-35aef48f312b)

The chains are basically the same, with differences of about $0.1\sigma$ for the MCMC methods, and $0.2\sigma$ for Pathfinder. But what about the computational performance?

- pybird analyzed the BOSS dataset with  ~ $1000$ CPUhours, even though it was using analytical marginalization
- `Effort.jl` + NUTS employed 1 hour of wall-time, with an ESS/s of 0.4
- `Effort.jl` + MCHMC required less than 1 hour of wall-time, with an ESS/s of 2.4

## Ok, and now?

Please, feel free to use the comments-tab here or to drop to me, or any of my colleagues, an email for questions or comments.

[^classnet]: [CosmicNet II: Emulating extended cosmologies with efficient and accurate neural networks (2022)](https://arxiv.org/abs/2207.05707)
[^cosmopower]: [COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys](https://arxiv.org/abs/2106.03846)
[^moustache]: The man with the most beautiful pair of moustaches of the East Coast. Here you can find his [website](https://fbianchini.github.io/).
[^phylosophy]: I love discussing with Jaime about no-physics related topics. The problem is that he has a way deeper knowledge of phylosophy than me, and every single time I have to admit (at least to myself) that he is right. Here you can find his [website](https://jaimeruizzapatero.net/).
[^simplechains]: Here you can find a link to [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) repository.
[^preprocess]: Although we already reached a nice performance, we wanna improve the preprocessing in a future work.
[^abstractemu]: we are registering in these days the package `AbstractCosmologicalEmulators.jl`, which is at the core of the CosmologicalEmulators ecosystem.
{{ addcomments }}
