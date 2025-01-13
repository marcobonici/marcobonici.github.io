@def title = "Effort.jl"

# The Julia EFTofLSS emulator: here comes Effort!

One of the recent hot topics in cosmology is the development of emulators—surrogate models designed to approximate computationally expensive functions with far more efficient alternatives. While many advancements have focused on emulating Cosmic Microwave Background (CMB) calculations, such as ClassNet[^classnet] and Cosmopower[^cosmopower], the spotlight has recently shifted to emulators for Large-Scale Structure (LSS) analyses, after the release of new datasets from the DESI collaboration.

`Effort.jl` focuses on emulating the galaxy power spectrum in the context of the Effective Field Theory of Large-Scale Structure (EFTofLSS), a cornerstone framework for interpreting LSS surveys. But why yet another emulator? What makes `Effort.jl` stand out from the many existing tools?

While other frameworks such as EmulateLSS[^emulatelss], Matryoshka[^matryoshka], COMET[^comet], and COBRA[^cobra] have laid the groundwork, `Effort.jl` aims to take the game a step further, with a focus on computational efficiency and flexibility. Here’s how:

⚡ Blazing Performance: `Effort.jl` can compute the galaxy power spectrum multipoles in approximately 15 µs—orders of magnitude faster than traditional Boltzmann solvers. This is thanks to its integration with the high-performance `Julia` ecosystem.

🧠 Physics-Based Preprocessing: To reduce training resources, `Effort.jl` incorporates analytical rescaling of input and output features based on the underlying physics of the problem. This preprocessing minimizes the emulator’s complexity while maintaining precision.

🔋 Effortless Training: Unlike many resource-heavy frameworks, `Effort.jl` is designed to be trained efficiently using standard hardware, completing training in about one hour on a standard CPU.

🎯 Gradient-Based Optimization: `Effort.jl` is fully differentiable, enabling seamless integration with gradient-based algorithms for Bayesian inference and parameter minimization. This opens up possibilities for faster and more accurate cosmological analyses.

These characteristics make `Effort.jl` an appealing choice for cosmologists working on next-generation surveys, such as DESI and Euclid, where precision and computational efficiency are paramount; for this reason, `Effort.jl` has already been applied in Zhang et al. and Paradiso et al., when where considered scenarios when standard pipelines whould have struggled.

`Effort.jl` is part of a broader vision to revolutionize cosmological emulators by combining speed, accessibility, and adaptability. Whether you’re exploring the intricacies of galaxy clustering or aiming for cutting-edge inference, `Effort.jl` promises to be a powerful tool in your arsenal.

Furthermore, `Effort.jl` is written, _ça va sans dire_, in pure Julia. This means that it can be easily deployed on different kind of machines, even on Android smartphones. Yes, I have used `Effort.jl` on my Android smartphone and it was incredibly performant also there[^girlfriend].

Let us now give a quick summary of the main points of the `Effort.jl` emulator.


\toc

## Installation & Usage

Installing `Effort.jl` is quite easy. After moving to the package environment, you can enter the Pkg REPL by pressing `]` from the Julia REPL. To get back to the Julia REPL, press ``Ctrl+C`` or backspace (when the REPL cursor is at the beginning of the input).

Upon entering the Pkg REPL, you should see the following prompt:

```
(@v1.10) pkg>
```
Now you can install `Effort.jl`
```
(@v1.10) pkg> add https://github.com/CosmologicalEmulators/Effort.jl
```

Now load the following packages

```julia
using Effort
```

Now you have to load the trained emulator. Assuming the trained emulator is in a specific folder,


```julia
emu = Effort.load_multipole_emulator("/path/to/folder")
```

Easy, isn't it?

We can now use the loaded emulator to compute some $P_\ell(k)$'s!

```julia
Effort.get_Pk(input_test, emu)
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
- Pathfinder, a variational inference algorithm, here used to initialize the chains of the previous methods

The `Julia` implementation of these algorithms is interfaced with `Turing.jl`, so we can easily test these algorithms with our likelihoods!
What is the result of this comparison?

![contour_comparison_Effort_BOSS](https://github.com/user-attachments/assets/f7df018e-bd37-4e7b-99bd-bbd3e9e86584)
The chains are basically the same, with differences of about $0.1\sigma$. But what about the computational performance?

- pybird analyzed the BOSS dataset with  ~ $1000$ CPUhours, even though it was using analytical marginalization
- `Effort.jl` + NUTS employed 1 hour of wall-time, with an ESS/s of 0.4
- `Effort.jl` + MCHMC required less than 1 hour of wall-time, with an ESS/s of 2.4

## Ok, and now?

Please, feel free to use the comments-tab here or to drop to me, or any of my colleagues, an email for questions or comments.

[^classnet]: [CosmicNet II: Emulating extended cosmologies with efficient and accurate neural networks (2022)](https://arxiv.org/abs/2207.05707)
[^cosmopower]: [COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys](https://arxiv.org/abs/2106.03846)
[^emulatelss]: [Neural network acceleration of large-scale structure theory calculations](https://arxiv.org/abs/2112.05889)
[^matryoshka]: [matryoshka II: accelerating effective field  theory analyses of the galaxy power spectrum](https://arxiv.org/abs/2202.07557)
[^comet]: [Clustering observables modelled by emulated perturbation theory](https://arxiv.org/abs/2208.01070)
[^cobra]: [COBRA: Optimal Factorization of Cosmological Observables](https://arxiv.org/abs/2407.04660)
[^girlfriend]: When I told this to my girlfriend her honest reaction was "Are trying to get single again?".
{{ addcomments }}
