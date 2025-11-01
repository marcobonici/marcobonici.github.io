@def title = "Effort.jl"

# The Julia EFTofLSS emulator: here comes Effort!

One of the recent hot topics in cosmology is the development of emulators, surrogate models designed to approximate computationally expensive functions with far more efficient alternatives. While many advancements have focused on emulating Cosmic Microwave Background (CMB) calculations, such as ClassNet[^classnet] and Cosmopower[^cosmopower], the spotlight has recently shifted to emulators for Large-Scale Structure (LSS) analyses, after the release of new analyses from the DESI collaboration.

`Effort.jl` focuses on emulating the galaxy power spectrum in the context of the Effective Field Theory of Large-Scale Structure (EFTofLSS), a cornerstone framework for interpreting LSS surveys. But why yet another emulator? What makes `Effort.jl` stand out from the many existing tools?

While other frameworks such as EmulateLSS[^emulatelss], Matryoshka[^matryoshka], COMET[^comet], COBRA[^cobra], and Pybird-JAX[^pybird] have laid the groundwork, `Effort.jl` aims to take the game a step further, with a focus on computational efficiency and flexibility. Here’s how:

⚡ Blazing Performance: `Effort.jl` can compute the galaxy power spectrum multipoles in approximately 15 µs—orders of magnitude faster than traditional Boltzmann solvers. This is thanks to its integration with the high-performance `Julia` ecosystem.

🧠 Physics-Based Preprocessing: To reduce training resources, `Effort.jl` incorporates analytical rescaling of input and output features based on the underlying physics of the problem. This preprocessing minimizes the emulator’s complexity while maintaining precision.

🔋 Effortless[^pun] Training: Unlike many resource-heavy frameworks, `Effort.jl` is designed to be trained efficiently using standard hardware, completing training in about one hour on a standard CPU.

🎯 Gradient-Based Optimization: `Effort.jl` is fully differentiable, enabling seamless integration with gradient-based algorithms for Bayesian inference and parameter minimization. This opens up possibilities for faster and more accurate cosmological analyses.

These characteristics make `Effort.jl` an appealing choice for cosmologists working on next-generation surveys, such as DESI and Euclid, where precision and computational efficiency are paramount; for this reason, `Effort.jl` has already been applied in a few works[^zhangone][^paradiso][^baleato][^zhangtwo][^morawetz], when where considered scenarios when standard pipelines whould have struggled.

`Effort.jl` is part of a broader vision to improve cosmological emulators by combining speed, accessibility, and adaptability. Whether you’re exploring the intricacies of galaxy clustering or aiming for cutting-edge inference, `Effort.jl` promises to be a powerful tool in your arsenal.

Furthermore, `Effort.jl` is written, _ça va sans dire_, in pure `Julia`. This means that it can be easily deployed on different kind of machines, even on Android smartphones. Yes, I have used `Effort.jl` on my Android smartphone and it was incredibly performant also there[^girlfriend].

Let us now give a quick summary of the main selling points of the `Effort.jl` emulator.


\toc

## Installation & Usage

Installing `Effort.jl` is quite easy. After moving to the package environment, you can enter the Pkg REPL by pressing `]` from the Julia REPL. To get back to the Julia REPL, press ``Ctrl+C`` or backspace (when the REPL cursor is at the beginning of the input).

Upon entering the Pkg REPL, you should see the following prompt:

```
(@v1.11) pkg>
```
Now you can install `Effort.jl`
```
(@v1.11) pkg> add Effort
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

We can now use the loaded emulator to compute some $P_\ell(k)$'s! In order to do that you need to pass the input cosmological parameters as a vector `input_test`, the growth factor `D` (more on this later), and the biases `biases`.

```julia
Effort.get_Pℓ(input_test, D, biases, emu)
```

That's it! The mean execution time of `Effort.jl` is of $15\,\mu s$!

## Precision tests: residual curves & chains

`Effort.jl` is the fastest EFTofLSS emulator available. But is it precise?

Let us start plotting the distribution of the percentage residuals. We are doing this for 4 different scenarios, in order to gauge the impact of the preprocessing and the training dataset size. Specifically, the 11 and counterterms contributions are linear in $P_\mathrm{lin}$ while the loop term is quadratic in $P_\mathrm{lin}$ (up to IR contributions).
We the decided to train the emulators by rescaling the output of `pybird` by a factor that accounts for the linear rescaling of the $P_\mathrm{lin}$. We considered four different approaches:

- No rescaling at all, thestandard baseline.
- Rescaling by $A_\mathrm{s}$. This takes into account the initial scalar fluctuations amplitude.
- Rescaling by $D^2(z)$. This takes into account the redshift evolution as a funciton of cosmological parameters.
- Rescaling by $A_\mathrm{s} D^2(z)$. This takes into account both previous effects.

We will gauge the impact of the size of the training dataset, using two datasets, with respectively 20,000 (dashed lines) and 60,000 (solid lines) elements.

Here are the results!

![Effort error distribution](https://github.com/user-attachments/assets/e0a6ecb5-8a02-4609-8ff6-3cac7cf648ff)

As can be seen, the impact of the preprocessing is more important than that of the training dataset size! This means that yes, having a larger dataset can help you getting a better emulator, but leveraging our domain knowledge as scientists is even better!

These are not the only tests we performed in order to assess the accuracy of our emulators. We analyzed the PT-challenge and BOSS datasets.

Here comes one of the `Julia` strenghts: the high interoperability of its ecosystem. Given our `Turing.jl` likelihood, we can use different analysis algorithms:
- NUTS, the No-U-Turn-Sampler. This is the state-of-the-art gradient-based sampler
- MicroCanonical Hamiltonian MonteCarlo, MCHMC, a recently developed Hamiltonian MonteCarlo Sampler
- Pathfinder, a variational inference algorithm, here used to initialize the chains of the previous methods

The `Julia` implementation of these algorithms is interfaced with `Turing.jl`, so we can easily test these algorithms with our likelihoods!
What is the result of this comparison?

![contour_comparison_Effort_BOSS](https://github.com/user-attachments/assets/f7df018e-bd37-4e7b-99bd-bbd3e9e86584)
The chains are basically the same, with differences of about $0.1\sigma$. But what about the computational performance?

- `pybird` analyzed the BOSS dataset with  ~ $1000$ CPUhours, even though it was using analytical marginalization
- `Effort.jl` + NUTS employed 1 hour of wall-time, with an ESS/s of 0.4
- `Effort.jl` + MCHMC required less than 1 hour of wall-time, with an ESS/s of 2.4

## Ok, and now?

Although the `Effort.jl` release paper has just been accepted on JCAP[^paper] (and there was even some media coverage about it[^press]!) there are a few things we are working on about:

- A `jax` compatible version, `jaxeffort`. This is mostly done, and it is available on [GitHub](https://github.com/CosmologicalEmulators/jaxeffort).
- Batteries included. I am including by default some emulators into the packages, such that they can be used out-of-the-box.
- A restructuring. In order to ease certain developments, I am reorganizing a bit the structure of the code (and most likely found a design that satisfies me).

Finally, the other important point: applications! A piece of code is as good as its applications, so here is what we have been doing:

- HOD-Informed Priors (HIP). With a colleague of mine, Hanyu Zhang, we have been working on a way to enhance the EFTofLSS analyses by placing tighter priors on the EFT parameters. This boils down to get some simulations, populate them with an HOD approach, measure the $P_\ell$ and get some bestfits. In these papers of ours[^zhangone, zhangtwo], this has been done for hundreds of thousands of simulations. This is where the speed and differentiability of the code really shine: the L-BFGS minimization algorithm was employed in these analyses and it proved extremely powerful.
- Orthogonalization approach to Projection effects. With another colleague of mine, Simone Paradiso, we have been working on a way to reduce the Projection effects by reparameterizing the EFT parameters. In such a manner we can reduce the projection effects that are plaguing this kind of analysis.
- Frequentist analysis. As for the HIP approach, this boils down to computing a lot of bestfits. Also in this case, the differentiability of the code helps like a lot: L-BFGS is a beast of an algorithm, and it can easily handle the large number parameters to adjust during the minimization. A paper on this topic[^morawetz] by my student, James Morawetz, has just been submitted to JCAP.

Please, feel free to use the comments-tab here or to drop to me, or any of my colleagues, an email for questions or comments.

[^classnet]: [CosmicNet II: Emulating extended cosmologies with efficient and accurate neural networks (2022)](https://arxiv.org/abs/2207.05707)
[^cosmopower]: [COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys](https://arxiv.org/abs/2106.03846)
[^emulatelss]: [Neural network acceleration of large-scale structure theory calculations](https://arxiv.org/abs/2112.05889)
[^matryoshka]: [matryoshka II: accelerating effective field  theory analyses of the galaxy power spectrum](https://arxiv.org/abs/2202.07557)
[^comet]: [Clustering observables modelled by emulated perturbation theory](https://arxiv.org/abs/2208.01070)
[^cobra]: [COBRA: Optimal Factorization of Cosmological Observables](https://arxiv.org/abs/2407.04660)
[^pybirdjax]: [PyBird-JAX: Accelerated inference in large-scale structure with model-independent emulation of one-loop galaxy power spectra](https://arxiv.org/abs/2507.20990)
[^zhangone]: [HOD-informed prior for EFT-based full-shape analyses of LSS](https://arxiv.org/abs/2409.12937)
[^paradiso]: [Reducing nuisance prior sensitivity via non-linear reparameterization, with application to EFT analyses of large-scale structure](https://arxiv.org/abs/2412.03503)
[^baleato]: [Selecting samples of galaxies with fewer Fingers-of-God](https://arxiv.org/abs/2501.10587)
[^zhangtwo]: [Enhancing DESI DR1 Full-Shape analyses using HOD-informed priors](https://arxiv.org/abs/2504.10407)
[^morawetz]: [Frequentist Cosmological Constraints from Full-Shape Clustering Measurements in DESI DR1](https://arxiv.org/abs/2508.11811)
[^pun]: Pun _totally_ intended.
[^girlfriend]: When I told this to my girlfriend her honest reaction was "Are you trying to get single again?".
[^paper]: [Effort.jl: a fast and differentiable emulator for the Effective Field Theory of the Large Scale Structure of the Universe](https://iopscience.iop.org/article/10.1088/1475-7516/2025/09/044)
[^press]: [Cosmic simulations that once needed supercomputers now run on a laptop](https://www.sciencedaily.com/releases/2025/09/250918225001.html). And someone even posted that on [Hacker News](https://news.ycombinator.com/item?id=45346538)!
{{ addcomments }}
