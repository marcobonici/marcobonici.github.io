@def title = "Marco Bonici"

I am a Postdoctoral Researcher at the [Waterloo Centre for Astrophysics](https://uwaterloo.ca/astrophysics-centre/), University of Waterloo.
I am a Cosmologist working at the intersection of theory, computation, and statistics to understand the large-scale structure of the Universe and to extract maximal information from current and upcoming cosmological surveys.

My research lines are tightly interconnected: in order to perform extensive cross-validation and field-level analyses of cosmological datasets, *fast, differentiable codes* are an absolute necessity. This drives the development of the methods I work on:

- **Accelerating theoretical predictions**. I develop and apply numerical techniques to speed up calculations of cosmological observables — including surrogate models and novel algorithms — to reduce computation time without sacrificing scientific accuracy.

- **Enabling gradient-based inference**. I specialize in automatic differentiation methods that power state-of-the-art gradient-based algorithms, both for minimization (L-BFGS) and sampling (Hamiltonian Monte Carlo, Microcanonical Langevin Monte Carlo), enabling scalable and efficient analysis.

- **Maximizing precision and accuracy in inference**. I work on extracting the most possible information from our data through field-level inference techniques, and I rigorously validate results using approaches such as Leave-One-Out and cross-validation to avoid bias and overfitting.

Much of this work is implemented in the [Julia](https://docs.julialang.org/en/v1/) programming language, leveraging its performance, composability, and differentiability to unify modeling, optimization, and validation in a single, coherent framework.


## Selected posts from my blog

* [The Julia EFTofLSS emulator: here comes Effort!](/blog/effort)
* [The Julia CMB emulator: here comes Capse!](/blog/capse)
* [Fisher matrix posterior contour plot with FisherPlot.jl.](/blog/fisher-plot)
