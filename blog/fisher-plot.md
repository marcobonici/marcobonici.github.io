@def title = "FisherPlot.jl"

# FisherPlot.jl: plotting your Fisher posterior contours

[FisherPlot.jl](https://github.com/marcobonici/FisherPlot.jl) is my first registered Julia package. It can be used to easily plot Fisher matrix contours. In the remainder of this post, I'll give a (short) introduction to Fisher matrices, before showing how actually use this package. Cool, isn't it?




\toc

## Fisher Matrix
The Fisher Matrix is a tool that can be used to make _forecasts_, i.e. to predict the sensitivity of an experiment to a set of parameters. Given the likelihood, the expression of the Fisher matrix is given by[^fisher]
\begin{equation}
\boldsymbol{F}_{i j} \equiv-\left\langle\frac{\partial^{2} \log L}{\partial \theta_{i} \partial \theta_{j}}\right\rangle;
\label{eq:fisher}
\end{equation}
_i.e._ the Fisher matrix is given by (minus) the Hessian of the log likelihood$\log L$.

## FisherPlot.jl


### References
[^fisher]: [Fisher, The logic of inductive science, Journal of the Royal Statistical Society, (1935)](https://www.jstor.org/stable/2342435?origin=JSTOR-pdf)

{{ addcomments }}
