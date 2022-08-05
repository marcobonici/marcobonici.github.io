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
Once the Fisher matrix $\boldsymbol{F}_{i j}$ has been computed, the parameters covariance matrix $\boldsymbol{C}_{ij}$ can be easily obtained inverting the Fisher Matrix

\begin{equation}
\boldsymbol{C} \equiv \boldsymbol{F}^{-1}.
\label{eq:covariance}
\end{equation}

The $1-\sigma$ error on the $i$-th parameter of the model is given by
\begin{equation}
\sigma_i = \sqrt{C_{ii}}
\end{equation}
## FisherPlot.jl
Let us consider we have the following Fisher matrix
```julia:import_packages
using FisherPlot
using LaTeXStrings

Fisher_matrix = [0.00375 -0.00125; -0.00125 0.00375]
Correlation_matrix = inv(Fisher_matrix)
LaTeXArray = [L"\Omega_\mathrm{M}", L"\Omega_\mathrm{B}"]
central_values =[0.0, 0.0]
probes = [L"\mathrm{WL}"]
colors = ["deepskyblue3"]

PlotPars = Dict("sidesquare" => 400,
"dimticklabel" => 50,
"parslabelsize" => 80,
"textsize" => 80,
"PPmaxlabelsize" => 60,
"font" => "Arial",
"xticklabelrotation" => 45.)

σa = sqrt(Correlation_matrix[1,1])
σb = sqrt(Correlation_matrix[2,2])

limits = [-4σa 4σa; -4σb 4σb]
ticks = [-3σa 3σa; -3σb 3σb]

canvas = FisherPlot.preparecanvas(LaTeXArray, limits, ticks, probes, colors, PlotPars::Dict)
FisherPlot.paintcorrmatrix!(canvas, central_values, Correlation_matrix, "deepskyblue3")
FisherPlot.save(joinpath(@OUTPUT, "fisher_contour.png"), canvas) # hide
```
\fig{fisher_contour}


### References
[^fisher]: [Fisher, The logic of inductive science, Journal of the Royal Statistical Society, (1935)](https://www.jstor.org/stable/2342435?origin=JSTOR-pdf)

{{ addcomments }}
