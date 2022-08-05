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
LaTeXArray = [L"w_0", L"w_a", L"M_\nu"]
central_values =[0.0, 0.0]
probes = [L"\mathrm{WL}", L"\mathrm{GC}",
          L"\mathrm{WL}\,+\,\mathrm{GC}_\mathrm{ph}\,+\,\mathrm{XC}"]
colors = ["deepskyblue3", "darkorange1", "green",]

PlotPars = Dict("sidesquare" => 600,
"dimticklabel" => 50,
"parslabelsize" => 80,
"textsize" => 40,
"PPmaxlabelsize" => 60,
"font" => "Dejavu Sans",
"xticklabelrotation" => 45.)


C_WL = [0.0193749   -0.0620328  -0.00296159; -0.0620328    0.225214    0.0119904;
       -0.00296159   0.0119904   0.043537]
C_GC = [0.0150589   -0.0561336    0.00109151; -0.0561336    0.215808    -0.00562308;
        0.00109151  -0.00562308   0.00219601]
C_3x2_pt = [0.000724956  -0.00241439   0.000119474; -0.00241439   0.00844342   -0.000546318;
            0.000119474  -0.000546318   0.00113283]


σa = sqrt(Correlation_matrix[1,1])
σb = sqrt(Correlation_matrix[2,2])
central_values = [-1., 0., 0.06]

limits = zeros(3,2)
ticks = zeros(3,2)
for i in 1:3
    limits[i,1] = -4sqrt(C_WL[i,i])+central_values[i]
    limits[i,2] = +4sqrt(C_WL[i,i])+central_values[i]
    ticks[i,1]  = -3sqrt(C_WL[i,i])+central_values[i]
    ticks[i,2]  = +3sqrt(C_WL[i,i])+central_values[i]
end

canvas = FisherPlot.preparecanvas(LaTeXArray, limits, ticks, probes, colors, PlotPars::Dict)
FisherPlot.paintcorrmatrix!(canvas, central_values, C_WL, "deepskyblue3")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_GC, "darkorange1")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_3x2_pt, "green")
FisherPlot.save(joinpath(@OUTPUT, "fisher_contour.png"), canvas) # hide
```
\fig{fisher_contour}


### References
[^fisher]: [Fisher, The logic of inductive science, Journal of the Royal Statistical Society, (1935)](https://www.jstor.org/stable/2342435?origin=JSTOR-pdf)

{{ addcomments }}
