@def title = "FisherPlot.jl"

# FisherPlot.jl: plotting your Fisher posterior contours

[FisherPlot.jl](https://github.com/marcobonici/FisherPlot.jl) is my first registered Julia
package. It can be used to easily plot Fisher matrix contours. In the remainder of this
post, I'll give a (short) introduction to Fisher matrices, before showing how actually use
this package. 




\toc

## Fisher Matrix
The Fisher Matrix is a statistical tool that can be used to make _forecasts_, i.e. to
predict the sensitivity of an experiment to a set of parameters. This can be done before
having the actual data, in order to make _experiment design_.

In order to compute the Fisher matrix, we just need to know our model and the measurement
uncertainties. Put in another way, we just need to write down a log likelihood $\log L$ and
compute its Hessian matrix:[^fisher]
\begin{equation}
\boldsymbol{F}_{i j} \equiv-\left\langle\frac{\partial^{2} \log L}{\partial \theta_{i} \partial \theta_{j}}\right\rangle.
\label{eq:fisher}
\end{equation}

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
In order to make our plots, we need some Fisher matrices. Here I am gonna define some
correlation matrices computed using my code
[CosmoCentral](https://github.com/marcobonici/CosmoCentral.jl). The parameters described by
these matrices are the Dark Energy equation of state parameters[^chevallier][^linder], $w_0$
and $w_a$, and the sum of the neutrino masses, $M_\nu$. These correlation matrices are
computed following the approach of the Euclid official forecasts[^euclid]


```julia:correlation_matrices
C_WL = [0.0193749   -0.0620328  -0.00296159; -0.0620328    0.225214    0.0119904;
       -0.00296159   0.0119904   0.043537]
C_GC = [0.0150589   -0.0561336    0.00109151; -0.0561336    0.215808    -0.00562308;
        0.00109151  -0.00562308   0.00219601]
C_3x2_pt = [0.000724956  -0.00241439   0.000119474; -0.00241439   0.00844342   -0.000546318;
            0.000119474  -0.000546318   0.00113283]
```
Now we have to import a couple of packages, FisherPlot and LaTeXStrings
```julia:import_packages
using FisherPlot
using LaTeXStrings
```
Now we need to define a few objects:
- an array containing the name of the parameters 
- an array containing the name of each Correlation matrix
- an array containing the color of each Correlation matrix
```julia:define_legends
LaTeXArray = [L"w_0", L"w_a", L"M_\nu"]
probes = [L"\mathrm{WL}", L"\mathrm{GC}",
          L"\mathrm{WL}\,+\,\mathrm{GC}_\mathrm{ph}\,+\,\mathrm{XC}"]
colors = ["deepskyblue3", "darkorange1", "green",]
```
Now we need to define some quantities related to the plot
> This is something that is likely to change in the future.
```julia:define_poltpars
PlotPars = Dict("sidesquare" => 600,
"dimticklabel" => 50,
"parslabelsize" => 80,
"textsize" => 40,
"PPmaxlabelsize" => 60,
"font" => "Dejavu Sans",
"xticklabelrotation" => 45.)

```
We are almost there! We now need to set up the central values for our parameters, the plot
ranges and where we want to put the ticks


```julia:define_limits
central_values = [-1., 0., 0.06]

limits = zeros(3,2)
ticks = zeros(3,2)
for i in 1:3
    limits[i,1] = -4sqrt(C_WL[i,i])+central_values[i]
    limits[i,2] = +4sqrt(C_WL[i,i])+central_values[i]
    ticks[i,1]  = -3sqrt(C_WL[i,i])+central_values[i]
    ticks[i,2]  = +3sqrt(C_WL[i,i])+central_values[i]
end
```
Finally, we need to prepare a white canvas and paint each contour.
```julia:plot_fisher
canvas = FisherPlot.preparecanvas(LaTeXArray, limits, ticks, probes, colors, PlotPars::Dict)
FisherPlot.paintcorrmatrix!(canvas, central_values, C_WL, "deepskyblue3")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_GC, "darkorange1")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_3x2_pt, "green")
FisherPlot.save(joinpath(@OUTPUT, "fisher_contour.png"), canvas) # hide
```
Now, we can see the result!
\fig{fisher_contour}


### References
[^fisher]: [Fisher, The logic of inductive science, Journal of the Royal Statistical Society (1935)](https://www.jstor.org/stable/2342435?origin=JSTOR-pdf)
[^chevallier]: [M. Chevallier, D. Polarski, Accelerating Universe with scaling Dark Matter, International Journal of Modern Physics D (2001)](https://www.worldscientific.com/doi/abs/10.1142/S0218271801000822)
[^linder]: [E. V. Linder, Exploring the Expansion History of the Universe, Physical Review Letter (2003)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.90.091301)
[^euclid]: [Euclid Collaboration, VII. Forecast validation for Euclid cosmological probes, Astronomy & Astrophysics (2020)](https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html)

{{ addcomments }}
