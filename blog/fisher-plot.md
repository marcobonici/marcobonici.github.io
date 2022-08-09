@def title = "FisherPlot.jl"

# FisherPlot.jl: plotting your Fisher posterior contours

[FisherPlot.jl](https://github.com/marcobonici/FisherPlot.jl) is my first registered Julia
package. It can be used to easily plot Fisher matrix contours. In the remainder of this
post, I'll give a (short) introduction to Fisher matrices, before showing how actually use
this package. 




\toc

## Fisher Matrix

You have an experiment that is going to collect some data $\boldsymbol{D}$ and you want to
understand how precisely you are going to measure some parameters $\boldsymbol{\theta}.$
Maybe you don't have collected any data, started your experiment or even obtained the money
to _build_ the experiment. Which error can we expect on the parameters we care about? Is
there any correlation between our parameters? How much precision will you gain by buying a
finer piece of equipment?  Which numbers are you gonna put in that grant proposal?
Performing a Fisher forecast can help you answer these questions.

In order to compute the Fisher matrix, you just need to know the likelihood of your data
given the model parameters, $L(\boldsymbol{D}|\boldsymbol{\theta})$, and compute minus the
Hassian matrix of the log likelihood, $\log L$:[^fisher][^computation]
\begin{equation}
\boldsymbol{F}_{i j} \equiv-\left\langle\frac{\partial^{2} \log L}{\partial \theta_{i}
\partial \theta_{j}}\right\rangle.
\label{eq:fisher}
\end{equation}

Once the Fisher matrix $\boldsymbol{F}_{i j}$ has been computed, the parameters covariance
matrix $\boldsymbol{C}_{ij}$ can be easily obtained inverting the Fisher Matrix

\begin{equation}
\boldsymbol{C} \equiv \boldsymbol{F}^{-1}.
\label{eq:covariance}
\end{equation}

Once you have computed the correlation matrix, the  $1-\sigma$ error on the $i$-th parameter
of the model is given by
\begin{equation}
\sigma_i = \sqrt{C_{ii}}.
\end{equation}

But we have not finished yet: we can also obtain the correlation between the model
parameters. Let us focus on the $2\mathrm{D}$ marginalized posterior of two parameters,
namely $x$ and $y$. The posterior is represented by an ellipses, whose semi-axes are given
by:
\begin{equation}
a^{2}=\frac{\sigma_{x}^{2}+\sigma_{y}^{2}}{2}+\sqrt{\frac{\left(\sigma_{x}^{2}-
\sigma_{y}^{2}\right)^{2}}{4}+\sigma_{x y}^{2}}
\end{equation}
\begin{equation}
b^{2}=\frac{\sigma_{x}^{2}+\sigma_{y}^{2}}{2}-\sqrt{\frac{\left(\sigma_{x}^{2}-
\sigma_{y}^{2}\right)^{2}}{4}+\sigma_{x y}^{2}}
\end{equation}
while the ellipse orientation is given by
\begin{equation}
\tan 2 \theta=\frac{2 \sigma_{x y}}{\sigma_{x}^{2}-\sigma_{y}^{2}}.
\label{eq:atan}
\end{equation}

\warning{
If you are going to write down your own Fisher plotter, beware of the fact that you should
give to the `atan` function not the ratio of the two elements in Equation \eqref{eq:atan},
but the numerator and the denominator separately. If you are not going to do it, it is
possible that the ellipse will be oriented in the wrong way!}


If you want a more detailed introduction to Fisher matrices, please refer to
[Coe (2009)](https://arxiv.org/pdf/0906.4123.pdf) and/or
[Heavens (2009)](http://www.bo.astro.it/~school/school09/Presentations/Bertinoro09_Alan_Heavens_Notes.pdf).

## FisherPlot.jl

In order to make our plots, we need some Correlation matrices. Rather than creating some
mock Correlation matrices, I prefer to use some real-world matrices that I have calculated by
myself[^fun]. In particular, I am going to use some Correlation matrices calculated
with my code [CosmoCentral](https://github.com/marcobonici/CosmoCentral.jl). We are going to
use three correlation matrices, the first considering measurements coming from Weak Lensing,
the second one considering Photometric Galaxy Clustering, and the last one the combination
of the first two probes plus their cross-correlation (usually known as $3\times2\,\text{pt}$).
These correlation matrices are computed following the approach of the Euclid official
forecasts[^euclid].

The parameters described by these matrices are the Dark Energy equation of state
parameters[^chevallier][^linder], $w_0$ and $w_a$, and the sum of the neutrino masses,
$M_\nu$. 


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
Now we need to define a few arrays, containing:
- the name of the model parameters 
- the name of each Correlation matrix
- the color of each Correlation matrix
```julia:define_probes
LaTeXArray = [L"w_0", L"w_a", L"M_\nu\,[\mathrm{eV}]"]
probes = [L"\mathrm{WL}", L"\mathrm{GC}_\mathrm{ph}",
L"\mathrm{WL}\,+\,\mathrm{GC}_\mathrm{ph}\,+\,\mathrm{XC}"]
colors = ["deepskyblue3", "darkorange1", "green",]
```
Now we need to define  a dictionary containing some objects needed by the plotter
\warning{This is something that is likely to change in the near future, with the next
release.}

```julia:define_plotpars
PlotPars = Dict("sidesquare" => 600,
"dimticklabel" => 50,
"parslabelsize" => 80,
"textsize" => 40,
"PPmaxlabelsize" => 60,
"font" => "Dejavu Sans",
"xticklabelrotation" => 0.)
```
We are almost there! We now need to set up the central values for each parameter, the plot
ranges and where we want to put the tick labels. This is something that you should take care
of. For instance, we know that the sum of the neutrino masses $M_\nu$ is positive,
so we want to avoid plotting the posterior for negative masses. In this particular
case, we are going to use the Weak Lensing errors in order to decide the plot ranges.

\suggestion{Plot ranges}{As a rule of thumb, take for the plot limits and ticks,
respectively, 4 and 3 $\sigma$'s and worse constrained Fisher Matrix. Of course you may
prefer something different, but this should give you a good starting point.}


```julia:define_plotranges
central_values = [-1., 0., 0.06]

limits = zeros(3,2)
ticks = zeros(3,2)
for i in 1:3
    limits[i,1] = -4sqrt(C_WL[i,i])+central_values[i]
    limits[i,2] = +4sqrt(C_WL[i,i])+central_values[i]
    ticks[i,1]  = -3sqrt(C_WL[i,i])+central_values[i]
    ticks[i,2]  = +3sqrt(C_WL[i,i])+central_values[i]
end
limits[3,1] = 0
```
Finally, we need to prepare a white ``canvas`` and paint each contour on top of each other.
```julia:plot_fisher
canvas = FisherPlot.preparecanvas(LaTeXArray, limits, ticks, probes, colors, PlotPars::Dict)
FisherPlot.paintcorrmatrix!(canvas, central_values, C_WL, "deepskyblue3")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_GC, "darkorange1")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_3x2_pt, "green")
FisherPlot.save(joinpath(@OUTPUT, "fisher_contour.png"), canvas) # hide
```
Now, we can see the result!
\fig{fisher_contour}

Quite nice, isn't it?


### References & Footnotes
[^fisher]: [Fisher, The logic of inductive science, Journal of the Royal Statistical Society (1935)](https://www.jstor.org/stable/2342435?origin=JSTOR-pdf)
[^computation]: You can actually work out an analytical expression for your Fisher matrix...or use modern tools such as Automatic Differentiation.
[^fun]: I mean: I am a scientist, and scientists are selfish.
[^euclid]: [Euclid Collaboration, VII. Forecast validation for Euclid cosmological probes, Astronomy & Astrophysics (2020)](https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html)
[^chevallier]: [M. Chevallier, D. Polarski, Accelerating Universe with scaling Dark Matter, International Journal of Modern Physics D (2001)](https://www.worldscientific.com/doi/abs/10.1142/S0218271801000822)
[^linder]: [E. V. Linder, Exploring the Expansion History of the Universe, Physical Review Letter (2003)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.90.091301)


{{ addcomments }}
