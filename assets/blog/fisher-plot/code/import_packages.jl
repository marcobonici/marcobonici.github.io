# This file was generated, do not modify it. # hide
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
"font" => "Dejavu Sans",
"xticklabelrotation" => 45.)

σa = sqrt(Correlation_matrix[1,1])
σb = sqrt(Correlation_matrix[2,2])

limits = [-4σa 4σa; -4σb 4σb]
ticks = [-3σa 3σa; -3σb 3σb]

canvas = FisherPlot.preparecanvas(LaTeXArray, limits, ticks, probes, colors, PlotPars::Dict)
FisherPlot.paintcorrmatrix!(canvas, central_values, Correlation_matrix, "deepskyblue3")
FisherPlot.save(joinpath(@OUTPUT, "fisher_contour.png"), canvas) # hide