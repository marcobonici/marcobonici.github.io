# This file was generated, do not modify it. # hide
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