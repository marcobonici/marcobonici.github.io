# This file was generated, do not modify it. # hide
canvas = FisherPlot.preparecanvas(LaTeXArray, limits, ticks, probes, colors, PlotPars::Dict)
FisherPlot.paintcorrmatrix!(canvas, central_values, C_WL, "deepskyblue3")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_GC, "darkorange1")
FisherPlot.paintcorrmatrix!(canvas, central_values, C_3x2_pt, "green")
FisherPlot.save(joinpath(@OUTPUT, "fisher_contour.png"), canvas) # hide