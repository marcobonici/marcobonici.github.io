# This file was generated, do not modify it. # hide
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1\\
x_2 \arrow[r] &  w_2\\
x_3 \arrow[r] &  w_3
""", options="row sep=tiny")
save(SVG(joinpath(@OUTPUT, "forwardpass_1")), tp)