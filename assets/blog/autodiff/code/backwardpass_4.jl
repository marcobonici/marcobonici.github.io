# This file was generated, do not modify it. # hide
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [ddr]  & w_4 \arrow[ddrr, shift left=0.75ex] \arrow [l, green, shift right=1.ex, "\bar{w}_4 \frac{\partial w_4}{\partial w_1}" {sloped,above} green] \\
\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[ddr] & w_5 \arrow[rr] \arrow [uul, green, shift right=1.ex, "\bar{w}_5 \frac{\partial w_5}{\partial w_1}" {sloped,above} green] \arrow [l, green, shift right=1.ex, "\bar{w}_5 \frac{\partial w_5}{\partial w_2}" {sloped,near end} green] &  & w_8 \arrow[r] \arrow[uull, green, shift right = 1.5ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_4}" {sloped,above} green] \arrow[ll, green, shift right = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_5}" {sloped,above} green] \arrow[ddl, green, shift left = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_7}" {sloped,below} green] & y
\arrow [l, green, shift right=1.ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
\\
x_3 \arrow[r] &  w_3 \arrow[r] & w_6 \arrow[r] \arrow [l, green, shift right=1.ex, "\bar{w}_6 \frac{\partial w_6}{\partial w_3}" {sloped,near end} green] \arrow [uul, green, shift right=1.ex, "\bar{w}_6 \frac{\partial w_6}{\partial w_2}" {sloped,above} green] & w_7 \arrow[uur] \arrow [l, green, shift right=1.ex, "\bar{w}_7 \frac{\partial w_6}{\partial w_7}" {sloped,above} green] 
""", options ="scale = 1.")
save(SVG(joinpath(@OUTPUT, "backwardpass_4")), tp)