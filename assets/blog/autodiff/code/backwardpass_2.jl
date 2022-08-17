# This file was generated, do not modify it. # hide
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4 \arrow[drrr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5 \arrow[rrr] & & &
+ \arrow[r] & w_8 \arrow[r] \arrow[ullll, green, shift right = 1.5ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_4}" {sloped,above} green] & y
\arrow [l, green, shift right=1.75ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & \exp \arrow[r] & w_7 \arrow[ur]
""")
save(SVG(joinpath(@OUTPUT, "backwardpass_2")), tp)