@def title = "Notes on Automatic Differentiation"

# Plots

@@small-imgc \begin{tikzcd}{backprop}
& x \arrow[dr] \arrow[dl, shift right=1.5ex, "\bar{x}\frac{\partial t}{\partial x}" {sloped,above}]\\
t \arrow[ur] \arrow[dr] & & z \arrow[dl, shift left=1.5ex, "\bar{z}\frac{\partial z}{\partial y}" {sloped,below}] \arrow[ul, shift right=1.5ex, "\bar{z}\frac{\partial z}{\partial x}" {sloped,above}]\\
& y  \arrow[ur] \arrow[ul, shift left=1.5ex, "\bar{y}\frac{\partial t}{\partial y}" {sloped,below}]
\end{tikzcd}@@

@@small-imgc \begin{tikzcd}{tcd1}
t \arrow[r, "\phi"] \arrow[d, red]
  & x \arrow[d, "\psi" red] \\
  y \arrow[r, red, "\eta" blue]
  & z
\end{tikzcd}@@

@@small-imgc \begin{tikzcd}{tcd2}
A \arrow[r, "\phi" above , "\psi"'] & B
\end{tikzcd}@@



@@small-imgc \begin{tikzcd}{backprop1}
A \arrow[r, red, shift left=1.5ex] \arrow[r]
\arrow[dr, blue, shift right=1.5ex] \arrow[dr]
& B \arrow[d, purple, shift left=1.5ex] \arrow[d]\\
& C
\end{tikzcd}@@