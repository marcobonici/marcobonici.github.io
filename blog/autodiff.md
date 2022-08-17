@def title = "Notes on Automatic Differentiation"

# Example

\begin{equation}
f(x_1,x_2, x_3)= \sin x_1 + x_1 x_2 + \exp(x_2 + x_3)
\end{equation}

For instance, if $\mathbf{x}_0= (0,0,0)$

\begin{equation}
\nabla f(\mathbf{x}_0) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}\right)(\mathbf{x}_0) = ?
\end{equation}

## Symbolic approach

\begin{equation}
\frac{\partial f}{\partial x_1} = \cos x_1 + x_2
\end{equation}

\begin{equation}
\frac{\partial f}{\partial x_2} = x_1 + \exp(x_2+x_3)
\end{equation}

\begin{equation}
\frac{\partial f}{\partial x_3} = \exp(x_2+x_3)
\end{equation}

\begin{equation}
\nabla f(\mathbf{x}) = \left(\cos x_1 + x_2, x_1 + \exp(x_2+x_3), \exp(x_2+x_3)\right)(\mathbf{x})
\end{equation}

\begin{equation}
\nabla f(\mathbf{x}_0) = \left(\cos 0 + 0, 0 + \exp(0+0), \exp(0+0)\right) = \left(1,1,1 \right)
\end{equation}


## Backward algorithmic differentiation

### Forward pass

```julia:forwardpass_1
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1\\
x_2 \arrow[r] &  w_2\\
x_3 \arrow[r] &  w_3
""", options="row sep=tiny")
save(SVG(joinpath(@OUTPUT, "forwardpass_1")), tp)
```
\fig{forwardpass_1}

| Variable  | Value  | 
|-----------|--------|
| $w_1$     | 0      | 
| $w_2$     | 0      | 
| $w_3$     | 0      | 

```julia:forwardpass_2
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6
""")
save(SVG(joinpath(@OUTPUT, "forwardpass_2")), tp)
```
\fig{forwardpass_2}

| Variable  | Value  | 
|-----------|--------|
| $w_1$     | 0      | 
| $w_2$     | 0      | 
| $w_3$     | 0      | 
| $w_4$     | 0      | 
| $w_5$     | 0      | 
| $w_6$     | 0      | 

```julia:forwardpass_3
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & \exp \arrow[r] & w_7
""")
save(SVG(joinpath(@OUTPUT, "forwardpass_3")), tp)
```
\fig{forwardpass_3}

| Variable  | Value  | 
|-----------|--------|
| $w_1$     | 0      | 
| $w_2$     | 0      | 
| $w_3$     | 0      | 
| $w_4$     | 0      | 
| $w_5$     | 0      | 
| $w_6$     | 0      |
| $w_7$     | 1      |

```julia:forwardpass_4
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4 \arrow[drrr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5 \arrow[rrr] & & & + \arrow[r] & w_8\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & \exp \arrow[r] & w_7 \arrow[ur]
""")
save(SVG(joinpath(@OUTPUT, "forwardpass_4")), tp)
```
\fig{forwardpass_4}

| Variable  | Value  | 
|-----------|--------|
| $w_1$     | 0      | 
| $w_2$     | 0      | 
| $w_3$     | 0      | 
| $w_4$     | 0      | 
| $w_5$     | 0      | 
| $w_6$     | 0      |
| $w_7$     | 1      |
| $w_8$     | 1      |

```julia:forwardpass_5
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4 \arrow[drrr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5 \arrow[rrr] & & & + \arrow[r] & w_8 \arrow[r] & y\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & \exp \arrow[r] & w_7 \arrow[ur]
""")
save(SVG(joinpath(@OUTPUT, "forwardpass_5")), tp)
```
\fig{forwardpass_5}

### Backward pass

```julia:backwardpass_1
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4 \arrow[drrr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5 \arrow[rrr] & & &
+ \arrow[r] & w_8 \arrow[r] \arrow[ullll, green, shift right = 1.5ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_4}" {sloped,above} green] & y
\arrow [l, green, shift right=1.75ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & \exp \arrow[r] & w_7 \arrow[ur]
""")
save(SVG(joinpath(@OUTPUT, "backwardpass_1")), tp)
```
\fig{backwardpass_1}

```julia:backwardpass_2
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
```
\fig{backwardpass_2}




# Plots


@@small-imgc \begin{tikzcd}{backprop}
& x \arrow[dr] \arrow[dl, shift right=1.5ex, "\bar{x}\frac{\partial x}{\partial t}" {sloped,above}]\\
t \arrow[ur] \arrow[dr] & & z \arrow[dl, shift left=1.5ex, "\bar{z}\frac{\partial z}{\partial y}" {sloped,below}] \arrow[ul, shift right=1.5ex, "\bar{z}\frac{\partial z}{\partial x}" {sloped,above}]\\
& y  \arrow[ur] \arrow[ul, shift left=1.5ex, "\bar{y}\frac{\partial y}{\partial t}" {sloped,below}]
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