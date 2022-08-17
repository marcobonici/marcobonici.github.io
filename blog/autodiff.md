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
| $w_1$     | $0$    | 
| $w_2$     | $0$    | 
| $w_3$     | $0$    | 

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
| $w_1$     | $0$    | 
| $w_2$     | $0$    | 
| $w_3$     | $0$    | 
| $w_4$     | $0$    | 
| $w_5$     | $0$    | 
| $w_6$     | $0$    | 

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
| $w_1$     | $0$    | 
| $w_2$     | $0$    | 
| $w_3$     | $0$    | 
| $w_4$     | $0$    | 
| $w_5$     | $0$    | 
| $w_6$     | $0$    |
| $w_7$     | $1$    |

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
| $w_1$     | $0$    | 
| $w_2$     | $0$    | 
| $w_3$     | $0$    | 
| $w_4$     | $0$    | 
| $w_5$     | $0$    | 
| $w_6$     | $0$    |
| $w_7$     | $1$    |
| $w_8$     | $1$    |

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

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\bf{x}}
\end{equation}

```julia:backwardpass_1
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4 \arrow[drrr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5 \arrow[rrr] & & &
+ \arrow[r] & w_8 \arrow[r] & y
\arrow [l, green, shift right=1.ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & \exp \arrow[r] & w_7 \arrow[ur]
""")
save(SVG(joinpath(@OUTPUT, "backwardpass_1")), tp)
```
\fig{backwardpass_1}

\begin{equation}
\bar{x} = \frac{\mathrm{d}y}{\mathrm{d} x}
\end{equation}

\begin{equation}
\bar{y} = 1
\end{equation}



\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\bf{x}}=\bar{y}\frac{\partial y}{\partial w_8}\frac{\mathrm{d}w_8}{\mathrm{d} x}
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\bf{x}}=\frac{\mathrm{d}w_8}{\mathrm{d} x}
\end{equation}

```julia:backwardpass_2
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr] & \sin \arrow[r] & w_4 \arrow[drr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & \times \arrow[r] & w_5 \arrow[rr] &  & w_8 \arrow[r] \arrow[ull, green, shift right = 1.5ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_4}" {sloped,above} green] \arrow[ll, green, shift right = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_5}" {sloped,above} green] \arrow[dl, green, shift left = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_7}" {sloped,below} green] & y
\arrow [l, green, shift right=1.ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
x_3 \arrow[r] &  w_3 \arrow[r] & + \arrow[r] & w_6 \arrow[r] & w_7 \arrow[ur]
""")
save(SVG(joinpath(@OUTPUT, "backwardpass_2")), tp)
```
\fig{backwardpass_2}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\bf{x}}=\left( \frac{\partial w_8}{\partial w_4}\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\partial w_8}{\partial w_5}\frac{\mathrm{d}w_5}{\mathrm{d} x} + \frac{\partial w_8}{\partial w_7}\frac{\mathrm{d}w_7}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\bf{x}}=\left(\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\mathrm{d}w_5}{\mathrm{d} x} + \frac{\mathrm{d}w_7}{\mathrm{d} x} \right)
\end{equation}

```julia:backwardpass_3
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow[r] \arrow [dr]  & w_4 \arrow[drr, shift left=0.75ex]\\
x_2 \arrow[r] &  w_2 \arrow[r] \arrow[dr] & w_5 \arrow[rr] &  & w_8 \arrow[r] \arrow[ull, green, shift right = 1.5ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_4}" {sloped,above} green] \arrow[ll, green, shift right = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_5}" {sloped,above} green] \arrow[dl, green, shift left = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_7}" {sloped,below} green] & y
\arrow [l, green, shift right=1.ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
x_3 \arrow[r] &  w_3 \arrow[r] & w_6 \arrow[r] & w_7 \arrow[ur] \arrow [l, green, shift right=1.ex, "\bar{w}_7 \frac{\partial w_6}{\partial w_7}" {sloped,above} green] 
""", options ="scale = 1.")
save(SVG(joinpath(@OUTPUT, "backwardpass_3")), tp)
```
\fig{backwardpass_3}

```julia:backwardpass_4
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
```
\fig{backwardpass_4}

```julia:backwardpass_5
#hideall
using TikzCDs
tp = TikzCD(L"""
x_1 \arrow[r] &  w_1 \arrow [l, green, shift right=1.ex, "\bar{w}_1 \frac{\mathrm{d} w_1}{\mathrm{d} x_1}" {sloped,above} green] \arrow[r] \arrow [ddr]  & w_4 \arrow[ddrr, shift left=0.75ex] \arrow [l, green, shift right=1.ex, "\bar{w}_4 \frac{\partial w_4}{\partial w_1}" {sloped,above} green] \\
\\
x_2 \arrow[r] &  w_2 \arrow [l, green, shift right=1.ex, "\bar{w}_2 \frac{\mathrm{d} w_2}{\mathrm{d} x_2}" {sloped,above} green] \arrow[r] \arrow[ddr] & w_5 \arrow[rr] \arrow [uul, green, shift right=1.ex, "\bar{w}_5 \frac{\partial w_5}{\partial w_1}" {sloped,above} green] \arrow [l, green, shift right=1.ex, "\bar{w}_5 \frac{\partial w_5}{\partial w_2}" {sloped,near end} green] &  & w_8 \arrow[r] \arrow[uull, green, shift right = 1.5ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_4}" {sloped,above} green] \arrow[ll, green, shift right = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_5}" {sloped,above} green] \arrow[ddl, green, shift left = 0.9ex, "\bar{w}_8 \frac{\partial w_8}{\partial w_7}" {sloped,below} green] & y
\arrow [l, green, shift right=1.ex, "\bar{y} \frac{\partial y}{\partial w_8}" {sloped,above} green]\\
\\
x_3 \arrow[r] &  w_3 \arrow [l, green, shift right=1.ex, "\bar{w}_3 \frac{\mathrm{d} w_3}{\mathrm{d} x_3}" {sloped,above} green]  \arrow[r] & w_6 \arrow[r] \arrow [l, green, shift right=1.ex, "\bar{w}_6 \frac{\partial w_6}{\partial w_3}" {sloped,near end} green] \arrow [uul, green, shift right=1.ex, "\bar{w}_6 \frac{\partial w_6}{\partial w_2}" {sloped,above} green] & w_7 \arrow[uur] \arrow [l, green, shift right=1.ex, "\bar{w}_7 \frac{\partial w_6}{\partial w_7}" {sloped,above} green] 
""", options ="scale = 1.")
save(SVG(joinpath(@OUTPUT, "backwardpass_5")), tp)
```
\fig{backwardpass_5}

