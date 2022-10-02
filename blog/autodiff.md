@def title = "Notes on Automatic Differentiation"

\warning{This is not meant to be a rigouros explanation of automatic differentiation.
These must be considered as some notes that I wrote for myself, to better understand the
topic of automatic differentiation. Furthermore, these notes are far to be completed and
need to be refined.}

Several computational techniques (minimization algorithms, training of Deep Neural Networks,
Hamiltonian MonteCarlo) requires the computation of gradients, jacobian, and hessians. While
we all learnt how to compute these quantities during calculus courses, these techniques may
be not well suited when dealing with numerical computations.

In the reminder of this post, I'll walk through the main techniques that can be used to
compute derivatives:

\toc

# Example
Let us consider the following function $f(\boldsymbol{x}):\mathbb{R}^3\rightarrow\mathbb{R}$
\begin{equation}
f(x_1,x_2, x_3)= \sin x_1 + x_1 x_2 + \exp(x_2 + x_3)
\end{equation}

For instance, if $\boldsymbol{x}_0= (0,0,0)$, which is the value of the gradient of $f$
evaluated at $\boldsymbol{x}_0$?

\begin{equation}
\nabla f(\boldsymbol{x}_0) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}\right)(\boldsymbol{x}_0) = ?
\end{equation}

## Symbolic approach

This is the classical approach that we all learnt during calculus courses: you simply have
to write down the analytical derivatives and compute their values

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
\nabla f(\boldsymbol{x}) = \left(\cos x_1 + x_2, x_1 + \exp(x_2+x_3), \exp(x_2+x_3)\right)(\boldsymbol{x})
\end{equation}

\begin{equation}
\nabla f(\boldsymbol{x}_0) = \left(\cos 0 + 0, 0 + \exp(0+0), \exp(0+0)\right) = \left(1,1,1 \right)
\end{equation}

Quite easy, isn't it?
What are the advantages of this approach?
- ✅ High precision, since derivatives are analytical
- ✅ As long as you have an analytical expression for the input function, this approach works

However, this approach has several drawbacks
- ❌ For complicated functions, the derivative may not have a maneageable analytical expression
- ❌ We don't always have an analytical functions to differentiate. For instance, when dealing with Differential Equations, we often have only a numerical solution

Some of these problems can be alleviated with the next approach: finite difference derivatives.

## Finite difference derivatives

The second approach replaces the limit in the derivative definition with a finite difference
derivative:
\begin{equation}
f^{\prime}(x)=\lim _{\epsilon \rightarrow 0} \frac{f(x+\epsilon)-f(x)}{\epsilon}\approx \frac{f(x+\Delta x)-f(x)}{\Delta x}
\end{equation}

If we have a function with more variables, this approach need to be extended to each of the
variables involved. Let us code it in Julia!
```julia:func_def
f(x) = sin(x[1])+x[1]*x[2]+exp(x[2]+x[3])
```

What about the efficiency of this function?
```julia:benchmark
using BenchmarkTools
@benchmark f([0,0,0])
```
\show{benchmark}
Let us evaluate the finite difference derivative![^finitedifference]
```julia:finite_difference_1
using LinearAlgebra
f(x) = sin(x[1])+x[1]*x[2]+exp(x[2]+x[3]) # hide

function  ∇f(f, x, ϵ)
    gradf = zeros(size(x))
    ϵ_matrix = ϵ * Matrix(I, length(x), length(x))
    for i in 1:length(x)
        gradf[i] = (f(x+ϵ_matrix[i,:])-f(x-ϵ_matrix[i,:]))/ϵ
    end
    return gradf
end

gradf = ∇f(f,[0,0,0], 1e-4)

println("∂f1=", gradf[1]) # hide
println("∂f2=", gradf[2]) # hide
println("∂f3=", gradf[3]) # hide
```
Let us see the result of the calculation!
\show{finite_difference_1}
Nice! We have evaluated the required gradient. However, the result is imprecise: there is a
truncation error! We have approximated the derivative and this gave us an imprecise result.
Furthermore, the error depends on the chosen step-size. If the step-size is too big, we are
not going to approximate the derivative...
```julia:finite_difference_2
gradf = ∇f(f,[0,0,0], 1e-1)

println("∂f1=", gradf[1]) # hide
println("∂f2=", gradf[2]) # hide
println("∂f3=", gradf[3]) # hide
```
\show{finite_difference_2}
...on the other hand, a small step-size will incure on floating-precision error
```julia:finite_difference_3
gradf = ∇f(f,[0,0,0], 1e-15)

println("∂f1=", gradf[1]) # hide
println("∂f2=", gradf[2]) # hide
println("∂f3=", gradf[3]) # hide
```
\show{finite_difference_3}
On the performance side, 

```julia:finite_difference_4
@benchmark gradf = ∇f(f,[0,0,0], 1e-4)
```
\show{finite_difference_4}
## Forward algorithmic differentiation

```julia:forward_diff_1
using ForwardDiff
println(ForwardDiff.gradient(f, [0,0,0])) # hide
@benchmark ForwardDiff.gradient(f, [0,0,0])
```
\show{forward_diff_1}

## Backward algorithmic differentiation
In this part, I'll show a different approach to compute gradient: backward automatic
differentiation. I'll write all steps and show them graphically. **Disclaimer**: I am going
to be a bit tedious.

### Forward pass

The first step requires the computation of the function. While computing the function, we
define some intermediate variables $w_i$ and store their values, writing down the
_Wengert list_. Let us start from the first step.

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
This was an easy one: basically, we simply had to define three variables, corresponding to
the three inputs. Let's move forward!

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
As we can see, we have included some real functions ($\sin$, $+$, ...) and combined the
previous step.

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
Let's just keep going on...

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

...and on...

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

...and here we are! We have computed the Wengert list and the output of the function. While
this look trivial and useless, we have evaluated al quantities required for the next step!

### Backward pass
We are now in the position to compute the gradient of the function.
Let us start defining the _adjoint_ of a quantity $x$, which is mapped in another quantity
$y$:

\begin{equation}
\bar{x} = \frac{\mathrm{d}y}{\mathrm{d} x}
\end{equation}

This quantity and the chain rule will be the key ingredients in the forecoming calculations.
Using the chain rule, we will start from the output, $y$, and we will pull back the
calculations, till we will reach the beginning of the calculation. How can we use the chain
rule? The gradient che be rewritten as 

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\frac{\partial y}{\partial w_8}\frac{\mathrm{d}w_8}{\mathrm{d} x}
\end{equation}

Since we know the relation between $y$ and $w_8$ we can compute the partial derivative we
added

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\frac{\mathrm{d}w_8}{\mathrm{d} x}
\end{equation}
But now, let's get back to the graph! I'll add in green each calcuation we have been doing


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

Let's keep going!
\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left( \frac{\partial w_8}{\partial w_4}\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\partial w_8}{\partial w_5}\frac{\mathrm{d}w_5}{\mathrm{d} x} + \frac{\partial w_8}{\partial w_7}\frac{\mathrm{d}w_7}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\mathrm{d}w_5}{\mathrm{d} x} + \frac{\mathrm{d}w_7}{\mathrm{d} x} \right)
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

Up to now, everything has been quite easy. Let's continue with the next step!

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\mathrm{d}w_5}{\mathrm{d} x} + \frac{\partial w_7}{\partial w_6}\frac{\mathrm{d}w_6}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\mathrm{d}w_5}{\mathrm{d} x} + \exp (w_6)\frac{\mathrm{d}w_6}{\mathrm{d} x} \right)
\end{equation}
Finally! Has we can see evaluating the partial derivative requires the value of $w_6$. But
we already know this values, since it is stored in the Wengert list!

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\mathrm{d}w_4}{\mathrm{d} x} + \frac{\mathrm{d}w_5}{\mathrm{d} x} + \frac{\mathrm{d}w_6}{\mathrm{d} x} \right)
\end{equation}

Now, the meaning of the first step should be clearer: we have evaluated and stored all
intermediate quantities. In this way, while moving back along the computationad graph, we
already have all quantities required to compute the gradient!


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

Let's keep going!

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\left(\frac{\partial w_4}{\partial w_1} \frac{\mathrm{d}w_1}{\mathrm{d} x}\right) + \left(\frac{\partial w_5}{\partial w_1} \frac{\mathrm{d}w_1}{\mathrm{d} x}+\frac{\partial w_5}{\partial w_2} \frac{\mathrm{d}w_2}{\mathrm{d} x}\right) + \left(\frac{\partial w_6}{\partial w_2} \frac{\mathrm{d}w_2}{\mathrm{d} x}+\frac{\partial w_6}{\partial w_3} \frac{\mathrm{d}w_3}{\mathrm{d} x}\right) \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\left(\frac{\partial w_4}{\partial w_1}+\frac{\partial w_5}{\partial w_1} \right)\frac{\mathrm{d}w_1}{\mathrm{d} x} + \left(\frac{\partial w_5}{\partial w_2}+\frac{\partial w_6}{\partial w_2} \right)\frac{\mathrm{d}w_2}{\mathrm{d} x} +  \frac{\partial w_6}{\partial w_3} \frac{\mathrm{d}w_3}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\left(\cos w_1+w_2 \right)\frac{\mathrm{d}w_1}{\mathrm{d} x} + \left(w_1+1 \right)\frac{\mathrm{d}w_2}{\mathrm{d} x} +  1 \frac{\mathrm{d}w_3}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\left(1+0 \right)\frac{\mathrm{d}w_1}{\mathrm{d} x} + \left(0+1 \right)\frac{\mathrm{d}w_2}{\mathrm{d} x} +  1 \frac{\mathrm{d}w_3}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\mathrm{d}w_1}{\mathrm{d} x} + \frac{\mathrm{d}w_2}{\mathrm{d} x} +  \frac{\mathrm{d}w_3}{\mathrm{d} x} \right)
\end{equation}




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

We are almost there!


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

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\partial w_1}{\partial x_1}\frac{\mathrm{d}x_1}{\mathrm{d} x} + \frac{\partial w_2}{\partial x_2}\frac{\mathrm{d}x_2}{\mathrm{d} x} +  \frac{\partial w_3}{\partial x_3}\frac{\mathrm{d}x_3}{\mathrm{d} x} \right)
\end{equation}

\begin{equation}
\frac{\mathrm{d}y}{\mathrm{d}\boldsymbol{x}}=\left(\frac{\partial w_1}{\partial x_1} + \frac{\partial w_2}{\partial x_2} +  \frac{\partial w_3}{\partial x_3}\right)
\end{equation}

We have now written the gradient of our function! We three terms, each of the proportional
to a partial derivative. The coefficient multiplying each of these derivatives is the
corresponding element of the gradient! Thus, we can conclude that the calculation give the
same result as before!

### References and Footnotes
[^finitedifference]: This is not the most efficient way to code the finite difference derivative, it is just something quick and dirt to show the method. A more efficient implementation can be found in (FiniteDifference.jl)[https://github.com/JuliaDiff/FiniteDifferences.jl]