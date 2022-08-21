# This file was generated, do not modify it. # hide
fx₀ = f([0,0,0])
Δx = 1e-6
∂f1 = (f([Δx,0,0])-fx₀)/Δx
∂f2 = (f([0,Δx,0])-fx₀)/Δx
∂f3 = (f([0,0,Δx])-fx₀)/Δx
println("∂f1=", ∂f1) # hide
println("∂f2=", ∂f2) # hide
println("∂f3=", ∂f3) # hide