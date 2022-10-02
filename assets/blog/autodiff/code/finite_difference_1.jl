# This file was generated, do not modify it. # hide
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