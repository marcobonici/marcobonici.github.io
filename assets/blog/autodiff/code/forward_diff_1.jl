# This file was generated, do not modify it. # hide
using ForwardDiff
println(ForwardDiff.gradient(f, [0,0,0])) # hide
@benchmark ForwardDiff.gradient(f, [0,0,0])