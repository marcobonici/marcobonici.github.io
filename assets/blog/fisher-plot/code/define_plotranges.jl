# This file was generated, do not modify it. # hide
central_values = [-1., 0., 0.06]

limits = zeros(3,2)
ticks = zeros(3,2)
for i in 1:3
    limits[i,1] = -4sqrt(C_WL[i,i])+central_values[i]
    limits[i,2] = +4sqrt(C_WL[i,i])+central_values[i]
    ticks[i,1]  = -3sqrt(C_WL[i,i])+central_values[i]
    ticks[i,2]  = +3sqrt(C_WL[i,i])+central_values[i]
end