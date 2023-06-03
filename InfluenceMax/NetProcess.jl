"""
	This File is used to pre-calculate some network parameters
	Transmission probabilities
	S distributions
"""

include("src/Graph.jl")
include("src/Tools.jl")
include("src/Sampling.jl")

using StatsBase
using JSON
using Feather
using DataFrames

# Read Graph
buf = split(ARGS[1], ',')
println(buf[1])
fileName = string(".in/", buf[1], ".txt")
networkType = "unweighted"
if size(buf, 1) > 1
	networkType = buf[2]
end
# Find LLC
G = readGraph(fileName, networkType)
n = G.n

######################## Edge Weight
G2 = assignWeightedCascade(G) 

########################

k = round(Int, parse(Float64, ARGS[2]) * n)
delta = 0.3

# rr-algorithm
(_, seeds, _, T) = ris_influence_max(G2, k, delta) ## RR to select seed nodes

nodes_select = [G.V[i] for i in seeds]

println("Now on dataset ", buf[1])
println("Now k is: ", k)
println("Spend time: ", T)

# Feather.write(".out/"*buf[1]*".feather", DataFrame(Dict("name"=>buf[1], "n"=> G.n, "seeds"=>nodes_select, "time"=>T)))