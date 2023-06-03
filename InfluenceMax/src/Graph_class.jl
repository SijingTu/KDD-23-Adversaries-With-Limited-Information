struct Graph
    n :: Int32 # |V|
    m :: Int32 # |E|
    V :: Array{Int32, 1} # V[i] = Real Index of node i
    E :: Array{Tuple{Int32, Int32, Int32, Float64}, 1} # (ID, u, v, w) in Edge Set
end

