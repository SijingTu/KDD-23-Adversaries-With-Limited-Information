include("Graph_class.jl")

using DataStructures # contain function: counter
using SpecialFunctions # contains function: logfactorial()
using Random
using SparseArrays
using Distributed



# (2) transform set to vector
@everywhere toarray(s::Union{Set, Vector}) = [toarray.(s)...]
@everywhere toarray(v::Number) = v

"""
    (3) Find the most common element in a vector
"""
function most_common(a) 
    c = collect(counter(a))
    pair = c[findmax(map(x->x[2], c))[2]]
    return pair[1], pair[2]
end

# (4) logfactorial
logfact(x, y) = logfact(x) / logfact(y)
logfact(x) = logfactorial(x)

function logchoose(N, k)
    return logfact(N) - logfact(k) - logfact(N-k)
end

# (5) topk, find the top k values in a vector
function topk(a, k)
    sort_a = sortperm(a, rev=true)
    if k <= length(a)
        return sort_a[1:k]
    else
        return sort_a[1:length(a)]
    end
end

