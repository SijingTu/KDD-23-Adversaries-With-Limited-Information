include("Graph_class.jl")
include("Tools.jl")

using Random
using SharedArrays
using Distributed

#addprocs(4)



"""
    list_of_nodes = getRRS(G :: Graph, n, delta)
    A: Activated nodes of all rounds

    2021-07-18, also return the nodes get picked
    
    2021-10-17, combine with getRRS_one_round

"""
function getRRS(g:: Array{Array{Tuple{Int32, Float64}, 1}, 1}, n::Number, delta::Number)
    rr_seeds = rand(1:n, 1)
    tmp = rand(sum([length(g[i]) for i in 1:n])) #m

    A = Set{Int32}(rr_seeds[1])
    round_seed = 1
    activated_nodes_of_last_round = A
    activated_nodes_of_this_round = Set{Int32}()
    count = 1 # random number 

    while length(activated_nodes_of_last_round) != 0
        for s in activated_nodes_of_last_round
            for nb in g[s]
                if nb[1] in A
                    continue
                end

                if round_seed == 1
                    if tmp[count] < nb[2]
                        push!(activated_nodes_of_this_round, nb[1])
                    end
                else
                    if tmp[count] < delta * nb[2]
                        push!(activated_nodes_of_this_round, nb[1])
                    end
                end                        
                count += 1
            end      
        end

        union!(A, activated_nodes_of_this_round)
        activated_nodes_of_last_round = activated_nodes_of_this_round
        activated_nodes_of_this_round = Set{Int32}()
        round_seed += 1
    end

    return toarray(A), rr_seeds[1]
end

"""
    node_selection_influence_max(R, k)

    Node selection procedure

    Return Fᵣ(SEED), and SEED
"""
function node_selection_influence_max(R, k)
    copy_k = k
    copy_R = deepcopy(R)

    SEED = zeros(Int32, copy_k)
    size = length(copy_R) # initial size 
    tmp_size,  cm, fc, ris_spread = 0, 0, 0, 0
    # The process of node-selection 
    for i in 1:copy_k
        if length(copy_R) == 0
            copy_k = i - 1 
            break
        end
        flat_list = vcat(copy_R...) # Flat the list 
        (cm, fc) = most_common(flat_list)
        tmp_size += fc
        SEED[i] = cm
        filter!(e -> !(cm in e), copy_R)
    end 

    ris_spread = tmp_size / size 

    return ris_spread, SEED[1:copy_k]
end


"""
    ris_influence_max(G::Graph, k, delta)

    delta: network parameter (ack, spread, reject)
    ℓ: accuracy parameter

    ris only for inluence maximization algorithm.
    Return: influence_spread, seed nodes, sample_size 

        Tang, Youze, Yanchen Shi, and Xiaokui Xiao. 
        "Influence maximization in near-linear time: A martingale approach." 
        Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. 2015.
"""
function ris_influence_max(G::Graph, k, delta = 0.5, ϵ = 0.3, ℓ = 1)
    # Create the "reversed" Link Table
    g = Array{Array{Tuple{Int32, Float64}, 1}, 1}(undef, G.n)
    for i = 1 : G.n
        g[i] = []
    end
   
    for (_, u, v, w) in G.E
        push!(g[v], (u, w))
    end
    
    T = time()

    n = G.n
    ϵ′ = √2 * ϵ
    λ′ = (2 + 2/3*ϵ′)*(logchoose(n, k) + ℓ * log(n) + log(log2(n)))*n / (ϵ′)^2
    λ⁺ = 4*n*(ϵ/3 + 2)*(ℓ * log(n) + log(2) + logchoose(n, k)) / (ϵ)^2
    LB = 1
    R, SEED = [], []
    ris_spread, x, θᵢ = 0, 0, 0

    for i in 1 : round(log2(n) - 1)
        x = n / (2^i)
        θᵢ = λ′ / x 
        if length(R) <= θᵢ
            append!(R, [getRRS(g, n, delta)[1] for _ in 1 : (θᵢ - length(R) + 1)])
        end

        ris_spread, _ = node_selection_influence_max(R, k)

        if n*ris_spread >= (1 + ϵ′)*x
            LB = n * ris_spread / (1 + ϵ′)
            break 
        end

    end

    θ = λ⁺ / LB 
    if length(R) <= θ
        append!(R, [getRRS(g, n, delta)[1] for _ in 1 : (θ - length(R) + 1)])
    end 

    ris_spread, SEED = node_selection_influence_max(R, k)


    T = time() - T
    return ris_spread * n, SEED, length(R), T
end



