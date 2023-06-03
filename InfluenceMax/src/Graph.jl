include("Graph_class.jl")
using Random

function readGraph(fileName, graphType) # Read graph from a file | graphType = [weighted, unweighted]
    """
        2022-07-30
        Edges are stored according to id, and vertices are stored according to their original label. 
    """
    # Initialized
    n = 0
    origin = Dict{Int32, Int32}()
    label = Dict{Int32, Int32}()
    edge = Set{Tuple{Int32, Int32, Float64}}()

    getid(x :: Int32) = haskey(label, x) ? label[x] : label[x] = n += 1

    open(fileName) do f1
        for line in eachline(f1)
            # Read origin data from file
            buf = split(line)
            u = parse(Int32, buf[1])
            v = parse(Int32, buf[2])
            if graphType == "weighted"
                w = parse(Float64, buf[3])
            else
                w = 1.0
            end
            if u == v
                continue
            end
            # Label the node
            u1 = getid(u)
            v1 = getid(v)
            origin[u1] = u
            origin[v1] = v
            # Store the edge
            if u1 > v1
                u1, v1 = v1, u1
            end
            push!(edge, (u1, v1, w)) # each edge stores only once
        end
    end

    # Store data into the struct Graph
    m = length(edge)
    V = Array{Int32, 1}(undef, n)
    E = Array{Tuple{Int32, Int32, Int32, Float64}, 1}(undef, m)

    for i = 1 : n
        V[i] = origin[i]
    end

    ID = 0
    for (u, v, w) in edge 
        ID = ID + 1
        E[ID] = (ID, u, v, w)
    end

    return Graph(n, m, V, E)
end

function BFS(st, g) # Search from the node st
    q = []
    h = Array{Int8, 1}(undef, size(g, 1))
    fill!(h, 0)
    push!(q, st)
    h[st] = 1
    front = 1
    rear = 1
    while front <= rear
        u = q[front]
        front = front + 1
        for v in g[u]
            if h[v] == 0
                h[v] = 1
                rear = rear + 1
                push!(q, v)
            end
        end
    end
    return q
end

function getLLC(G :: Graph) # Find the largest link clique
    # Create the Link Table
    g = Array{Array{Int32, 1}, 1}(undef, G.n)
    for i = 1 : G.n
        g[i] = []
    end
    for (ID, u, v, w) in G.E
        push!(g[u], v)
        push!(g[v], u)
    end
    # Find the node Set of LLC
    n2 = 0
    S = Array{Int32, 1}(undef, G.n)
    visited = Array{Int8, 1}(undef, G.n)
    fill!(visited, 0)
    for i = 1 : G.n
        if visited[i] == 0
            nodeSet = BFS(i, g)
            for x in nodeSet 
                visited[x] = 1
            end
            tn = size(nodeSet, 1)
            if (tn > n2)
                n2 = tn
                for j = 1 : n2
                    S[j] = nodeSet[j]
                end
            end
        end
    end
    # Store the result
    label = Array{Int32, 1}(undef, G.n)
    fill!(label, 0)
    V2 = Array{Int32, 1}(undef, n2)
    for i = 1 : n2
        label[S[i]] = i
        V2[i] = G.V[S[i]]
    end
    E2 = []
    nID = 0
    for (ID, u, v, w) in G.E
        if (label[u] > 0) && (label[v] > 0)
            nID += 1
            push!(E2, (nID, label[u], label[v], w))
        end
    end
    m2 = size(E2, 1)

    #G2 = Graph(n2, m2, V2, E2)
    #g_new = Array{Array{Int32, 1}, 1}(undef, G2.n)
    #for i = 1 : G2.n
    #    g_new[i] = []
    #end
    #for (ID, u, v, w) in G2.E
    #    push!(g_new[u], v)
    #end
    
    #edgelist = BFS(1, g_new);
    #println("The size of travelled list is ", size(edgelist, 1))
    
    return Graph(n2, m2, V2, E2)
end


### New
### !!! NOTE: there exists only one edge between two nodes in the graph

## (1) Weighted cascade model to assign edge weights
function assignWeightedCascade(G :: Graph)
    g = Array{Array{Int32, 1}, 1}(undef, G.n)
    for i = 1 : G.n
        g[i] = []
    end
    for (ID, u, v, w) in G.E
        push!(g[v], u) # in-edges for v
    end
    p1 = Array{Float64, 1}(undef, G.n) # store edge probabilities
    for i = 1 : G.n
        if size(g[i], 1) > 0
            p1[i] = 1 / size(g[i], 1)
        else 
            p1[i] = 0
        end
    end

    E2 = []
    for (ID, u, v, w) in G.E
        push!(E2, (ID, u, v, p1[v])) #reassign edge weights according to weighted cascade model
    end

    return Graph(G.n, G.m, G.V, E2)
end
