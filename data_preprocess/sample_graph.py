import networkx as nx
import random
import sys

# Reading the graph from a text file

# graph_name = sys.argv[1]

def bfs(G, start_node, upp_size):
    """
    Perform a breadth-first search (BFS) on a graph up to a maximum number of nodes.

    This function starts the BFS at a given node and continues until either all connected nodes have been visited
    or a specified upper limit (upp_size) on the number of visited nodes is reached.

    Parameters:
    G (networkx.Graph): The graph on which to perform the BFS.
    start_node (node): The node at which to start the BFS.
    upp_size (int): The upper limit on the number of nodes to visit.

    Returns:
    list: A list of the visited nodes in the order they were visited.
    """
    visited = []
    queue = [start_node]

    while queue and len(visited) < upp_size:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(n for n in G.neighbors(node) if n not in visited)

    return visited

def sample_connected_component(G, upp_size):
    """
    Perform a breadth-first search (BFS) from a randomly selected start node up to a maximum number of nodes, 
    and then extract the largest connected component of the visited nodes as a subgraph.

    This function picks a node at random from the graph, performs a BFS from that node, and collects the visited nodes. 
    It then creates a subgraph from these nodes and removes the start node. The function finally extracts and returns 
    the largest connected component of this subgraph.

    Parameters:
    upp_size (int): The upper limit on the number of nodes to visit during BFS.

    Returns:
    networkx.Graph: The largest connected component of the subgraph of visited nodes.
    """
    
    # Choose a start node uniformly at random from the graph
    start_node = random.choice(list(G.nodes()))

    visited_nodes = bfs(G, start_node, upp_size)

    #print("Nodes visited in BFS:", visited_nodes)

    subgraph = G.subgraph(visited_nodes)

    sample_graph = nx.Graph(subgraph)
    sample_graph.remove_node(start_node)
    
    connected_sample_graph = sample_graph.subgraph(max(nx.connected_components(sample_graph), key=len)).copy()
    
    return connected_sample_graph



graph_name = 'gplus/gplus.txt'
G = nx.read_edgelist(graph_name, nodetype = int)

sample_graph = sample_connected_component(G, 23000)
nx.write_edgelist(sample_graph, "gplus/gplusl2.txt", data=False)