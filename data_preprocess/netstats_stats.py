import numpy as np
from scipy.sparse import csgraph
import pandas as pd
import scipy
import scipy.io
import networkx as nx
import copy
import random

np.random.seed(1024)

def calculate_graph_metrics_and_pack_in_dict(n, m, s, L, k_0, d, name, tag, with_matrix=True):
    """
    This function calculates polarization and disagreement indices for a given graph, 
    and returns a dictionary with these indices along with other graph-related parameters.

    Parameters:
    n (int): Number of nodes in the graph.
    m (int): Number of edges in the graph.
    s (numpy.ndarray): A vector containing the innate opinions of the graph.
    L (numpy.ndarray): Laplacian matrix of the graph. It's an n by n matrix.
    k_0 (float): The size of smaller community. The community is created according to the opinion distribution. <0.5 or >0.5
    d (list of int): The list of degree of the nodes.  
    name (str): Name of the dataset.
    tag (int): How the innate opinions are assigned:
    - Tag = 1: Assign innate opinions directly according to normal distribution `mu` = 0.5, `sigma` = 0.2.
    - Tag = 2: Assign real innate opinions.
    with_matrix: whether return a full version, or clean version.

    Returns:
    dict: A dictionary containing the following elements:
        - 'n': number of nodes.
        - 'm': number of edges.
        - 'name': name of the graph.
        - 'tag': 1 or 2 or other numbers, according to how initial innate opinions are assigned. 
        - 'polMat': polarization matrix.
        - 'disMat': disagreement matrix.
        - 'polIdx': polarization index.
        - 'disIdx': disagreement index.
        - 's': initial innate opinions. 
        - 'k_0': The size of smaller community. The community is created according to the opinion distribution. <0.5 or >0.5
        - 'd': The list of degrees of the nodes.
    """
    pol, dis = calculate_polarization_disagreement_matrices(n, L)
    polidx = np.dot(np.dot(s, pol), s)
    disidx = np.dot(np.dot(s, dis), s)
    
    if with_matrix:
        dict_for_graph = {'n':n, 'm':m, 'name':name, 'tag':tag, 'polMat':pol, 'disMat': dis, 'polIdx': polidx, 'disIdx': disidx, 's':s, 'k_0':k_0, 'd':d}
    else:
        dict_for_graph = {'n':n, 'm':m, 'name':name, 'tag':tag, 'polIdx': polidx, 'disIdx': disidx, 's':s, 'k_0':k_0, 'd':d}
    
    return dict_for_graph


def create_edge_list(L, loc):
    """
    This function creates an edge list file from a given Laplacian Matrix.

    Parameters:
    L (numpy.ndarray): Input graph Laplacian matrix.
    loc (str): Path to the output file where the edge list will be written.

    Output:
    This function does not return any value. It writes an edge list to the specified file. The edge list is a list of pairs (i, j), where i and j are nodes that share an edge.
    """
    tmpL = copy.deepcopy(L)
    np.fill_diagonal(tmpL, 0) #obtain adjacency matrix
    A = -tmpL 
    G = nx.from_numpy_matrix(A)
    nx.write_edgelist(G, loc, data=False)
    

def generate_initial_innate_opinions(n, s, sigma = 0.2):
    """
    This function generates innate opinions for each node in a graph. 
    The opinions are drawn from a Gaussian distribution centered around a input 's' for each node, with a standard deviation of 'sigma'. 
    If the drawn opinion is greater than 1 or less than 0, it is set to 1 or 0 respectively. 

    Parameters:
    n (int): Number of nodes in the graph.
    s (numpy.ndarray): A vector containing the mean of initial innate opinion of each node in the graph. This acts as the mean for the Gaussian distribution.
    sigma (float, optional): Standard deviation for the Gaussian distribution. Defaults to 0.2.

    Returns:
    numpy.ndarray: A vector of size 'n' representing the generated innate opinions of the nodes.
    """

    np.random.seed(123)
    index = np.array([np.random.normal(s[i], sigma) for i in range(n)])
    for i in range(n):
        if index[i] > 1:
            index[i] = 1
        if index[i] < 0:
            index[i]= 0
    return index

def extract_small_reddit_parameters_and_matrices():
    # Based on Chen and Racz's codes
    
    reddit_data = scipy.io.loadmat("small/Reddit.mat")
    
    n = reddit_data['Reddit'][0,0][0].shape[0]     # number of vertices = 556
    A = reddit_data['Reddit'][0,0][0].toarray()     # adjacency matrix in compressed sparse column format, convert to array
    s = reddit_data['Reddit'][0,0][5]     # labeled "recent innate opinions"


    # remove isolated vertices from the graph
    s = np.delete(s, 551)
    s = np.delete(s, 105)
    s = np.delete(s, 52)
    n -= 3

    A = np.delete(A, 551, 1)
    A = np.delete(A, 551, 0)
    A = np.delete(A, 105, 1)
    A = np.delete(A, 105, 0)
    A = np.delete(A, 52, 1)
    A = np.delete(A, 52, 0)

    L = scipy.sparse.csgraph.laplacian(A, normed=False)
    m = np.sum(np.diag(L), dtype=int) // 2

    d = np.diag(L)

    k_0 = min(sum(s>0.5), sum(s<0.5))
    
    return n, m, s, L, k_0, d

def extract_small_twitter_parameters_and_matrices():
    
    # based on Chen and Racz's code
    
    s_df = pd.read_csv('small/preprocess-twitter/opinion_twitter.txt', sep = '\t', header = None)
    w_df = pd.read_csv('small/preprocess-twitter/edges_twitter.txt', sep = '\t', header = None)
    # number of vertices
    n = len(s_df[0].unique())
    s_df.columns = ["ID", "Tick", "Opinion"]

    # we take the opinion from the last appearance of the vertex ID in the list as its innate opinion
    s = s_df.groupby(["ID"]).last()["Opinion"].values

    # create adjacency matrix
    A = np.zeros((n, n))
    for i in range(1, n + 1):
        idx = np.where(w_df[0].values == i)[0]
        js = w_df[1].values[idx]
        for j in js:
            A[i-1, j-1] = 1
            A[j-1, i-1] = 1
            
        idx = np.where(w_df[1].values == i)[0]
        js = w_df[0].values[idx]
        for j in js:
            A[i-1, j-1] = 1
            A[j-1, i-1] = 1


    L = scipy.sparse.csgraph.laplacian(A, normed=False)
    m = np.sum(np.diag(L), dtype=int) // 2
    d = np.diag(L)
    #[pol, dis, internal] = build_constants(n, L)

    #s = 2 *(s - 0.5) # rescale s in [0, 1] to [-1, 1]

    k_0 = min(sum(s>0.5), sum(s<0.5))
    
    return n, m, s, L, k_0, d


def extract_small_graph_parameters_and_matrices(graph):
    """
    This function constructs the adjacency matrix and the Laplacian matrix for a given graph. 
    It also calculates various parameters such as the number of nodes, innate opinions of the nodes, the degree of nodes, 
    and the size of the smaller community.

    Parameters:
    graph (str): Path to the folder containing the graph data. 
    The folder is expected to contain 'com1.txt' and 'com2.txt' files representing two communities 
    and 'edges.txt' file representing the edges of the graph.

    Returns:
    tuple: A tuple containing the following elements:
        - n (int): Number of nodes in the graph.
        - m (int): Number of edges in the graph.
        - s (numpy.ndarray): Initial innate opinions of the nodes. The opinions are generated from gaussian distribution N(0.1,0.2) and N(0.3,0.2).
        - L (numpy.ndarray): Laplacian matrix of the graph. It's an n by n matrix.
        - k_0 (int): Size of the smaller community in the graph.
        - d (numpy.ndarray): Degree of each node in the graph.

    The innate opinions 's' are initialized to N(0.1,0.2) for community 1 nodes and N(0.3,0.2) for community 2 nodes, 
    and then diversity index is added to these values using the 'create_diversity_index' function (not defined in this code snippet).
    """
    
    # Define the paths to community 1 and community 2 files
    cm1 = graph + "/com1.txt"
    cm2 = graph + "/com2.txt"
    
    # Load community 1 and community 2 data
    c1 = np.loadtxt(cm1, dtype = int)
    c2 = np.loadtxt(cm2, dtype = int)
    
    # Get the total number of nodes in the graph
    n = len(c1) + len(c2)
    
    # Initialize adjacency matrix and degree list
    A = np.zeros((n, n))
    d = np.zeros(n)

    # Construct the adjacency matrix by reading edge data from 'edges.txt' file
    with open(graph + '/edges.txt', "r") as ins:
        for line in ins:
            edge = line.strip("\n").split(" ")
            e0 = int(edge[0])
            e1 = int(edge[1])
            A[e0, e1] = 1
            A[e1, e0] = 1

    # Calculate the Laplacian matrix
    L = csgraph.laplacian(A, normed=False)
    # Calculate the number of edges in the graph
    m = np.sum(np.diag(L), dtype=int) // 2
    
    # Initialize innate opinions and add diversity index
    s = np.ones(n) * 0.1
    s[c2] = 0.3
    s = generate_initial_innate_opinions(n, s, sigma=0.1) # (0.1, 0.1), (0.3, 0.1)
    
    # Calculate degree of each node
    for i in range(n):
        d[i] = L[i,i]
        
    # Get the size of the smaller community
    k_0 = min(len(c1), len(c2))

    return n, m, s, L, k_0, d


def get_large_graph_matrix(graph, type = 0):
    """
    Construct large_graph matrix, with one file containing edge list, another file containing the nodes leaning.  
    
    Args:
        graph ([string]): the folder where data stores
        type (int): if type = 0, the opinion is in range -1, 1 then we need to scale it; if type == 1, then opinion is in range 0, 1, we do not need to do anything. 
    
    Returns:
        n [int]: number of nodes
        s [list of n elements]: innate opinions, entries are either 1 or -1
        A [float n*n]: adjacency matrix
        L [float n*n]: graph laplacian    
        k_0 [int]: the size of the smaller part of the community
        d [int n]: a list of degree of nodes
    """

    leaning = graph + "/leanings_nodes.txt"
    if type == 0:
        s = (np.loadtxt(leaning) + 1)/2 # load s, and rescale s to [0, 1]
    elif type ==1:
        s = np.loadtxt(leaning)
    
    n = len(s)

    A = np.zeros((n, n))
    d = np.zeros(n)

    with open(graph + '/graph.txt', "r") as ins:
        for line in ins:
            edge = line.strip("\n").split(" ")
            e0 = int(edge[0])
            e1 = int(edge[1])
            A[e0, e1] = 1
            A[e1, e0] = 1

    L = csgraph.laplacian(A, normed=False)
    m = np.sum(np.diag(L), dtype=int) // 2
    
    for i in range(n):
        d[i] = L[i,i]
        
    k_0 = min(len(s[ np.where( s > 0.5 ) ]), len(s[ np.where( s < 0.5 ) ]))

    # ensure most people hold "peaceful opinions", 2022-10-09
    if len(s[np.where(s > 0.5)]) > n /2: 
        s = 1-s #flip all the opinions

    return n, m, s, L, k_0, d

#its old name is build_constants
def calculate_polarization_disagreement_matrices(n, L):
    """
    This function constructs two matrices that are used for calculating polarization and disagreement in a graph.

    Parameters:
    n (int): Number of nodes in the graph.
    L (numpy.ndarray): Laplacian matrix of the graph. It's an n by n matrix.

    Returns:
    list: A list containing two matrices.
        1. The first matrix is used for calculating polarization: (I+L)^-1(I - 11T/n)(I+L)^-1.
        2. The second matrix is used for calculating disagreement: (I+L)^-1L(I+L)^-1.

    Note:
    I represents an identity matrix of size n by n. 
    11T/n represents a matrix with all entries as 1/n.
    The '^-1' operation represents matrix inversion.
    """
    invIL = np.linalg.inv(np.eye(n) + L)
    cM = np.eye(n) - np.ones((n, n))/n 
    
    return [np.dot(np.dot(invIL, cM), invIL), np.dot(np.dot(invIL, L), invIL)]


def graph_and_leaning(leaning, graph):
    """From the edge list and the leaning, get the connected graph, political leaning from 0

    Args:
        leaning (.csv file): node and its political leaning
        graph (.txt): edge list
    """
    
    G = nx.read_edgelist(graph)
    arr = np.loadtxt(leaning, dtype='str', delimiter=',', skiprows=1)
    
    score = []
    node_dic = {}
    ind = 0
    for row in arr:
        score.append(row[1])
        node_dic[row[0]] = ind
        ind += 1
        
    new_edge_list = []
    for edge in G.edges():
        new_edge_list.append([node_dic[edge[0]], node_dic[edge[1]]])
    np.savetxt("twitter/graph.txt", new_edge_list, fmt='%s %s') # replace with location 1
    
    np.savetxt("twitter/leanings_nodes.txt", score, fmt='%s') # replace with location 2
    

# The SBM part. 

def generate_connected_sbm_graph(n, k, p_in, p_out):
    """
    Generate a connected graph using the stochastic block model (SBM).
    If the graph is not connected, add edges randomly between different connected components until it is.

    Parameters:
    n (int): Total number of nodes in the graph.
    k (int): Number of communities.
    p_in (float): Probability of edges within communities.
    p_out (float): Probability of edges between communities.

    Returns:
    networkx.Graph: A connected graph generated with the SBM.
    """
    
    # Ensure the number of nodes can be evenly divided into communities
    assert n % k == 0, "The total number of nodes must be divisible by the number of communities"
    
    # Number of nodes in each community
    n_community = n // k

    # Create the sizes and the probability matrix for the SBM
    sizes = [n_community] * k
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    
    G = nx.stochastic_block_model(sizes, probs)
    
    # Check if the graph is connected
    n_connected_components, labels = csgraph.connected_components(nx.adjacency_matrix(G))
    
    # If the graph is not connected, add edges randomly between different connected components until it is
    while n_connected_components > 1:
        # Get nodes from different connected components
        components = [[] for _ in range(n_connected_components)]
        for node, label in enumerate(labels):
            components[label].append(node)
        
        component1, component2 = random.sample(components, 2)

        # Pick one node from each component
        node1 = random.choice(component1)
        node2 = random.choice(component2)

        # Add an edge between them
        G.add_edge(node1, node2)

        n_connected_components, labels = csgraph.connected_components(nx.adjacency_matrix(G))
    
    return G

    
def extract_SBM_graph_parameters_and_matrices():
    """
    Extract graph parameters and matrices for a stochastic block model (SBM) graph.

    This function generates an SBM graph with given parameters, computes the adjacency and Laplacian matrices, 
    and generates an innate opinions vector with a Gaussian distribution around specific values for each community. 
    Additionally, the function computes the degree of each node and the size of the smaller community.

    Process:
    1. Generates a connected SBM graph with 1000 nodes, 4 equal-sized communities, and intra- and inter-community
       edge probabilities of 0.4 and 0.1, respectively.
    2. Defines the initial opinions for each community as [0.2, 0.3, 0.4, 0.5].
    3. Creates the adjacency matrix from the graph and computes the Laplacian matrix.
    4. Calculates the number of edges in the graph.
    5. Creates an innate opinions vector with values drawn from a Gaussian distribution around the initial values 
       for each community.
    6. Computes the degree of each node.
    7. Determines the size of the smaller community.

    Returns:
    n (int): The number of nodes in the graph.
    m (int): The number of edges in the graph.
    s (numpy.ndarray): The vector of innate opinions.
    L (numpy.ndarray): The Laplacian matrix of the graph.
    k_0 (int): The size of the smaller community.
    d (numpy.ndarray): The degree of each node in the graph.
    """
    # The rest of your function code here

    
    G = generate_connected_sbm_graph(1000, 4, 0.3, 0.1) # create SBM
    s_values = [0.2, 0.3, 0.4, 0.5] # create initial opinions. 
    k = 4
    
    # Number of nodes in each community
    n = len(G.nodes)
    n_community = len(G.nodes) // k
    d = np.zeros(n) # initialize d
    
    # Create the adjacency matrix from the graph
    A = nx.adjacency_matrix(G).toarray()
    # Calculate the Laplacian matrix
    L = csgraph.laplacian(A, normed=False)
    # Calculate the number of edges in the graph
    m = np.sum(np.diag(L), dtype=int) // 2
    
    # Create the vector of innate opinions
    s = np.concatenate([np.full(n_community, value) for value in s_values])
    s = generate_initial_innate_opinions(n, s, sigma=0.1) # (0.1, 0.1), (0.3, 0.1)
    
    # Calculate degree of each node
    for i in range(n):
        d[i] = L[i,i]
        
    # Get the size of the smaller community
    k_0 = n_community

    return n, m, s, L, k_0, d
    
