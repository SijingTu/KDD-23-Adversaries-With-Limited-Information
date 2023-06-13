"""
    This script is applied to calculate the following statistics. The script should only be used for small scale datasets. 
    Handel datasets under folder `\data`
    `m`: number of edges
    `n`: number of vertices
    `tag`: 1: uniformly randomly assign innate opinions, 2: real data
    `init polarization`: initial polarization index
    `init disagreement`: initial disagreement index
    `matrix polarization`, `matrix disagreement`, `matrix conflict`.
    `k_0`: size of smaller community
    `d`: average degree of a node      
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from netstats_stats import *

import sys

data_type = int(sys.argv[1]) # which type of the indices that are going to be created. 

"""
Dataset 1: Run all the datasets under folder small/data. Tag = 1: Assign innate opinions directly according to normal distribution
`mu` = 0.5, `sigma` = 0.2

Dataset 2: Initialize reddit dataset, same as Mayee F. Chen and Miklós Z. Rácz's github codes: [gitlink](https://github.com/mayeechen/network-disruption)

Dataset 3: Initialize the twitter data, same as Mayee F. Chen and Miklós Z. Rácz's github codes: [gitlink](https://github.com/mayeechen/network-disruption)
"""


if data_type == 1:
    # Dataset 1
    #np.random.seed(1) # set up random seeds
    df = pd.DataFrame()
    FileList = [x[0] for x in os.walk("small/data")][1:]
    for FileName in FileList:
        [n, m, s, L, k_0, d] = extract_small_graph_parameters_and_matrices(FileName)
        #create_edge_list(L, "edgelist/"+os.path.split(FileName)[-1] + ".txt")
        dict_for_graph = calculate_graph_metrics_and_pack_in_dict(n, m, s, L, k_0, d, os.path.split(FileName)[-1], 1)
        df = df.append(dict_for_graph, ignore_index = True)

    # Dataset 2
    [n, m, s, L, k_0, d] = extract_small_reddit_parameters_and_matrices()
    #create_edge_list(L, "edgelist/"+'reddit' + ".txt")
    dict_for_graph = calculate_graph_metrics_and_pack_in_dict(n, m, s, L, k_0, d, 'reddit', 2)
    df = df.append(dict_for_graph, ignore_index = True)

    # Dataset 3
    [n, m, s, L, k_0, d] = extract_small_reddit_parameters_and_matrices()
    #create_edge_list(L, "edgelist/"+'real_twitter' + ".txt")
    dict_for_graph = calculate_graph_metrics_and_pack_in_dict(n, m, s, L, k_0, d, 'real_twitter', 2)
    df = df.append(dict_for_graph, ignore_index = True)

    df.to_pickle("../data/small_keep.pkl") # the datasets are the old datasets, that do not need to be replaced. 


# create a SBM dataset

if data_type == 2:
    # Dataset SBM
    df = pd.DataFrame()
    np.random.seed(1024)
    [n, m, s, L, k_0, d] = extract_SBM_graph_parameters_and_matrices()
    create_edge_list(L, "../InfluenceMax/.in/"+'sbm' + ".txt")
    dict_for_graph = calculate_graph_metrics_and_pack_in_dict(n, m, s, L, k_0, d, 'sbm', 1)
    df = df.append(dict_for_graph, ignore_index = True)

    df.to_pickle("../data/sbm.pkl") # sbm dataset 