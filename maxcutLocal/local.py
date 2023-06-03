"""
Obtain various results
"""
import time
import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('..', 'algorithm')))
from MaxCutGSP import *
from MaxCutLocal import *

def func_topology_changes(P, s, k, x_GSP_list):
    """First obtain GSP, then perform local changes. Note we obtain a ilst of x_GSP 
    2022-08-04
    ratio for self_change, and reverse_change denotes the how much the first part weighs on the whole formula.
    ratio for random_change, denotes the std that we run the algorithm several times. 

    Args:
        P (float[n, n]): PolMat, DisMat, or IntMat.  
        s_local (float[n]): the average of local opinions.
        s (float[n]): vector of innate opinions
        k (int): cardinality constraint
        x_GSP (float[n]): GSP solution
    
    Returns:
        FinalScore, FinalTime
    """
    std = 0 # initialize
        
    # Perform local changes. 
    starttime = time.time()
    LocalPreScore, _, std = positive_changes(x_GSP_list, s, P, k) # change the innate opinions of seed nodes to 1
        
    endtime = time.time() - starttime
        
    print("Local time: " + str(endtime)) # printout the local time
    print("Local score: " + str(LocalPreScore)) # printout the local score
    if (x_GSP_list.ndim ==2):
        print("std is:", std) #ONLY WORK FOR 2 & 3
    

    return [LocalPreScore, endtime, std]

def func_NonAdaptiveGreedy(P, n, s, k):
    """ non adaptive greedy

    Args:
        P (float[n, n]): PolMat, DisMat, or IntMat.
        n (int): number of nodes.
        s (float[n]): vector of innate opinions
        k (int): cardinality constraint.
    
    Returns:
        SpScore (float): Sp score.
        SpTime (float): Sp time.
        x_GSP (float[n]): GSP solution.
    """
    
    P_greedy, c  = construct_greedy_constants(s, P) 

    StartSpGreedyTime = time.time()
    SpScore, _  = NonAdaptiveGreedy(k, n, P_greedy, c)
    SpTime = time.time() - StartSpGreedyTime
    
    print("SpScore: " + str(SpScore), flush=True)
        
    return [SpScore, SpTime] 

        
def func_AdaptiveGreedy(P, n, s, k):
    """Greedy funciton from Chen and Racz. 

    Args:
        P (float[n, n]): PolMat, DisMat, or IntMat.
        n (int): number of nodes.
        s (float[n]): vector of innate opinions
        k (int): cardinality constraint.
    
    Returns:
        SpScore (float): Sp score.
        SpTime (float): Sp time.
        x_GSP (float[n]): GSP solution.
    """
    
    P_greedy, c  = construct_greedy_constants(s, P) 

    StartSpGreedyTime = time.time()
    SpScore, _  = AdaptiveGreedy(k, n, P_greedy, c)
    SpTime = time.time() - StartSpGreedyTime
    
    print("SpScore: " + str(SpScore), flush=True)
        
    return [SpScore, SpTime]  
    

# def choice_pad(FileLoc, method, Extra, ratio = 0, flag = 0, k_0 = 0):
#     """Choice of the padding method.

#     Args:
#         FileLoc (str): File location.
#         method (str): method of padding.
#         ratio (float): ratio of cardinality constraint. 
#         ExtraLoc (str): Extra file location, can be `small` or `synthetic`.
#         flag (int): if flag = 0, then apply the limited information.
#     """
#     df = pd.read_pickle(FileLoc)
#     df_out = pd.DataFrame()
   
#     if flag != 0:
#         if k_0 == 0:
#             print("currently loading x_GPS: " + "../select_nodes/" +Extra + "/"+method+str(ratio)+".pkl") 
#             df_x_GSP = pd.read_pickle("../select_nodes/"+Extra + "/"+method+str(ratio)+".pkl")
#         else:
#             print("currently loading x_GPS: " + "../select_nodes/k/" +Extra + "/"+method+str(k_0)+".pkl") 
#             df_x_GSP = pd.read_pickle("../select_nodes/k/"+Extra + "/"+method+str(k_0)+".pkl")            
    
#     for _, ts in df.iterrows():
#         name = ts['name']
#         tag = ts['tag']
#         print("Start running on: ", name, flush=True)
#         print("Start on method: ", method)
        
#         out_dict = {"name":name, "tag": tag}
        
#         if flag != 0:
#             ts_x_GSP = df_x_GSP.loc[df_x_GSP['name'] == name] # load x_GSP
        
#         n = int(ts['n'])
#         s = ts['s']
#         if k_0 == 0:
#             k = int(n * ratio) # cardinality constraint
#         else:
#             k = k_0

#         for mat in ['pol', 'dis']:
#             print("Start on matrix: ", mat)
            
#             Score, Time = 0, 0
            
#             P = ts[mat+'Mat']
            
#             if flag != 0:
#                 x_GSP = ts_x_GSP[mat].values[0]
                
#                 [Score, Time, std] = func_topology_changes(P, s, k, x_GSP) # if the input x is single dimensional
#                 out_dict[mat+"Std"] = std
#             else:
#                 if method == "AdaptiveGreedy":
#                     [Score, Time] = func_AdaptiveGreedy(P, n, s, k)
#                 if method == "NonAdaptiveGreedy":
#                     [Score, Time] = func_NonAdaptiveGreedy(P, n, s, k)
            
#             out_dict[mat+"Score"] = Score
#             out_dict[mat+"Time"] = Time
            
#         print(out_dict)
#         df_out = df_out.append(out_dict, ignore_index=True)
    
#     if k_0 == 0:
#         if flag != 0:
#             df_out.to_pickle("out/"+Extra + "/"+method + str(flag) +str(ratio)+".pkl")
#         else:
#             df_out.to_pickle("out/"+Extra + "/"+method +str(ratio)+".pkl")
#     else:
#         if flag != 0:
#             df_out.to_pickle("out/k/"+Extra + "/"+method + str(flag) +str(k_0)+".pkl")
#         else:
#             df_out.to_pickle("out/k/"+Extra + "/"+method +str(k_0)+".pkl")    
