"""Import public modules."""
import time
import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('..', 'algorithm')))
from MaxCutGSP import *

def func_SDP(P, n, k, M):
    """SDP function. Repeat M times and take an average.
    Args:
        P (float[n,n]): PolMat, DisMat, or IntMat. 
        n (int): number of nodes. 
        k (int): cardinality constraint. 
        M: Repeat M times
    
    Returns:
        SDPScore (float): SDP score.
        SDPTime (float): SDP time.
        x_GSP (float[n]): GSP solution.
    """
    #pars
    
    SDPScoreList = np.zeros(M)
    x_GSP_list = np.zeros(shape=(M, n))
    
    StartMosekSDPTime = time.time()
    X_opt = mosek_scp_k(P, n, k, round_bisection=True)
    MosekSDPTime = time.time() - StartMosekSDPTime
    
    StartRMTime = time.time()
    for i in range(M):
        SDPScoreList[i], x_GSP_list[i] = simple_rounding(X_opt, n, k, P, 100)
    
    RoundingMovingTime = time.time() - StartRMTime

    MovingScore = np.mean(SDPScoreList)
    FinalTime = MosekSDPTime + RoundingMovingTime/M

    return [MovingScore, FinalTime, x_GSP_list]

def func_NonAdaptiveGreedy(P, n, s, k):
    """Non adaptive greedy, note constructing the greedy constrants costs most of the time. 

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
    SpScore, x_GSP  = NonAdaptiveGreedy(k, n, P_greedy, c)
    SpTime = time.time() - StartSpGreedyTime
    
    print("SpScore: " + str(SpScore), flush=True)
        
    return [SpScore, SpTime, x_GSP] 
        
def func_AdaptiveGreedy(P, n, s, k):
    """Adaptive greedy, note constructing the greedy constants costs most of the time  

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
    SpScore, x_GSP  = AdaptiveGreedy(k, n, P_greedy, c)
    SpTime = time.time() - StartSpGreedyTime
    
    print("SpScore: " + str(SpScore), flush=True)
        
    return [SpScore, SpTime, x_GSP]        

def func_HighDegree(P, n, s, d, k):
    """Choose the highest degree nodes, and select the sign according to the current best choice.  

    Args:
        P (float[n, n]): PolMat, DisMat, or IntMat.
        n (int): number of nodes.
        s (float[n]): vector of innate opinions
        d (int [n]): vector of degrees
        k (int): cardinality constraint
        
    Returns:
        DegreeScore (float): Mean score.
        DegreeTime (float): Mean time.
    """
    

    StartTime = time.time()
    DegreeScore, x_GSP = HighDegreeGSP(P, s, n, d, k)
    DegreeTime = time.time() - StartTime
    
    return [DegreeScore, DegreeTime, x_GSP] 

def func_Random(P, n, s, k, M):
    """Randomly select k nodes. Repeat M(10) times and take an average

    Args:
        P (float[n, n]): PolMat, DisMat, or IntMat.
        n (int): number of nodes.
        s (float[n]): vector of innate opinions
        k (int): cardinality constraint
        M: repeat M times
        
    Returns:
        DegreeScore (float): Mean score.
        DegreeTime (float): Mean time.
    """
    Scorelist = np.zeros(M)
    x_GSP_list = np.zeros(shape=(M,n))
    
    StartTime = time.time()
        
    for i in range(M):
        Scorelist[i], x_GSP_list[i] = RandomGSP(P, s, n, k)
    #ss = change_s_largex(s, max_x_list, n) # not sure if it will be useful, for future check
    #print(str(np.dot(np.dot(ss, L), ss)))
    RandomScore = np.mean(Scorelist)
    RandomTime = (time.time() - StartTime) / M
        
    return [RandomScore, RandomTime, x_GSP_list] 


# def choice_pad(FileLoc, method, Extra, ratio = 0, k_0 = 0):
#     """According to the choice of method, and choice of ratio or k_0, use different algorithms, and write to different files. 
    
#     For SDP and Random, get 10 sets of seed nodes. 
#     For IM, do not store x_gsp into files.

#     Args:
#         FileLoc (str): File location.
#         method (str): choose from "SDP", "RtGreedy", "SpGreedy", "HighDegree", "Random", "IM"
#         ratio (float): ratio of n, if ratio ==0, set k = k_0 
#         k_0 (int): if ratio == 0, set k = k_0
#     """
#     M = 5
    
#     df = pd.read_pickle(FileLoc)
#     df_out = pd.DataFrame()
#     df_GSP = pd.DataFrame() # especially save GSP solutions. 
    
#     for name in df['name']:
#         print("Start running on: ", name)
#         print("Start on method: ", method)
        
#         ts = df.loc[df['name'] == name]
#         n = int(ts['n'].values[0])
#         s = np.zeros(n) # innate opinions with 0
        
#         if ratio != 0:
#             k = int(ratio*n) # cardinality constraint, set k according to ratio
#         else:
#             k = k_0 # set the ratio according to the ratio, set k according to k_0 
#         out_dict = {"name":name, "polScore": 0, "disScore": 0, "polTime": 0, "disTime": 0}
        
#         if (method == "SDP") or (method == "Random"):
#             x_GSP_dict = {"name":name, "pol":np.zeros(shape=(M, n)), "dis":np.zeros(shape=(M, n))} # only for Random and SDP
#         else:
#             x_GSP_dict = {"name":name, "pol": np.zeros(n), "dis": np.zeros(n)}

#         for mat in ['pol', 'dis']:
#             print("Start on matrix: ", mat)
            
#             Score, Time = 0, 0
#             P = ts[mat+'Mat'].values[0]

#             if method == "NonAdaptiveGreedy":
#                 [Score, Time, x_GSP] = func_NonAdaptiveGreedy(P, n, s, k)            
#             if method == "AdaptiveGreedy":
#                 [Score, Time, x_GSP] = func_AdaptiveGreedy(P, n, s, k)
#             if method == "HighDegree":
#                 [Score, Time, x_GSP] = func_HighDegree(P, n, s, ts['d'].values[0], k)
#             if method == "Random":
#                 [Score, Time, x_GSP_list] = func_Random(P, n, s, k, M)
#             if method == "SDP":
#                 [Score, Time, x_GSP_list] = func_SDP(P, n, k, M)
#             if method == "IM":
#                 df_x_GSP = pd.read_pickle("../select_nodes/"+Extra + "/LocalIM"+str(ratio)+".pkl")
#                 ts_x_GSP = df_x_GSP.loc[df_x_GSP['name'] == name]
#                 x_GSP = ts_x_GSP[mat].values[0]
#                 Score = np.dot(np.dot(x_GSP, P), x_GSP)
            
#             out_dict[mat+"Score"] = Score
#             out_dict[mat+"Time"] = Time
            
#             if (method == "SDP") or (method == "Random"):
#                 x_GSP_dict[mat] = x_GSP_list
#             else:
#                 x_GSP_dict[mat] = x_GSP
            
#         print(out_dict, flush=True)
        
#         df_out = df_out.append(out_dict, ignore_index=True)
#         df_GSP = df_GSP.append(x_GSP_dict, ignore_index=True)
    
#     if k_0 == 0:
#         # we choose cut according to ratio    
#         df_out.to_pickle("out/"+ Extra + "/" +method+str(ratio)+".pkl")
        
#         if method != "IM":
#             df_GSP.to_pickle("../select_nodes/" + Extra + "/" + "Local"+method+str(ratio)+".pkl")
#     else:
#         # we choose the cut according to k_0
#         df_out.to_pickle("out/k/" + Extra + "/" + method + str(k_0) + ".pkl")
#         if method != "IM":
#             df_GSP.to_pickle("../select_nodes/k/" + Extra + "/" + "Local"+method+str(k_0)+".pkl")
