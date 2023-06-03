"""Import public modules."""

import sys
from gsp import *
# sys.path.append(os.path.abspath(os.path.join('..', 'algorithm')))


def initialize_dataframes():
    """Initializes necessary dataframes."""
    df_out = pd.DataFrame()
    df_GSP = pd.DataFrame() # especially save GSP solutions. 
    return df_out, df_GSP

def calculate_k_by_ratio(n, ratio):
    """Calculates the cardinality constraint k."""
    
    k = int(ratio*n) # set k according to ratio
    return k

def initialize_dict(name, n, method):
    """Initializes dictionaries for output and x_GSP."""
    M = 5
    out_dict = {"name":name, "polScore": 0, "disScore": 0, "polTime": 0, "disTime": 0}
    if (method == "SDP") or (method == "Random"):
        x_GSP_dict = {"name":name, "pol":np.zeros(shape=(M, n)), "dis":np.zeros(shape=(M, n))}
    else:
        x_GSP_dict = {"name":name, "pol": np.zeros(n), "dis": np.zeros(n)}
    return out_dict, x_GSP_dict

def cal_NonAdaptiveGreedy_ratio(FileLoc, ratio=0):
    df = pd.read_pickle(FileLoc)
    df_out, df_GSP = initialize_dataframes()
    
    for name in df['name']:
        print("Start running on: ", name)
        print("Start on method: NonAdaptiveGreedy")
        
        ts = df.loc[df['name'] == name]
        n = int(ts['n'].values[0])
        s = np.zeros(n) # innate opinions with 0
        
        k = calculate_k_by_ratio(n, ratio)
        out_dict, x_GSP_dict = initialize_dict(name, n, 'NonAdaptiveGreedy')
        
        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            Score, Time = 0, 0
            P = ts[mat+'Mat'].values[0]
            [Score, Time, x_GSP] = func_NonAdaptiveGreedy(P, n, s, k)
        
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time
            x_GSP_dict[mat] = x_GSP 
            
        print(out_dict, flush=True)
        
        df_out = df_out.append(out_dict, ignore_index=True)
        df_GSP = df_GSP.append(x_GSP_dict, ignore_index=True)
        
        df_out.to_pickle("out/"+ name + "/NonAdaptiveGreedy"+str(ratio)+".pkl")
        df_GSP.to_pickle("../select_nodes/" + name + "/" + "LocalNonAdaptiveGreedy"+str(ratio)+".pkl")
            
def cal_AdaptiveGreedy_ratio(FileLoc, ratio=0):
    df = pd.read_pickle(FileLoc)
    df_out, df_GSP = initialize_dataframes()
    
    for name in df['name']:
        print("Start running on: ", name)
        print("Start on method: AdaptiveGreedy")
        
        ts = df.loc[df['name'] == name]
        n = int(ts['n'].values[0])
        s = np.zeros(n) # innate opinions with 0
        
        k = calculate_k_by_ratio(n, ratio)
        out_dict, x_GSP_dict = initialize_dict(name, n, 'AdaptiveGreedy')
        
        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            Score, Time = 0, 0
            P = ts[mat+'Mat'].values[0]
            [Score, Time, x_GSP] = func_AdaptiveGreedy(P, n, s, k)
        
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time
            x_GSP_dict[mat] = x_GSP 
            
        print(out_dict, flush=True)
        
        df_out = df_out.append(out_dict, ignore_index=True)
        df_GSP = df_GSP.append(x_GSP_dict, ignore_index=True)
        
        df_out.to_pickle("out/"+ name + "/AdaptiveGreedy"+str(ratio)+".pkl")
        df_GSP.to_pickle("../select_nodes/" + name + "/" + "LocalAdaptiveGreedy"+str(ratio)+".pkl")
        

def cal_HighDegree_ratio(FileLoc, ratio=0):
    df = pd.read_pickle(FileLoc)
    df_out, df_GSP = initialize_dataframes()
    
    for name in df['name']:
        print("Start running on: ", name)
        print("Start on method: HighDegree")
        
        ts = df.loc[df['name'] == name]
        n = int(ts['n'].values[0])
        s = np.zeros(n) # innate opinions with 0
        
        k = calculate_k_by_ratio(n, ratio)
        out_dict, x_GSP_dict = initialize_dict(name, n, 'HighDegree')
        
        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            Score, Time = 0, 0
            P = ts[mat+'Mat'].values[0]
            [Score, Time, x_GSP] = func_HighDegree(P, n, s, ts['d'].values[0],k)
        
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time
            x_GSP_dict[mat] = x_GSP 
            
        print(out_dict, flush=True)
        
        df_out = df_out.append(out_dict, ignore_index=True)
        df_GSP = df_GSP.append(x_GSP_dict, ignore_index=True)
        
        df_out.to_pickle("out/"+ name + "/HighDegree"+str(ratio)+".pkl")
        df_GSP.to_pickle("../select_nodes/" + name + "/" + "LocalHighDegree"+str(ratio)+".pkl")
        
def cal_Random_ratio(FileLoc, ratio=0):
    df = pd.read_pickle(FileLoc)
    df_out, df_GSP = initialize_dataframes()
    M = 5
    
    for name in df['name']:
        print("Start running on: ", name)
        print("Start on method: Random")
        
        ts = df.loc[df['name'] == name]
        n = int(ts['n'].values[0])
        s = np.zeros(n) # innate opinions with 0
        
        k = calculate_k_by_ratio(n, ratio)
        out_dict, x_GSP_dict = initialize_dict(name, n, 'Random')
        
        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            Score, Time = 0, 0
            P = ts[mat+'Mat'].values[0]
            [Score, Time, x_GSP_list] = func_Random(P, n, s, k, M) # get x_GSP_list
        
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time
            x_GSP_dict[mat] = x_GSP_list 
            
        print(out_dict, flush=True)
        
        df_out = df_out.append(out_dict, ignore_index=True)
        df_GSP = df_GSP.append(x_GSP_dict, ignore_index=True)
        
        df_out.to_pickle("out/"+ name + "/Random"+str(ratio)+".pkl")
        df_GSP.to_pickle("../select_nodes/" + name + "/" + "LocalRandom"+str(ratio)+".pkl")
        

def cal_SDP_ratio(FileLoc, ratio=0):
    df = pd.read_pickle(FileLoc)
    df_out, df_GSP = initialize_dataframes()
    M = 5
    
    for name in df['name']:
        print("Start running on: ", name)
        print("Start on method: SDP")
        
        ts = df.loc[df['name'] == name]
        n = int(ts['n'].values[0])
        s = np.zeros(n) # innate opinions with 0
        
        k = calculate_k_by_ratio(n, ratio)
        out_dict, x_GSP_dict = initialize_dict(name, n, 'SDP')
        
        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            Score, Time = 0, 0
            P = ts[mat+'Mat'].values[0]
            [Score, Time, x_GSP_list] = func_SDP(P, n, k, M) # get x_GSP_list
        
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time
            x_GSP_dict[mat] = x_GSP_list 
            
        print(out_dict, flush=True)
        
        df_out = df_out.append(out_dict, ignore_index=True)
        df_GSP = df_GSP.append(x_GSP_dict, ignore_index=True)
        
        df_out.to_pickle("out/"+ name + "/SDP"+str(ratio)+".pkl")
        df_GSP.to_pickle("../select_nodes/" + name + "/" + "LocalSDP"+str(ratio)+".pkl")
     

# the functions
if __name__ == "__main__":
    FileLoc = sys.argv[1]
    ratio = float(sys.argv[2])
    Method = sys.argv[3]

    if Method == "NonAdaptiveGreedy":
        cal_NonAdaptiveGreedy_ratio(FileLoc, ratio)
    if Method == "AdaptiveGreedy":
        cal_AdaptiveGreedy_ratio(FileLoc, ratio)
    if Method == "HighDegree":
        cal_HighDegree_ratio(FileLoc, ratio)
    if Method == "Random":
        cal_Random_ratio(FileLoc, ratio)
    if Method == "SDP":
        cal_SDP_ratio(FileLoc, ratio)

