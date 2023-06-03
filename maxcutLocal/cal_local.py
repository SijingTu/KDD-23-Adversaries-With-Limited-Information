"""
Obtain various results
"""

import sys
from local import *
import numpy as np

def read_input(FileLoc):
    """Reads input data from the provided file location."""
    df = pd.read_pickle(FileLoc)
    return df

def load_GSP_limitedinfo(Extra, method, ratio):
    """Loads GSP data if flag is not equal to zero."""
    print("currently loading x_GPS: " + "../select_nodes/" +Extra + "/"+method+str(ratio)+".pkl") 
    df_x_GSP = pd.read_pickle("../select_nodes/"+Extra + "/"+method+str(ratio)+".pkl")         
    return df_x_GSP

def calculate_k(n, ratio):
    """Calculates the cardinality constraint k."""
    k = int(n * ratio) 
    return k

def save_output(df_out, Extra, method, ratio):
    """Saves the output dataframes to the files."""
    
    df_out.to_pickle("out/"+Extra + "/"+method +str(ratio)+".pkl")


def cal_AdaptiveGreedy_ratio(FileLoc, Extra, method, ratio=0):
    
    df = read_input(FileLoc)
    df_out = pd.DataFrame()
    
    print("Start on method: ", method)

    for _, ts in df.iterrows():
        print("Start running on: ", ts['name'], flush=True)
        out_dict = {"name": ts['name']}
        
        n = int(ts['n'])
        s = ts['s']

        k = calculate_k(n, ratio)

        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            P = ts[mat+'Mat']
            
            [Score, Time] = func_AdaptiveGreedy(P, n, np.array(s), k)
            
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time

        print(out_dict)
        df_out = df_out.append(out_dict, ignore_index=True)

    save_output(df_out, Extra, method, ratio)

def cal_NonAdaptiveGreedy_ratio(FileLoc, Extra, method, ratio=0):
    
    df = read_input(FileLoc)
    df_out = pd.DataFrame()
    
    print("Start on method: ", method)

    for _, ts in df.iterrows():
        print("Start running on: ", ts['name'], flush=True)
        out_dict = {"name": ts['name']}
        
        n = int(ts['n'])
        s = ts['s']
        
        # print("s is ", s)

        k = calculate_k(n, ratio)

        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            P = ts[mat+'Mat']
            
            [Score, Time] = func_NonAdaptiveGreedy(P, n, np.array(s), k)
            
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time

        print(out_dict)
        df_out = df_out.append(out_dict, ignore_index=True)

    save_output(df_out, Extra, method, ratio)
    

def cal_LimitedInfo_ratio(FileLoc, Extra, method, ratio=0):
    
    df = read_input(FileLoc)
    df_out = pd.DataFrame()
    df_x_GSP = load_GSP_limitedinfo(Extra, method, ratio)
    
    print("Start on method: ", method)

    for _, ts in df.iterrows():
        name = ts['name']
        print("Start running on: ", name, flush=True)
        out_dict = {"name": name}
        ts_x_GSP = df_x_GSP.loc[df_x_GSP['name'] == name]
        
        n = int(ts['n'])
        s = ts['s']
        k = calculate_k(n, ratio)

        for mat in ['pol', 'dis']:
            print("Start on matrix: ", mat)
            
            P = ts[mat+'Mat']
            
            x_GSP = ts_x_GSP[mat].values[0]
            [Score, Time, std] = func_topology_changes(P, s, k, x_GSP)
            
            out_dict[mat+"Score"] = Score
            out_dict[mat+"Time"] = Time
            out_dict[mat+"Std"] = std 

        print(out_dict)
        df_out = df_out.append(out_dict, ignore_index=True)

    save_output(df_out, Extra, method, ratio)
    
# the functions
if __name__ == "__main__":
    FileLoc = sys.argv[1]
    Extra = sys.argv[2]
    ratio = float(sys.argv[3])
    Method = sys.argv[4]
    
    print(FileLoc, Extra, ratio, Method)

    if Method == "NonAdaptiveGreedy":
        cal_NonAdaptiveGreedy_ratio(FileLoc, Extra, Method, ratio)
    elif Method == "AdaptiveGreedy":
        cal_AdaptiveGreedy_ratio(FileLoc, Extra, Method, ratio)
    else:
        cal_LimitedInfo_ratio(FileLoc, Extra, Method, ratio)