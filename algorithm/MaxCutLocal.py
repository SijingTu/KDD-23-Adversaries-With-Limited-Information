"""Import high level packages."""
import numpy as np
import copy

def positive_changes(x_init, s_init, P, k):
    """Perform one changes after GSP. Directly change the opinions to 1. 
    % 2022-08-01, original boost. 

    Args:
        x_init (float[n]): solution of GSP. 
        s (float[n]): innate opinions. 
        P (float[n, n]): PolMat, DisMat, or IntMat.
        k (int): the inital constraint of GSP (k, n-k)

    Returns:
        score (float): the score of the solution.
        x float[n]: the solution of this heuristic. 
    """

    if (x_init.ndim != 1) and (x_init.ndim !=2):
        exit("ERROR: X_gsp's dimension.")
    
    if x_init.ndim == 1:
        x = copy.deepcopy(x_init)
        #print("x is", x)
        s = copy.deepcopy(s_init)
        pool = [0, 1]
        u = pool[np.argmin([len(x[x == i]) for i in pool])] # u is the least frequent element in x
        u_size = len(x[x == u])  # u_size is the number of u in x 
        
        if u_size != k:
            print(u_size, k, len(x))
            print("there is a bug in previous step of getting (k, n-k)")
            exit()
        
        T = np.array([i for i in range(len(x)) if x[i] == u])
        
        for i in T:
            s[i] = 1
        
        #print(s)      
                
        score = np.dot(np.dot(s, P), s)
        return score, s, 0 
    else:
        # if a list of x_gsp is given.
        x_list = copy.deepcopy(x_init)
        M = np.size(x_list, 0) # # of x_gsp
        score_list = np.zeros(M)
        
        for i in range(M):
            x = x_init[i]
            # print("x is", x)
            s = copy.deepcopy(s_init)
            pool = [0, 1]
            u = pool[np.argmin([len(x[x == t]) for t in pool])] # u is the least frequent element in x
            u_size = len(x[x == u])  # u_size is the number of u in x 
            
            if u_size != k:
                print(u_size, k, len(x))
                print("there is a bug in previous step of getting (k, n-k)")
                exit()
            
            T = np.array([t for t in range(len(x)) if x[t] == u])
            
            for t in T:
                s[t] = 1
            
            #print(s)      
                    
            score = np.dot(np.dot(s, P), s)
            score_list[i] = score
            
        return np.mean(score_list), s, np.std(score_list) 
            

