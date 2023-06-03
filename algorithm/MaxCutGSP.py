"""Import high level packages."""
import numpy as np
import os
from mosek.fusion import *
import copy


def sphere_uniform(n: int):
    vec = np.random.normal(size=n)
    norm = np.sqrt(np.sum(vec ** 2))
    return vec / norm

#set a timer, should not run tooo long
def mosek_scp_k(P, n, k, round_bisection=False):
    """
    Use Mosek to solve the quadratic function without constraints. 
    """
    
    with Model("Max Cut with Given Size of Parts") as M:
        X = M.variable("X", Domain.inPSDCone(n))
        x = X.diag()
        # constraints
        
        if round_bisection == False:
            M.constraint("C1", Expr.sum(X), Domain.equalsTo((n - 2*k)**2))
        else:
            M.constraint("C1", Expr.sum(X), Domain.inRange(0.0, 1.0)) # should be bisection but with error bound considered
        M.constraint("C2", x, Domain.equalsTo(1.))

        #objective 
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(P, X))

        # solve
        M.solve()
        
        #Get the solution
        #obj = M.primalObjValue() / 4.0 

        sol = X.level().reshape(n, n)
    
    return sol

    
def simple_rounding(X, n, k, P, I):
    """
    An intuitive rounding technique. Cutting plane. 

    Loop, I times
    
    returns obj, and x_opt
    """

    r = 0
    obj = 0
    lower_triangle = np.linalg.cholesky(X) # obtain the lower triangle 
    
    for _ in range(I):
        r = sphere_uniform(n)
        x = np.sign(np.dot(lower_triangle, r))
        
        o, x = moving_k(x, P, k, n)
        if o > obj:
            obj = o
            x_opt = x
            
    return 1/4*obj, np.array([1 if i == 1 else 0 for i in x_opt])

def cal(P, idx, x):
    """Calculate the gain if changing x[idx] to x[idx]*(-1), we confine x[idx] = -1
        Args:
        P (float[n][n]): n*n matrix, in formate np.array 
        idx (int): index to change
        x (int[n]):a vector of -1 and 1
        
    """

    gain = 4*np.dot(P[idx, :], x) + 4*P[idx, idx]
    #print("gain: %f" % gain)
    
    return gain 


def moving_k(x_init, P, k, n):
    """Moving vertices from oneside to another side, until |S| = k, and |U| = n-k.

    Args:
        x (int[n]): A vector of -1 and 1
        P (float[n][n]): The matrices, PolMat, DisMat, IntMat
        k (int): Parameter, the vector divides to -1 and 1 with size (k, n-k), the input ensures that k < n/2 
        n (int): The size of the vector x
    
    Returns:
        opt (float): The value of the objective function
        x (int[n]): A vector of -1 and 1
    """
    
    x = copy.deepcopy(x_init)
    
    pool = [-1, 1]
    u, u_size_init = pool[np.argmin([len(x[x==i]) for i in pool])], np.min([len(x[x==i]) for i in pool]) # looking for the smaller part
    
    u_size = u_size_init 
    #print("u_size: %d" % u_size)
    #print("u is ", u)
    
    if u_size_init < k: # if the smaller part is smaller than k, then we need to add more vertices
        if u == -1:
            x = x * (-1) 
        
        T = [i for i in range(n) if x[i] == -1]
        
        while n - u_size > n - k: # move vertices from T to U until |U| = n-k
            maxidx = np.argmax(np.array([cal(P, i, x) for i in T]))
            x[T[maxidx]] = 1
            T.remove(T[maxidx])
            
            u_size += 1
        
        if len(x[x==1]) != k:
            print("type 1 error")
            print(len(x[x==1]), k, n)
            print("there is a bug in previous step of getting (k, n-k)")
            exit()
        
        
    elif u_size_init > k: # if the smaller part is larger than k, then we need to remove some vertices
        if u == 1:
            x = x * (-1)
        
        T = [i for i in range(n) if x[i] == -1]
        
        while u_size > k:
            maxidx = np.argmax(np.array([cal(P, i, x) for i in T]))
            x[T[maxidx]] = 1
            T.remove(T[maxidx])
            
            u_size -= 1 
        
        if len(x[x==-1]) != k:
            print("type 2 error")
            print(len(x[x==-1]), k, n)
            print("there is a bug in previous step of getting (k, n-k)")
            exit()
    
    return np.dot(np.dot(x, P), x), x
    
def construct_greedy_constants(s, L):
    """
    switch s to 1
    """
    epsilon_diag = np.diag(1 - s)
    
    part_1 = np.dot(np.dot(epsilon_diag, L), epsilon_diag) 
    part_2 = 2*np.dot(np.diag(np.dot(s, L)), epsilon_diag)

    #print(part_2)
    
    P = part_1 + part_2 

    c = np.dot(np.dot(s, L), s)

    return P, c

def NonAdaptiveGreedy(k, n, P, c):
    """
    Absolute Greedy algorithm
    """
    max_diversity = 0

    init_list = np.array([P[it, it]  for it in range(n)])
    ind = init_list.argsort()[::-1]
    x = np.zeros(n)

    T, U = [], []
    count = 0

    for i in ind:
        if P[i, i] + 2*sum([P[i, jt] for jt in T])> 0: # the relative gain > 0 
            x[i] = 1
            T.append(i)
            count += 1
            if count >= k -0.01: #IN CASE OF ROUNDING
                break
        else:
            U.append(i)
    
    if count < k -0.01: # add up the missed nodes
        print(k - count, ' nodes are missed') # checked how many nodes the nonadaptive greedy algorithm picks randomly 
        for i in U[:k-count]:
            x[i] = 1

    max_diversity = np.dot(np.dot(x, P), x) + c
    
    return max_diversity, np.copy(x)

    
def AdaptiveGreedy(k, n, P, c):
    """
    Simple Greedy by Chen and Racz, modified, removed condition to gain at each time step, and ensure k nodes are changed. 
    """
    max_diversity = 0

    init_list = np.array([P[it, it] for it in range(n)])
    i = init_list.argsort()[::-1][0]

    check_list = [j for j in range(n)]
    check_list.remove(i)
    T = [i]
    while len(T) < k:
        density_list = [P[it, it] + 2*sum([P[it, jt] for jt in T]) for it in check_list]
        d_id = np.argmax(np.array(density_list))  #density id

        idd = check_list[d_id]
        T.append(idd)
        check_list.remove(idd)
        
    x = np.zeros(n)
    x[np.array(T)] = 1

    #print("len(T) is ", len(T), flush=True)
        
    max_diversity = np.dot(np.dot(x, P), x) + c
            
    print("max_diversity: %f" % max_diversity)
    
    return max_diversity, x

def HighDegreeGSP(P, s, n, d, k):
    
    vertices = np.array(range(n))
    arr_s = np.array(s)
    
    T = []
    
    while len(T) < k:
        v = vertices[np.argmax(d[vertices])] # choose the vertex with the highest degree
        vertices = np.delete(vertices, np.where(vertices == v)) # delete v from vertices
        T.append(v)
        
        arr_s[v] = 1
    
    #print("selected degrees are: ", d[T], flush=True)
        
    return np.dot(np.dot(arr_s, P), arr_s), arr_s

def RandomGSP(P, s, n, k):
    random_data = os.urandom(4)
    np.random.seed(int.from_bytes(random_data, byteorder="big"))
    
    vertices = np.array(range(n))
    select_nodes = np.random.choice(vertices, k, replace=False) # choose k vertices randomly
    
    arr_s = np.array(s)
    arr_x = np.zeros(n)
    
    arr_s[select_nodes] = 1 # assign new signs to selected vertices 
    arr_x[select_nodes] = 1
        
    return np.dot(np.dot(arr_s, P), arr_s), arr_x
    
    