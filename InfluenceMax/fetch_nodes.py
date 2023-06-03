import sys
import numpy as np
import pandas as pd


filename = sys.argv[1]
datasetname = sys.argv[2]
name = sys.argv[3]
ratio = float(sys.argv[4])


file_loc = '.out/' + filename
init_data = pd.read_pickle("../data/" + datasetname)
init_data = init_data.filter(['name', 'n'])

df_GSP = pd.DataFrame()
f = pd.read_feather(file_loc)
n = int(init_data['n'].values[0])

print(name)

x_GSP_dict = {"name":name, "pol":np.zeros(n), "dis":np.zeros(n)}

k = ratio*n
print(k)

# get x_gsp
select_nodes = f["seeds"][:int(k)]
x_gsp = np.zeros(n)
x_gsp[select_nodes] = 1

# write to dict, since these three are the same
for mat in ['pol', 'dis']:
    x_GSP_dict[mat] = x_gsp

df_GSP = df_GSP.append(x_GSP_dict, ignore_index = True)
df_GSP.to_pickle("../select_nodes/" + name + "/LocalIM" + str(ratio) + ".pkl")