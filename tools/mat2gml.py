import networkx as nx
import numpy as np
import scipy.io as sio

dataset_name = "dblp"
mat_dict = sio.loadmat("../mat_files/"+dataset_name+".mat")
print(mat_dict['network'].shape)

n = mat_dict['network'].shape[1]
c = mat_dict['group'].shape[1]

g = nx.Graph()
networx = mat_dict['network']
cx = networx.tocoo()

for i in range(len(cx.row)):
    if cx.data[i]:
        g.add_edge(str(cx.row[i]), str(cx.col[i]))

print(g.number_of_nodes())
print(g.number_of_edges())

nx.write_gml(g, "../datasets/"+dataset_name+".gml")



