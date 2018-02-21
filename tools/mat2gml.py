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
for i, j, val in zip(cx.row, cx.col, cx.data):
    if val:
        g.add_edge(str(i), str(j))


nx.write_gml(g, "../datasets/"+dataset_name+".gml")



