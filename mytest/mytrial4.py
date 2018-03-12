import scipy.io as sio
import networkx as nx
from scipy.sparse import *

oldg = nx.read_gml("../datasets/dblp.gml")

matfile = sio.loadmat("../mat_files/dblp.mat")

count = 0

M = csr_matrix(matfile['group'])


clusters = {}
for i, j in zip(M.tocoo().row, M.tocoo().col):
    clusters.update({str(i): int(j)})

newg = oldg.copy()
nx.set_node_attributes(newg, name='clusters', values=clusters)
nx.write_gml(newg, "../datasets/cdblp.gml")


print(clusters)

