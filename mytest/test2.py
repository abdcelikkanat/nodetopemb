from nodetopemb import *
import networkx as nx
import numpy as np


#g = nx.read_gml("../datasets/citeseer.gml")

edges = [[0,1], [1,2], [0,2], [2,3], [3,4], [2,4], [5,4], [3,5]]
myg = Graph()
myg.add_edges_from(edge_list=edges)
sp = myg.count_triangles_on_edges()

doc = myg.graph2doc(number_of_paths=1, path_length=8, method="TriWalk", params={"alpha":0.0})

for line in doc:
    print(line)

g = nx.read_gml("../datasets/citeseer.gml")




