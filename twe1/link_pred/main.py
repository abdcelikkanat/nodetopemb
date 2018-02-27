import numpy as np
import networkx as nx
from split2train_test import *


dataset_name = "karate"

edges0 = [[0,1], [0,2], [1,2],[1,3],[2,3]]
edges1 = [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[4,5],[3,5]]


gmlfile = "../../datasets/{}.gml".format(dataset_name)

g = nx.read_gml(gmlfile)
print(g.number_of_nodes())


train, test = split(g=g, method="Exact", params={'size':20})

