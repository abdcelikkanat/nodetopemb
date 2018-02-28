import numpy as np
import scipy.sparse as scio
import networkx as nx
from split2train_test import *
from utils import *
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time


#dataset_name = "astro-ph"
dataset_name = "facebook"

edges0 = [[0,1], [0,2], [1,2],[1,3],[2,3]]
edges1 = [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[4,5],[3,5]]


gmlfile = "../../datasets/{}.gml".format(dataset_name)

g = nx.read_gml(gmlfile)
print("Number of nodes: {}\nNumber of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

size = 44000

start_time = time.time()
res_graph, tp_edges, tn_edges = split(g=g, method="Exact", params={'pos_sample_size': size, 'neg_sample_size': size})

print("Scores are being computed")
edges = [] + tp_edges + tn_edges
print(len(edges))
score_matrix = compute_scores(graph=res_graph, method="Jaccard", edges=edges).todense()

y_true = []
y_score = []
for edge in tp_edges:
    y_true.append(1)
    y_score.append(score_matrix[int(edge[0]), int(edge[1])])

for edge in tn_edges:
    y_true.append(0)
    y_score.append(score_matrix[int(edge[0]), int(edge[1])])

print("Total elapsed time {} (in minutes)".format((time.time()-start_time)/60.0))

auc = roc_auc_score(y_true=y_true, y_score=y_score)
print(auc)


fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
#print(fpr)
plt.figure()
plt.plot(fpr, tpr, lw=2, color='darkorange', label='ROC curve ')
plt.show()