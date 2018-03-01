import numpy as np
import scipy.sparse as scio
import networkx as nx
from utils import *
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time

#dataset_name = "karate"
#dataset_name = "astro-ph"
#dataset_name = "facebook"
dataset_name = "citeseer"

gmlfile = "../../datasets/{}.gml".format(dataset_name)
g = nx.read_gml(gmlfile)

print("Number of nodes: {}\nNumber of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

# Consider the half of the edges
#size = np.floor(g.number_of_edges()/2)
size = 2000

print("Positive and negative samples are being generated")
start_time = time.time()
train_graph, tp_edges, tn_edges = split(g=g, method="Exact",
                                        params={'pos_sample_size': size, 'neg_sample_size': size})






print("Scores are being computed for Jaccard")
edges = [] + tp_edges + tn_edges
score_matrix = compute_scores(graph=train_graph, embeddings=None, method="Jaccard", edges=edges)

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
plt.figure(1)
plt.plot(fpr, tpr, lw=2, color='darkorange', label='ROC curve ')





embedding_file_path = "../output/citeseer/citeseer_word.embedding"
number_of_nodes, embedding_size, embeddings = read_embedding_file(embedding_file_path)

print("Scores are being computed for Hadamard")
edges = [] + tp_edges + tn_edges
score_matrix = compute_scores(graph=None, embeddings=embeddings, method="Hadamard", edges=edges)

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
plt.figure(2)
plt.plot(fpr, tpr, lw=2, color='darkorange', label='ROC curve ')
plt.show()




"""

embedding_file_path = "../output/citeseer/citeseer_combined.embedding"
number_of_nodes, embedding_size, embeddings = read_embedding_file(embedding_file_path)

print("Scores are being computed for Hadamard")
edges = [] + tp_edges + tn_edges
score_matrix = compute_scores(graph=None, embeddings=embeddings, method="Hadamard", edges=edges)

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
plt.figure(2)
plt.plot(fpr, tpr, lw=2, color='darkorange', label='ROC curve ')
plt.show()


"""