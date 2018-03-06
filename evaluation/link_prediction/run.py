import numpy as np
import scipy.sparse as scio
import networkx as nx
from utils import *
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import normalize

#dataset_name = "karate"
#dataset_name = "astro-ph"
#dataset_name = "facebook"
dataset_name = "facebook"
suffix = "_temp"



base = "../../twe1/temp_files/output/"
word_embedding_file = base + "/" + dataset_name +"/" + dataset_name + suffix + "_word.embedding"
combined_embedding_file = base + "/" + dataset_name +"/" + dataset_name + suffix + "_combined.embedding"



#embed_score_method = "AdamicAdarIndex"
embed_score_method = "L2"
classic_score_method = "ComNeigh"
suffix += "_{}".format(embed_score_method)

print("Dataset: {} Suffix: {}".format(dataset_name, suffix))

gmlfile = "../../datasets/{}.gml".format(dataset_name)
g = nx.read_gml(gmlfile)


print("Number of nodes: {}\nNumber of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

# Consider the half of the edges
size = np.floor(g.number_of_edges()*0.75)
#size += np.floor(size/5.0)
#size = 5

print("Positive and negative samples are being generated")
start_time = time.time()
train_graph, tp_edges, tn_edges = split(g=g, method="Exact",
                                        params={'pos_sample_size': size, 'neg_sample_size': size})




print("Scores are being computed for "+classic_score_method)
edges = [] + tp_edges + tn_edges

score_matrix = compute_scores(graph=train_graph, embeddings=None, method=classic_score_method, edges=edges)
# Normalize score matrix
score_matrix = normalize(np.asarray(score_matrix.todense()), axis=1, norm='l1')



y_true = []
y_score = []
for edge in tp_edges:
    y_true.append(1)
    y_score.append(score_matrix[int(edge[0]), int(edge[1])])
print("tp edges {}".format(len(tp_edges)))
for edge in tn_edges:
    y_true.append(0)
    y_score.append(score_matrix[int(edge[0]), int(edge[1])])
print("tn edges {}".format(len(tp_edges)))
print("Total elapsed time {} (in minutes)".format((time.time()-start_time)/60.0))

auc = roc_auc_score(y_true=y_true, y_score=y_score)
print(auc)


fpr1, tpr1, _ = roc_curve(y_true=y_true, y_score=y_score)
#print(fpr)
plt.figure(1)
plt.title("Jaccard")
plt.plot(fpr1, tpr1, lw=1, color='darkorange', label='ROC curve ')
plt.savefig("./images/" + suffix + "_"+classic_score_method+".png", bbox_inches="tight")
plt.show()



"""


number_of_nodes, embedding_size, embeddings = read_embedding_file(word_embedding_file)

print("Scores are being computed for "+embed_score_method)
edges = [] + tp_edges + tn_edges
score_matrix = compute_scores(graph=None, embeddings=embeddings, method=embed_score_method, edges=edges)

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


fpr2, tpr2, _ = roc_curve(y_true=y_true, y_score=y_score)
#print(fpr)
plt.figure(2)
plt.title("Word Embedding")
plt.plot(fpr2, tpr2, lw=1, color='darkblue', label='ROC curve ')
plt.savefig("./images/" + suffix + "_word.png", bbox_inches="tight")






number_of_nodes, embedding_size, embeddings = read_embedding_file(combined_embedding_file)

print("Scores are being computed for "+embed_score_method)
edges = [] + tp_edges + tn_edges
score_matrix = compute_scores(graph=None, embeddings=embeddings, method=embed_score_method, edges=edges)

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


fpr3, tpr3, _ = roc_curve(y_true=y_true, y_score=y_score)
#print(fpr)
plt.figure(3)
plt.title("Word+Topic Embedding")
plt.plot(fpr3, tpr3, lw=1, color='darkred', label='Word+Topic')
plt.savefig("./images/" + suffix + "_combined.png", bbox_inches="tight")
plt.show()


plt.figure(4)
plt.title("All")
plt.plot(fpr1, tpr1, lw=1, color='darkorange', label='Jaccard')
plt.plot(fpr2, tpr2, lw=1, color='darkblue', label="Word "+embed_score_method)
plt.plot(fpr3, tpr3, lw=1, color='darkred', label="Word+Topic "+embed_score_method)
plt.legend((classic_score_method, 'Word', 'Word+Topic'), loc=4)
plt.savefig("./images/"+dataset_name + suffix + "_all.png", bbox_inches="tight")
plt.show()


"""