import networkx as nx
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


g = nx.read_gml("../datasets/citeseer.gml")
N = g.number_of_nodes()




def common_vertices(u, v):
    uset = set(nx.neighbors(g, u))
    vset = set(nx.neighbors(g, v))

    for uu in nx.neighbors(g, u):
        uset.union(set(nx.neighbors(g, uu)))

    for vv in nx.neighbors(g, v):
        vset.union(set(nx.neighbors(g, vv)))

    inter = uset.intersection(vset)

    return len(inter)


# Pre
weights = {node: int(node) for node in g.nodes()}
for node in g.nodes():
    w = [common_vertices(u=node, v=nb) for nb in nx.neighbors(g, node)]
    max_inxs = [i for i in range(len(w)) if w[i] == max(w)]

    max_inx = max_inxs[0]
    if len(max_inxs) > 1:
        nb_list = list(nx.neighbors(g, node))
        for i in max_inxs:
            if nx.degree(g, nb_list[i]) > nx.degree(g, nb_list[max_inx]):
                max_inx = i

    weights[node] = list(nx.neighbors(g, node))[max_inx]



labels = {node: -1 for node in g.nodes()}
label_inx = 0
queue = deque()
labeled_node_count = 0

while labeled_node_count != N:

    current_node = np.random.choice(a=[node for node in g.nodes() if labels[node] < 0], size=1)[0]

    visited_list = []
    temp_node = current_node
    while labels[temp_node] < 0 and temp_node not in visited_list:
        visited_list.append(temp_node)
        temp_node = weights[temp_node]

    if labels[temp_node] >= 0:
        cluster_label = labels[temp_node]
    else:
        cluster_label = label_inx
        label_inx += 1
    for visited_node in visited_list:
        labels[visited_node] = cluster_label
        labeled_node_count += 1

print(weights)
print(labels)
print(label_inx)
print(label_inx - nx.number_connected_components(g))
"""
plt.figure()
pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos,
                       nodelist=[node for node in g.nodes() if labels[node] == 0],
                       node_color='b',
                       node_size=100, alpha=0.8)
nx.draw_networkx_nodes(g, pos,
                       nodelist=[node for node in g.nodes() if labels[node] == 1],
                       node_color='r',
                       node_size=100, alpha=0.8)
nx.draw_networkx_edges(g,pos, width=1.0,alpha=0.5)
plt.show()
"""



number_of_paths = 1
path_len = 512

corpus = []
for node in g.nodes():
    path = []
    for n in range(number_of_paths):

        cluster_label = labels[node]
        current_node = node
        for l in range(path_len):
            path.append(current_node)

            pos_next_nodes = [nb for nb in nx.neighbors(g, current_node) if labels[nb] == cluster_label]

            if len(pos_next_nodes) == 0:
                pos_next_nodes = [current_node]

            current_node = np.random.choice(a=pos_next_nodes, size=1)[0]
        corpus.append(path)

with open("./test_citeseer_walk.corpus", 'w') as f:
    for path in corpus:
        f.write("{}\n".format(" ".join(path)))

import gensim
word_embed_size = 128
window_size = 50
corpus_file = "./test_citeseer_walk.corpus"
word_embed_file = "./test_citeseer_walk.embedding"

sentences = gensim.models.word2vec.Text8Corpus(corpus_file)
w = gensim.models.Word2Vec(sentences, size=word_embed_size, window=window_size,
                            sg=1, hs=1, workers=3,
                            sample=0.001, negative=1000)
w.wv.save_word2vec_format(word_embed_file)



"""
number_of_clusters = 6
clusters = nx.get_node_attributes(g, 'clusters')
true_counts = 0
for node in g.nodes():
    cluster_counts = np.zeros(shape=number_of_clusters, dtype=np.int)
    node_label = labels[node]
    for nb in nx.neighbors(g, node):
        if node_label == labels[nb]:
            cluster_counts[clusters[nb]] += 1

        for nb_nb in nx.neighbors(g, nb):
            if node_label == labels[nb_nb]:
                cluster_counts[clusters[nb_nb]] += 1



    if clusters[node] == np.argmax(cluster_counts):
        true_counts += 1

n = g.number_of_nodes()
acc = float(true_counts)*100.0 / float(n)
print("Number of nodes: {} Prediction accuracy: {}".format(n, acc))
"""