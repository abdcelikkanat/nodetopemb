import networkx as nx
import numpy as np



dataset_name = "cdblp"
if dataset_name == "citeseer":
    number_of_clusters = 6
elif dataset_name == "cblogcatalog":
    number_of_clusters = 39
elif dataset_name == "cdblp":
    number_of_clusters = 4

g = nx.read_gml("../datasets/{}.gml".format(dataset_name))
clusters = nx.get_node_attributes(g, 'clusters')


true_counts = 0
for node in g.nodes():
    cluster_counts = np.zeros(shape=number_of_clusters, dtype=np.int)

    for nb in nx.neighbors(g, node):
        cluster_counts[clusters[nb]] += 1

        for nb_nb in nx.neighbors(g, nb):
            cluster_counts[clusters[nb_nb]] += 1



    if clusters[node] == np.argmax(cluster_counts):
        true_counts += 1

n = g.number_of_nodes()
acc = float(true_counts)*100.0 / float(n)
print("Number of nodes: {} Prediction accuracy: {}".format(n, acc))

