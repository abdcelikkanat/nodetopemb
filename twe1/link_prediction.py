import numpy
import networkx as nx
import random as rand
import numpy as np



def remove_edges(graph, method="Random", params={}):
    newg = nx.Graph()
    newg.add_nodes_from(graph.nodes())

    removedEdges = []

    if method == "Random":
        # Remove each edge with probability p
        p = params["p"]
        if p is None:
            p = 0.2

        for edge in graph.edges():
            if rand.random() > p:
                removedEdges.append(edge)
            else:
                newg.add_edge(edge[0], edge[1])

    if method == "Exact":
        newg = graph.copy()

        size = params['size']
        chosen_edges = np.random.choice([edge_inx for edge_inx in range(graph.number_of_edges())], replace=False, size=size)

        removedEdges = [list(graph.edges())[edge_inx] for edge_inx in chosen_edges]

        newg.remove_edges_from(removedEdges)


    if method == "GivenProb":
        probs = params['probs']

        newg = graph.copy()
        chosen_edges = np.random.choice([edge_inx for edge_inx in range(graph.number_of_edges())], p=probs, replace=False, size=size)

        removedEdges = [list(graph.edges())[edge_inx] for edge_inx in chosen_edges]

        newg.remove_edges_from(removedEdges)

    return newg, removedEdges


def compute_score(graph, u, v, method):
    if method == "CommonNeighbors":
        u_neigh = set(nx.neighbors(graph, u))
        v_neigh = set(nx.neighbors(graph, v))

        score = len(u_neigh.intersection(v_neigh))

    if method == "Jacard":
        u_neigh = set(nx.neighbors(graph, u))
        v_neigh = set(nx.neighbors(graph, v))

        score = float(len(u_neigh.intersection(v_neigh))) / float(len(u_neigh.union(v_neigh)))

    if method == "PrefAttachmenet":
        u_size = len((nx.neighbors(graph, u)))
        v_size = len((nx.neighbors(graph, v)))

        score = u_size * v_size

    return score


def predict(graph, source, metric):
    scores = {nb:compute_score(graph, u=source, v=nb, method=metric) for nb in nx.neighbors(graph, source)}
    max_node = max(scores, key=scores.get)
    return max_node

dataset_name = "karate"

edges0 = [[0,1], [0,2], [1,2],[1,3],[2,3]]
edges1 = [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[4,5],[3,5]]

"""
g = nx.Graph()
g.add_edges_from(edges0)
"""

g = nx.read_gml("../datasets/"+dataset_name+".gml")


"""
s = score(graph=g, u=0, v=1, method="Jacard")
print(s)
"""

removedGraph, removedEdges = remove_edges(graph=g, method="Exact", params={'size': 10})

count = 10
scores = [[0.0 for _ in nx.neighbors(removedGraph, node)] for node in removedGraph.nodes()]

score_matrix = np.zeros(shape=(removedGraph.number_of_nodes(), removedGraph.number_of_nodes()), dtype=np.float)

for i in range(removedGraph.number_of_nodes()):
    for j in range(i+1, removedGraph.number_of_nodes()):
        score_matrix[i, j] = compute_score(removedGraph, u=str(i), v=str(j), method="Jacard")
        #score_matrix[j, i] = score_matrix[i, j]




k = 10
predictions = []
max_indices = np.argsort(score_matrix, axis=None)[::-1][:k]

n = removedGraph.number_of_nodes()
for inx in max_indices:
    row = np.int(inx/np.float(n))
    col = inx - row*n
    predictions.append([row, col])


print(g.number_of_edges())
print(removedGraph.number_of_edges())

print(predictions)
print(removedEdges)