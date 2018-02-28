import networkx as nx
import scipy.sparse as scio
import numpy as np

def remove_edges(g, edge_inx):
    newg = g.copy()
    for inx in edge_inx:
        edge = g.edges()[inx]
        newg.remove_edge(edge[0], edge[1])

    return newg

def compute_scores(graph, method, edges):

    n = graph.number_of_nodes()
    score_matrix = scio.lil_matrix(np.zeros(shape=(n, n), dtype=np.float))
    if method == "CommonNeighbors":
        u=1
        v=1
        u_neigh = set(nx.neighbors(graph, u))
        v_neigh = set(nx.neighbors(graph, v))

        score = len(u_neigh.intersection(v_neigh))

    if method == "Jaccard":
        for u, v, coeff in nx.jaccard_coefficient(graph, edges):
            score_matrix[int(u), int(v)] = coeff
            score_matrix[int(v), int(u)] = coeff

        """
        for edge in edges:
            u, v = edge
            u_neigh = set(nx.neighbors(graph, u))
            v_neigh = set(nx.neighbors(graph, v))

            s = float(len(u_neigh.intersection(v_neigh))) / float(len(u_neigh.union(v_neigh)))

            score_matrix[u, v] = s
            score_matrix[v, u] = s
        
        """

    if method == "PrefAttachmenet":
        u_size = len((nx.neighbors(graph, u)))
        v_size = len((nx.neighbors(graph, v)))

        score = u_size * v_size

    return score_matrix