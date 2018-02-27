import networkx as nx


def remove_edges(g, edge_inx):
    newg = g.copy()
    for inx in edge_inx:
        edge = g.edges()[inx]
        newg.remove_edge(edge[0], edge[1])

    return newg

def compute_score(g, num_of_nodes):

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