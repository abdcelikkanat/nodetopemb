import networkx as nx
import numpy as np
import random as rand


def split(g, method, params):
    n = g.number_of_nodes()
    train_edges_inx = []
    test_edges_inx = []

    if method == "Random":
        # Remove each edge with probability p
        p = params["p"]
        if p is None:
            p = 0.2

        for i in range(g.number_of_edges()):
            if rand.random() < p:
                test_edges_inx.append(i)
            else:
                train_edges_inx.append(i)

    if method == "Exact":
        size = params['size']

        test_edges_inx = np.random.choice(range(n), replace=False, size=size)
        train_edges_inx = [inx for inx in range(n) if inx not in test_edges_inx]

    return train_edges_inx, test_edges_inx