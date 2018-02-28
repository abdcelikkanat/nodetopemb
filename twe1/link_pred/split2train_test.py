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
        pos_sample_size = params['pos_sample_size']
        neg_sample_size = params['neg_sample_size']

        # Copy the original graph
        residual_g = g.copy()


        num_of_edges = g.number_of_edges()
        # The initial number of connected components in the graph
        num_of_cc = nx.number_connected_components(g)
        edges_of_graph = list(g.edges())


        test_pos_edge_samples = []
        test_neg_edge_samples = []

        chosen_pos_sample_count = 0

        frozen_count = 0
        # Select the true positive samples
        while chosen_pos_sample_count < pos_sample_size:

            candidate_edge_inx = np.random.choice(a=range(residual_g.number_of_edges()), size=1)[0]

            candidate_edge = list(residual_g.edges())[candidate_edge_inx]
            # If the edge is loop, ignore it
            if candidate_edge[0] == candidate_edge[1]:
                continue

            residual_g.remove_edge(candidate_edge[0], candidate_edge[1])
            # Keep the number of connected components
            if nx.number_connected_components(residual_g) == num_of_cc:
                test_pos_edge_samples.append(candidate_edge)
                chosen_pos_sample_count += 1
                frozen_count = 0
            else:
                residual_g.add_edge(candidate_edge[0], candidate_edge[1])
                frozen_count += 1

            assert frozen_count != 10000, "Could not find a suitable edge in {} trials".format(10000)

        chosen_neg_sample_count = 0
        # Choose true negative samples
        while chosen_neg_sample_count < neg_sample_size:
            u, v = np.random.choice(range(n), size=2, replace=True)

            candidate_edge = (str(u), str(v))

            if candidate_edge not in g.edges():
                test_neg_edge_samples.append(candidate_edge)
                chosen_neg_sample_count += 1

    return residual_g, test_pos_edge_samples, test_neg_edge_samples