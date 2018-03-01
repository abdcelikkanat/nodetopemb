import networkx as nx
import scipy.sparse as scio
import numpy as np


def compute_scores(graph=None, embeddings=None, method=None, edges=list()):

    if graph is not None:
        n = graph.number_of_nodes()
        score_matrix = scio.lil_matrix(np.zeros(shape=(n, n), dtype=np.float))

        if method == "ComNeigh":
            for edge in edges:
                u, v = edge
                coeff = float(len(list(nx.common_neighbors(graph, u, v))))
                score_matrix[int(u), int(v)] = coeff
                score_matrix[int(v), int(u)] = coeff

        if method == "Jaccard":
            for u, v, coeff in nx.jaccard_coefficient(graph, edges):
                score_matrix[int(u), int(v)] = coeff
                score_matrix[int(v), int(u)] = coeff

        if method == "AdamicAdarIndex":
            for u, v, coeff in nx.adamic_adar_index(graph, edges):
                score_matrix[int(u), int(v)] = coeff
                score_matrix[int(v), int(u)] = coeff

        if method == "PrefAttach":
            for u, v, coeff in nx.preferential_attachment(graph, edges):
                score_matrix[int(u), int(v)] = coeff
                score_matrix[int(v), int(u)] = coeff

    if embeddings is not None:
        n = len(embeddings)

        if method == "Average":
            for edge in edges:
                u, v = int(edge[0]), int(edge[1])
                coeff = np.sum([(x+y)/2.0 for x, y in zip(embeddings[u], embeddings[v])])
                score_matrix[u, v] = coeff
                score_matrix[v, u] = coeff

        if method == "Hadamard":
            for edge in edges:
                u, v = int(edge[0]), int(edge[1])
                coeff = np.sum([x*y for x, y in zip(embeddings[u], embeddings[v])])
                score_matrix[u, v] = coeff
                score_matrix[v, u] = coeff

        if method == "L1":
            for edge in edges:
                u, v = int(edge[0]), int(edge[1])
                coeff = np.sum([np.abs(x-y) for x, y in zip(embeddings[u], embeddings[v])])
                score_matrix[u, v] = coeff
                score_matrix[v, u] = coeff

        if method == "L2":
            for edge in edges:
                u, v = int(edge[0]), int(edge[1])
                coeff = np.sum([np.power(x-y, 2) for x, y in zip(embeddings[u], embeddings[v])])
                score_matrix[u, v] = coeff
                score_matrix[v, u] = coeff


    return score_matrix


def read_embedding_file(file_name):

    # it is assumed that for every node with labels from 0 to n-1, there is a corresponding embedding
    with open(file_name, 'r') as f:
        # The first line is in the form of (the number of nodes, vector size)
        firstline = f.readline()
        n, size = (int(val) for val in firstline.strip().split())

        embedding = [[] for _ in range(n)]
        for line in f:
            tokens = line.strip().split()
            embedding[int(tokens[0])].extend([float(val) for val in tokens[1:]])

    return n, size, embedding


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