import random
import node2vec
import networkx as nx

class Graph:
    """

    """
    adj_list = []
    num_of_nodes = 0
    num_of_edges = 0

    def __init__(self):
        pass

    def number_of_nodes(self):

        return self.num_of_nodes

    def number_of_edges(self):

        return self.num_of_edges

    def add_edges_from(self, edge_list):
        # It is assumed that node labels starts from 0 and up to n

        self.num_of_nodes = max([max(edge) for edge in edge_list]) + 1

        self.adj_list = [[] for _ in range(self.num_of_nodes)]
        for edge in edge_list:
            self.adj_list[edge[0]].append(edge[1])
            self.adj_list[edge[1]].append(edge[0])

            # Increase number of edges by 1 since it is a simple graph
            self.num_of_edges += 1

    def deepwalk_step(self, path_length, alpha=0.0, rand=random.Random(), starting_node=None):

        if starting_node is None:
            starting_node = rand.choice(range(self.number_of_nodes()))

        path = [starting_node]
        current_path_length = 1

        while current_path_length < path_length:
            # Get the latest appended node
            latest_node = path[-1]
            # If the current node has any neighbour
            if len(self.adj_list[latest_node]) > 0:
                # Return to the starting node with probability alpha
                if rand.random() >= alpha:
                    path.append(rand.choice(self.adj_list[latest_node]))
                else:
                    path.append(path.append(self.adj_list[0]))

                current_path_length += 1
            else:
                break

        return path

    def graph2doc(self, number_of_paths, path_length, params=dict(), rand=random.Random(), method="Deepwalk"):

        corpus = []

        node_list = range(self.num_of_nodes)

        if method == "Deepwalk":
            alpha = params['alpha']

            for _ in range(number_of_paths):
                # Shuffle the nodes
                rand.shuffle(node_list)
                # For each node, initialize a random walk
                for node in node_list:
                    walk = self.deepwalk_step(path_length=path_length, rand=rand, alpha=alpha,
                                              starting_node=node)
                    corpus.append(walk)

        if method == "Node2Vec":
            # Generate the desired networkx graph
            p = params['p']
            q = params['q']

            nxg = nx.Graph()
            for i in range(self.number_of_nodes()):
                for j in self.adj_list[i]:
                    nxg.add_edge(str(i), str(j))
                    nxg[str(i)][str(j)]['weight'] = 1

            G = node2vec.Graph(nx_G=nxg, p=p, q=q, is_directed=False)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(num_walks=number_of_paths, walk_length=path_length)

            corpus = walks


        return corpus
