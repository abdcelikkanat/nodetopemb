import random

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

