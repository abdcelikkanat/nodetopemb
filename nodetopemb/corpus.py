from graph import *
import random


class Corpus:

    g = None

    def __init__(self, graph=None):
        self.g = graph

    def set_graph(self, graph):
        self.g = graph

    def random_walk(self, path_length, alpha=0.0, rand=random.Random(), starting_node=None):

        if starting_node is None:
            starting_node = rand.choice(range(self.g.number_of_nodes()))

        path = [starting_node]
        current_path_length = 1

        while current_path_length < path_length:
            # Get the latest appended node
            latest_node = path[-1]
            # If the current node has any neighbour
            if len(self.g.adj_list[latest_node]) > 0:
                # Return to the starting node with probability alpha
                if rand.random() >= alpha:
                    path.append(rand.choice(self.g.adj_list[latest_node]))
                else:
                    path.append(path.append(self.g.adj_list[0]))

                current_path_length += 1
            else:
                break

        return path

    def generate_corpus(self, number_of_paths, path_length, alpha=0.0, rand=random.Random()):

        corpus = []

        node_list = range(self.g.number_of_nodes())

        for _ in range(number_of_paths):
            # Shuffle the nodes
            rand.shuffle(node_list)
            # For each node, initialize a random walk
            for node in node_list:
                walk = self.random_walk(path_length=path_length, rand=rand, alpha=alpha, starting_node=node)
                corpus.append(walk)

        return corpus


    def test(self):
        print("Test function")


