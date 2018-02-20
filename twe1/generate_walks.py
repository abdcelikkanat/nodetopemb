import networkx as nx
from nodetopemb import Graph
import random


def saveGraphWalks(graph, output_walks_file, num_of_paths, path_length, num_of_documents, together=True):

    data_size = num_of_paths * path_length * graph.number_of_nodes()

    print("Data size: {} = The num. of paths x path length x the num. of nodes".format(data_size))

    with open(output_walks_file, "w") as f:

        for _ in range(num_of_documents):
            document = graph.graph2doc(number_of_paths=num_of_paths,
                                       path_length=path_length,
                                       alpha=0.0, rand=random.Random())
            if together:
                # Save documents line by line
                f.write(u"{}\n".format(u" ".join(str(node) for line in document for node in line)))
            else:
                for line in document:
                    f.write(u"{}\n".format(u" ".join(str(node) for node in line)))

dataset_name = "karate"
nx_graph_path = "../datasets/"+dataset_name+".gml"
output_walks_file = "./input/"+dataset_name+".corpus"

nx_graph = nx.read_gml(nx_graph_path)

g = Graph()
edges = [[int(edge[0]), int(edge[1])] for edge in nx_graph.edges()]
g.add_edges_from(edge_list=edges)
saveGraphWalks(graph=g,
               output_walks_file=output_walks_file,
               num_of_paths=5,
               path_length=4,
               num_of_documents=2,
               together=False)
