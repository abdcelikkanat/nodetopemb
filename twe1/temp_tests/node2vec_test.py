from nodetopemb import graph
import networkx as nx

g = nx.read_gml("../../datasets/karate.gml")

edges = [[int(edge[0]), int(edge[1])] for edge in g.edges()]

myg = graph.Graph()
myg.add_edges_from(edge_list=edges)


corpus = myg.graph2doc(number_of_paths=5, path_length=3, params={'p':0.2, 'q':0.4}, method="Node2Vec")
print(corpus)