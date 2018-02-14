from nodetopemb import *
import networkx as nx

def main(filename, corpus, num_of_paths, path_length, num_of_documents, max_data_size):

    data_size = num_of_paths * path_length

    print("Data size : HATALI", data_size)



    with open(filename, "w") as f:
        f.write(u"{}\n".format(num_of_documents))
        for _ in range(num_of_documents):
            document = corpus.generate_corpus(number_of_paths=num_of_paths,
                                              path_length=path_length,
                                              alpha=0.0, rand=random.Random())

            f.write(u"{}".format(u" ".join(str(v) for walk in document for v in walk )))
            f.write(u"{}\n")
"""
edges = [[0,1], [0,2], [0,3], [0,4], [0,5], [1,2], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [3,5], [4,5]]

g = Graph()
g.add_edges_from(edge_list=edges)
"""
nxgraph = nx.read_gml("../datasets/citeseer.gml")


edges = []
for edge in nxgraph.edges():
    edges.append([int(edge[0]), int(edge[1])])

g = Graph()
g.add_edges_from(edge_list=edges)


corpus = Corpus()
corpus.set_graph(g)

main(filename="../output/citeseer.dat", corpus=corpus, num_of_paths=80, path_length=40, num_of_documents=1, max_data_size=None)