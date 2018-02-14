from nodetopemb import *

edges = [[0,1], [0,2], [0,3], [0,4], [0,5], [1,2], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [3,5], [4,5]]

g = Graph()
g.add_edges_from(edge_list=edges)

corpus = Corpus()
corpus.set_graph(g)


corpus = corpus.generate_corpus(number_of_paths=5, path_length=3)
print(corpus)



