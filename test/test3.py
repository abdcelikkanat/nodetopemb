from nodetopemb import *
import gensim #modified gensim version

dataset = "citeseer"
g = nx.read_gml("../datasets/" + dataset+".gml")

num_of_documents = 1
params = {"alpha": 0.0}
num_of_paths = 40 #80
path_length = 256 #40
window_size = 10
num_of_workers = 3
method = "Deepwalk"

num_of_nodes = g.number_of_nodes()
word_embed_size = 128
print("Number of nodes: {}".format(num_of_nodes))

walks_file = "../output/triwalk_test.walks"
embedding_file = "../output/triwalk.embedding"

graph = Graph()
graph.add_edges_from([[int(edge[0]), int(edge[1])] for edge in g.edges()])

with open(walks_file, "w") as f:
    for _ in range(num_of_documents):
        document = graph.graph2doc(number_of_paths=num_of_paths,
                                   path_length=path_length,
                                   params=params, rand=random.Random(),
                                   method=method)
        for line in document:
            f.write("{}\n".format(" ".join(str(node) for node in line)))


# Learn embeddings

sentences = gensim.models.word2vec.LineSentence(walks_file)

w1 = gensim.models.Word2Vec(sentences, size=word_embed_size, window=window_size,
                            sg=1, hs=1, workers=num_of_workers,
                            sample=0.001, negative=5)

w1.wv.save_word2vec_format(embedding_file)