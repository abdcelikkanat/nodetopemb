from extract_embeddings import extract_embedding
from nodetopemb import Graph
from generate_walks_corpus import saveGraphWalks
import time
import networkx as nx
import oldgensim.gensim as gensim2 #modified gensim version

dataset_name = "citeseer"

nx_graph_path = "../datasets/"+dataset_name+".gml"
nx_graph = nx.read_gml(nx_graph_path)

number_of_topics = 65
number_of_nodes = nx_graph.number_of_nodes()

generate_walks = True
num_of_paths = 80
path_length = 40
window_size = 10
num_of_documents = 1
together = False

word_embed_size = 128
topic_embed_size = 128



walks_file = "./input/" + dataset_name + "/" + dataset_name + "_walk.corpus"
topic_file = "./input/" + dataset_name + "/" + dataset_name + "_topic.corpus"
word2topic_file = "./input/" + dataset_name + "/" + dataset_name + "_word2topic.map"

word_embed_file = "./output/" + dataset_name + "/" + dataset_name +"_word.embedding"
topic_embed_file = "./output/" + dataset_name + "/" + dataset_name +"_topic.embedding"
combined_embed_file = "./output/" + dataset_name + "/" + dataset_name +"_combined.embedding"




if generate_walks is True:
    g = Graph()
    edges = [[int(edge[0]), int(edge[1])] for edge in nx_graph.edges()]
    g.add_edges_from(edge_list=edges)
    saveGraphWalks(graph=g,
                   output_walks_file=walks_file,
                   num_of_paths=num_of_paths,
                   path_length=path_length,
                   num_of_documents=num_of_documents,
                   together=together)


start_time = time.time()
extract_embedding(corpus_file=walks_file, topic_file=topic_file,
                  number_of_nodes=number_of_nodes, number_of_topics=number_of_topics, window_size=window_size,
                  word2topic_file=word2topic_file,
                  word_embed_file=word_embed_file, word_embed_size=word_embed_size,
                  topic_embed_file=topic_embed_file, topic_embed_size=topic_embed_size,
                  combined_embed_file=combined_embed_file)
total_time = time.time()-start_time
print("Total time: {} sec, {} min".format(total_time, total_time/60.0))