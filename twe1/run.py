from graph.graph import Graph
from learn_embeddings import learn_embedding
from generate_walks_corpus import saveGraphWalks
import time
import networkx as nx
import oldgensim.gensim as gensim2 #modified gensim version


dataset_name = "blogcatalog"
#suffix = "deepwalk_numpath40_pathlen10"
#suffix = "deepwalk_numpath10_pathlen80"
#suffix = "numpath10_pathlen80_p025_q025"
#suffix = "numpath10_pathlen80_p025_q025"
suffix = "pathlen10_numpaths40_topic300_iter1000"


nx_graph_path = "../datasets/"+dataset_name+".gml"
nx_graph = nx.read_gml(nx_graph_path)

#nx_graph = max(nx.connected_component_subgraphs(nx_graph), key=len)

number_of_topics = 300 #65
number_of_nodes = nx_graph.number_of_nodes()

generate_walks = True
num_of_paths = 40 # 80
path_length = 10 # 40
window_size = 10 # 10

num_of_documents = 1
together = False
num_of_workers = 3
passes = 1

method = "Deepwalk"
#method = "Node2Vec"
#method = "degreeBasedWalk"
params = {'alpha': 0.0}
#params = {'p':0.25, 'q':0.25 }

word_embed_size = 128
topic_embed_size = 128



walks_file = "./temp_files/input/" + dataset_name + "/" + dataset_name + "_" + suffix + "_walk.corpus"
topic_file = "./temp_files/input/" + dataset_name + "/" + dataset_name + "_" + suffix + "_topic.corpus"
word2topic_file = "./temp_files/input/" + dataset_name + "/" + dataset_name + "_" + suffix + "_word2topic.map"

word_embed_file = "./temp_files/output/" + dataset_name + "/" + dataset_name + "_" + suffix + "_word.embedding"
topic_embed_file = "./temp_files/output/" + dataset_name + "/" + dataset_name + "_" + suffix + "_topic.embedding"
combined_embed_file = "./temp_files/output/" + dataset_name + "/" + dataset_name + "_" + suffix + "_combined.embedding"


print(dataset_name+"_"+suffix)


if generate_walks is True:
    g = Graph()
    edges = [[int(edge[0]), int(edge[1])] for edge in nx_graph.edges()]
    g.add_edges_from(edge_list=edges)
    saveGraphWalks(graph=g,
                   output_walks_file=walks_file,
                   num_of_paths=num_of_paths,
                   path_length=path_length,
                   num_of_documents=num_of_documents,
                   params=params, method=method,
                   together=together)


start_time = time.time()
learn_embedding(corpus_file=walks_file, topic_file=topic_file,
                number_of_nodes=number_of_nodes, number_of_topics=number_of_topics, window_size=window_size,
                word2topic_file=word2topic_file,
                word_embed_file=word_embed_file, word_embed_size=word_embed_size,
                topic_embed_file=topic_embed_file, topic_embed_size=topic_embed_size,
                combined_embed_file=combined_embed_file,
                num_of_workers=num_of_workers, passes=passes)
total_time = time.time()-start_time
print("Total time: {} sec, {} min".format(total_time, total_time/60.0))
