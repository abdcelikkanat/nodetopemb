from extract_embeddings import extract_embedding
import time
import networkx as nx
import oldgensim.gensim as gensim2 #modified gensim version

dataset_name = "karate"
number_of_topics = 2
number_of_nodes = nx.read_gml("../datasets/"+dataset_name+".gml").number_of_nodes()

word_embed_size = 128
topic_embed_size = 128
window_size = 10


walks_file = "./input/" + dataset_name + "_walk.corpus"
topic_file = "./input/" + dataset_name + "_topic.corpus"
word2topic_file = "./input/" + dataset_name + "_word2topic.map"

word_embed_file = "./output/" + dataset_name +"_word.embedding"
topic_embed_file = "./output/" + dataset_name +"_topic.embedding"
#combined_embed_file = "../../deepwalk/example_graphs3/ " + dataset_name +"_combined_embedding"
combined_embed_file = "./output/" + dataset_name +"_combined.embedding"

start_time = time.time()
extract_embedding(corpus_file=walks_file, topic_file=topic_file,
                  number_of_nodes=number_of_nodes, number_of_topics=number_of_topics, window_size=window_size,
                  word2topic_file=word2topic_file,
                  word_embed_file=word_embed_file, word_embed_size=word_embed_size,
                  topic_embed_file=topic_embed_file, topic_embed_size=topic_embed_size,
                  combined_embed_file=combined_embed_file)
print("Total time: {}".format(time.time()-start_time))