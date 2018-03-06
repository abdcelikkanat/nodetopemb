import gensim
import numpy as np
from nodetopemb import *
import time


number_of_topics = 350

embedding_dim = 128
window_size = 10
workers = 3
#walks_file = "../output/citeseer_unheader.dat"
#output_file = "../output/citeseer_raw.embeddings"

folder = "citeseerOneLine"
#folder = "simple1"
walks_file = "../output/"+folder+"_unheader.dat"
output_file = "../output/"+folder+".embeddings"


def update_walks_for_twe2(wordmap_file, tassing_file, number_of_topics=0):
    # Read tassing file and extract
    number_of_nodes = 0
    id2word = {}
    with open(wordmap_file, 'r') as f:
        number_of_nodes = int(f.readline().strip().split()[0])  # skip the first line
        for line in f:
            m = line.strip().split()
            id2word.update({m[1]: m[0]})

    #wordXtopicMatrix = np.zeros(shape=(len(id2word, number_of_topics)))

    new_walk = []
    with open(tassing_file, 'r') as f:
        content = f.readline().strip().split()

        for pair in content:
            wordId, topic = pair.split(':')
            new_walk.append(id2word[wordId]+"-"+topic)

    return new_walk, number_of_nodes


new_walks, number_of_nodes = update_walks_for_twe2(wordmap_file="../input/"+folder+"_wordmap.txt", tassing_file="../input/"+folder+"_model-final.tassign")
new_walks_file = "../output/"+folder+"_twe2.dat"
with open(new_walks_file, "w") as f:
    f.write(" ".join(new_walks))


print("Training")
start = time.time()
raw_walks = gensim.models.word2vec.LineSentence(walks_file)
modell = gensim.models.Word2Vec(raw_walks, size=embedding_dim, window=window_size,
                               min_count=0, sg=1, hs=1,
                               workers=workers)
modell.wv.save_word2vec_format("../output/"+folder+"_raw.embeddings")
print("Exec time: "+ str(time.time()-start))




print("Training")
start = time.time()
new_walks = gensim.models.word2vec.LineSentence(new_walks_file)
model = gensim.models.Word2Vec(new_walks, size=embedding_dim, window=window_size,
                               min_count=0, sg=1, hs=1,
                               workers=workers)

output_file = "../output/"+folder+"_inprocess_twe2.embeddings"
model.wv.save_word2vec_format(output_file)
print("Exec time: "+ str(time.time()-start))


embedIndex = []

word2topicMatrix = np.zeros(shape=(number_of_nodes, number_of_topics), dtype=np.int)
with open(output_file, 'r') as f:
    f.readline()
    for line in f.readlines():
        l = line.strip().split()

        word, topic = l[0].split('-')
        word2topicMatrix[int(word), int(topic)] += 1

        ###########
        embedIndex.append(int(word))
        ###########

embeddings = [[] for _ in range(number_of_nodes)]
with open(output_file, 'r') as f:
    pair_counts, _ = f.readline().strip().split()
    pair_counts = int(pair_counts)


    for line in f.readlines():
        l = line.strip().split()

        word, topic = l[0].split('-')

        #embeddings[int(word)].append([np.float(val) for val in l[1:]])
        embeddings[int(word)].append([np.float(val)*word2topicMatrix[int(word), int(topic)] for val in l[1:]])



#embeddings_mean = [np.mean(embeddings[i], axis=0) for i in range(number_of_nodes)]
embeddings_mean = [np.sum(embeddings[i], axis=0)/np.sum(word2topicMatrix[i]) for i in range(number_of_nodes)]

output_file = "../output/"+folder+"_twe2.embeddings"
with open(output_file, 'w') as f:
    f.write(str(number_of_nodes) + " " + str(embedding_dim) + "\n")
    for i in range(number_of_nodes):
    #for i in embedIndex:
        f.write(str(i) + " " + " ".join([str(val) for val in embeddings_mean[i]]) + "\n")