from nodetopemb import *
import numpy as np
import csv
import networkx as nx

# topic_vector.txt contains embeddings of topics
# word_vector.txt contains embeddings of words
# model-final.phi is p(word|topic) where each line corresponds to topic and columns corresponds to words

number_of_nodes = nx.read_gml("../datasets/citeseer.gml").number_of_nodes()
number_of_topics = 100


word_embedding = [[] for _ in range(number_of_nodes)]
with open("../input/word64_vector.txt", 'r') as word:
    content = csv.reader(word, delimiter="\n")
    for line in content:
        temp = [val for val in line[0].split(' ') if val]
        node = np.int(temp[0])
        values = [np.float(val) for val in temp[1:]]

        word_embedding[node] += values

topic_embedding = []
with open("../input/topic64_vector.txt", 'r') as word:
    content = csv.reader(word, delimiter="\n")
    for line in content:
        values = [np.float(val) for val in line[0].split(' ') if val]

        topic_embedding.append(values)


word_topic_distr = []
with open("../input/model-final.phi") as phi:
    content = csv.reader(phi, delimiter="\n")
    n = 1
    for line in content:
        l = []
        words = [w for w in line[0].split(' ') if w]

        for w in words:
            l.append(np.float(w))

        word_topic_distr.append(l)

        n += 1

word_topic_distr = np.asarray(word_topic_distr)


# Concatenate embeddings
embeddings = [[] for _ in range(number_of_nodes)]

for i in range(number_of_nodes):
    topic_inx = np.argmax(word_topic_distr[:, i])
    embeddings[i] += word_embedding[i] + topic_embedding[topic_inx]

embeddings = np.asarray(embeddings)


# Write embeddings to a file
with open("../output/citeseer64.embeddings", 'w') as out:
    out.write(str(number_of_nodes) + " " + str(embeddings.shape[1]) + "\n")

    for i in range(number_of_nodes):
        line = str(i) + " " + " ".join(str(embed) for embed in embeddings[i, :]) + str('\n')
        out.write(line)