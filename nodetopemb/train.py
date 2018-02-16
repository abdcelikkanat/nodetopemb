#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import numpy as np
from nodetopemb import *
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

number_of_nodes = 0
number_of_topics = 5

embedding_dim = 128
window_size = 10
workers = 3
#walks_file = "../output/citeseer_unheader.dat"
#output_file = "../output/citeseer_raw.embeddings"

folder = "simple"
#folder = "simple1"
walks_file = "../input/"+folder+".dat"
output_file = "../output/"+folder+".embeddings"

# Read the document -> number of walks
## It is assumed that the file consists of one line
document = []
with open(walks_file) as f:
    for line in f:
        for v in line.strip().split():
            if v:
                document.append(unicode(int(v)))

                node = int(v)
                if node > number_of_nodes:
                    number_of_nodes = node
number_of_nodes += 1

# Set the dictionary
dict = gensim.corpora.Dictionary([[unicode(node) for node in range(number_of_nodes)]])

# Extract the corpus
corpus = [dict.doc2bow(document)]

# Run LDA
id2word = {i:v for i, v in dict.items()}
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=number_of_topics)

# Find the topic assignments of each word
word2topic = {}
for word in dict.values():
    top_prob = lda.get_term_topics(dict.token2id[word])
    word2topic.update({word: max(top_prob,key=lambda item:item[1])[0]})


print("Training for original walks")
# Generate new walks in the form of (word:topic) for TWE2
new_walks = []
for node in document:
    node_topic = word2topic[node]
    new_walks.append(node+":"+str(node_topic))

## Run the word2vec to extract embeddings, and write the embeddings to a file
new_walks_model = gensim.models.Word2Vec(new_walks, size=embedding_dim, window=window_size,
                                         min_count=0, sg=1, hs=1, workers=workers)
## Save the embeddings
new_walks_model.wv.save_word2vec_format("../output/"+folder+"_.embeddings")


print("Training for TWE2")
# Generate new walks in the form of (word:topic) for TWE2
new_walks = []
for node in document:
    node_topic = word2topic[node]
    new_walks.append(node+":"+str(node_topic))

## Run the word2vec to extract embeddings, and write the embeddings to a file
new_walks_model = gensim.models.Word2Vec(new_walks, size=embedding_dim, window=window_size,
                                         min_count=0, sg=1, hs=1, workers=workers)
## Save the embeddings
new_walks_model.wv.save_word2vec_format("../output/"+folder+"_twe2.embeddings")




