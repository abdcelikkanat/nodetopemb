import gensim
import numpy
from nodetopemb import *

embedding_dim = 128
window_size = 10
workers = 3
#walks_file = "../output/citeseer_unheader.dat"
#output_file = "../output/citeseer_raw.embeddings"


walks_file = "../output/simple.dat"
output_file = "../output/simple.embeddings"


walks = gensim.models.word2vec.LineSentence(walks_file)

print("Training")
model = gensim.models.Word2Vec(walks, size=embedding_dim, window=window_size,
                               min_count=0, sg=1, hs=1,
                               workers=workers)

model.wv.save_word2vec_format(output_file)


