#/home/abdulkadir/anaconda2/envs/twe/bin python

import oldgensim.gensim as gensim2 #modified gensim version
from get_topics import get_topics

def concatenate_embeddings(word_embed_file, topic_embed_file, word2topic, output_filename):
    # Combine Embeddings
    print("Embeddings are being concatenated...")
    word_embed = {}
    with open(word_embed_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            word_embed.update({tokens[0]: [val for val in tokens[1:]]})

    topic_num = 0
    topic_embed = {}
    with open(topic_embed_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embed.update({unicode(topic_num): [val for val in tokens]})
            topic_num += 1



    #with open(word2topic_file, 'r') as f:
    #    for line in f:
    #        tokens = line.strip().split()
    #        word2topic.update({tokens[0]: tokens[1]})


    combined_embed = {}
    for word in word_embed:
        combined_embed.update({word: word_embed[word] + topic_embed[word2topic[word]]})

    number_of_nodes = len(combined_embed.keys())
    total_embedding_size = len(combined_embed.values()[0])
    with open(output_filename, 'w') as f:
        f.write("{} {}\n".format(number_of_nodes, total_embedding_size))
        for word in combined_embed:
            f.write("{} {}\n".format(word, " ".join(combined_embed[word])))

def extract_embedding(corpus_filename, topic_filename, number_of_topics, word_embed_filename, word_embed_size, topic_embed_filename, topic_embed_size, combined_embed_filename):
    """
    Files which will be used
    - corpus_filename
    Files which will be generated
    - topic_filename
    - word_embed_filename
    - topic_embed_filename
    -

    """
    print("The word corpus file is being read...")
    word_sentences = gensim2.models.word2vec.Text8Corpus(corpus_filename)


    print "Training the word vector..."
    w1 = gensim2.models.Word2Vec(word_sentences, size=word_embed_size, window=window_size,
                                 sg=1, hs=1, workers=1,
                                 sample=0.001, negative=5)
    print "Saving the word vectors..."
    w1.save_wordvector(word_embed_filename)

    print("Topics of each node are being detected")
    word2topic = get_topics(walks_file=corpus_filename, output_topic_file=topic_filename,
                            number_of_topics=number_of_topics, number_of_nodes=number_of_nodes)

    print("The word corpus and topic files are being combined...")
    combined_sentences = gensim2.models.word2vec.CombinedSentence(corpus_filename, topic_filename)

    print "Training the topic vector..."
    if word_embed_size == topic_embed_size:
        w1.train_topic(number_of_topics, combined_sentences)
        print "Saving the topic vectors..."
        w1.save_topic(topic_embed_filename)
    else:
        w2 = gensim2.models.Word2Vec(size=topic_embed_size, window=window_size,
                                     sg=1, hs=1, workers=1,
                                     sample=0.001, negative=5)
        w2.build_vocab(word_sentences)
        w2.train_topic(number_of_topics, combined_sentences)
        print "Saving the topic vectors..."
        w2.save_topic(topic_embed_file)

    print("Embeddings are being concatenated...")
    concatenate_embeddings(word_embed_file=word_embed_filename, topic_embed_file=topic_embed_file,
                           word2topic=word2topic, output_filename=combined_embed_filename)

dataset_name = "karate"
number_of_topics = 2
number_of_nodes = 34

word_embed_size = 128
topic_embed_size = 128
window_size = 10


walks_file = "./input/" + dataset_name + "_walk.corpus"
topic_file = "./input/" + dataset_name + "_topic.corpus"
word2topic_file = "./input/" + dataset_name + "_word2topic.dat"

word_embed_file = "./output/" + dataset_name +"_word.embedding"
topic_embed_file = "./output/" + dataset_name +"_topic.embedding"
#combined_embed_file = "../../deepwalk/example_graphs3/ " + dataset_name +"_combined_embedding"
combined_embed_file = "./output/" + dataset_name +"_combined.embedding"


extract_embedding(corpus_filename=walks_file, topic_filename=topic_file, number_of_topics=number_of_topics,
                  word_embed_filename=word_embed_file, word_embed_size=word_embed_size,
                  topic_embed_filename=topic_embed_file, topic_embed_size=topic_embed_size,
                  combined_embed_filename=combined_embed_file)
