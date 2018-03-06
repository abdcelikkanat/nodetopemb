#/home/abdulkadir/anaconda2/envs/twe/bin python

import os
import time
import oldgensim.gensim as gensim2 #modified gensim version
import preprocess as pre_process
import numpy as np

def concatenate_embeddings(word_embed_file, topic_embed_file, word2topic_file, output_filename):
    # Combine Embeddings
    print("Embeddings are being concatenated...")
    word_embed = {}
    with open(word_embed_file, 'r') as f:
        f.readline() # ignore the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            word_embed.update({tokens[0]: [val for val in tokens[1:]]})

    topic_num = 0
    topic_embed = {}
    with open(topic_embed_file, 'r') as f:
        f.readline()  # ignore the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embed.update({unicode(topic_num): [val for val in tokens]})
            topic_num += 1

    # Read the word2topic file, file format: (word topic)
    word2topic = {}
    with open(word2topic_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            word2topic.update({tokens[0]: tokens[1]})

    combined_embed = {}
    for word in word_embed:
        combined_embed.update({word: word_embed[word] + topic_embed[word2topic[word]]})

    number_of_nodes = len(combined_embed.keys())
    total_embedding_size = len(combined_embed.values()[0])
    with open(output_filename, 'w') as f:
        f.write("{} {}\n".format(number_of_nodes, total_embedding_size))
        for word in combined_embed:
            f.write("{} {}\n".format(word, " ".join(combined_embed[word])))


def learn_embedding(corpus_file, topic_file, window_size, number_of_nodes, number_of_topics, word2topic_file, word_embed_file, word_embed_size, topic_embed_file, topic_embed_size, combined_embed_file, num_of_workers, passes):
    """
    Files which will be used
    - corpus_filename
    Files which will be generated
    - topic_filename
    - word_embed_filename
    - topic_embed_filename
    -

    """





    print("Topics of each node are being detected...")
    """
    start_time = time.time()
    #word2topic = get_topics(walks_file=corpus_filename, output_topic_file=topic_filename,
    #                        number_of_topics=number_of_topics, number_of_nodes=number_of_nodes)
    gensim_path = "/home/abdulkadir/anaconda2/envs/twegensim/bin/python"
    os.system(gensim_path + " ./get_topics.py "+corpus_file+" "+topic_file+" "+word2topic_file+" "+str(number_of_topics)+" "+str(number_of_nodes)+" "+str(passes))
    """
    corpus_lines = []
    with open(corpus_file, 'r') as f:
        corpus_lines.append(f.readline())

    with open("./gibbslda/tmp/walks.data", 'w') as f:
        f.write("{}\n".format(len(corpus_lines)))
        for line in corpus_lines:
            f.write(line)

    os.system("./gibbslda/lda -est -alpha 0.5 -beta 0.1 -savestep 2000 -ntopics "+str(number_of_topics)+" -niters 1000 -dfile ./gibbslda/tmp/walks.data")
    wordmapfile = "./gibbslda/tmp/wordmap.txt"
    id2word = pre_process.load_id2word(wordmapfile)
    tassignfile = "./gibbslda/tmp/model-final.tassign"
    pre_process.load_sentences(tassignfile, id2word)


    topic_word_prob = np.zeros(shape=(number_of_topics, number_of_nodes), dtype=np.float)
    say = 0
    with open("./gibbslda/tmp/model-final.phi", 'r') as f:
        for vals in f.readlines():
            topic_word_prob[say, :] = [float(v) for v in vals.strip().split()]
            say += 1
    print(topic_word_prob)
    argmaxinx = np.argmax(topic_word_prob, axis=0)
    word2topic = {}
    print(argmaxinx)
    print(argmaxinx.shape)
    for i in range(argmaxinx.shape[0]):
        word2topic.update({id2word[i]: argmaxinx[i]})

    # Save the word2topic file
    with open(word2topic_file, 'w') as f:
        for word in word2topic:
            f.write("{} {}\n".format(word, word2topic[word]))


    print("The word corpus file is being read...")
    word_sentences = gensim2.models.word2vec.Text8Corpus("./gibbslda/tmp/word.file")
    start_time = time.time()
    print "Training the word vector..."
    w1 = gensim2.models.Word2Vec(word_sentences, size=word_embed_size, window=window_size,
                                 sg=1, hs=1, workers=num_of_workers,
                                 sample=0.001, negative=5)
    print "Saving the word vectors..."
    w1.save_wordvector(word_embed_file)
    print("The computation time for the word vectors: {}".format(time.time() - start_time))


    print("The computation time of topics: {}".format(time.time() - start_time))
    #########################################
    print("The word corpus and topic files are being combined...")
    combined_sentences = gensim2.models.word2vec.CombinedSentence("./gibbslda/tmp/word.file","./gibbslda/tmp/topic.file")

    start_time = time.time()
    print "Training the topic vector..."
    if word_embed_size == topic_embed_size:
        w1.train_topic(number_of_topics, combined_sentences)
        print "Saving the topic vectors..."
        w1.save_topic(topic_embed_file)
    else:
        w2 = gensim2.models.Word2Vec(size=topic_embed_size, window=window_size,
                                     sg=1, hs=1, workers=num_of_workers,
                                     sample=0.001, negative=5)
        w2.build_vocab(word_sentences)
        w2.train_topic(number_of_topics, combined_sentences)
        print "Saving the topic vectors..."
        w2.save_topic(topic_embed_file)
    print("The computation time for the topic vectors: {}".format(time.time() - start_time))

    start_time = time.time()
    print("Embeddings are being concatenated...")
    concatenate_embeddings(word_embed_file=word_embed_file, topic_embed_file=topic_embed_file,
                           word2topic_file=word2topic_file, output_filename=combined_embed_file)
    print("The computation time for the concatenation operations: {}".format(time.time() - start_time))
