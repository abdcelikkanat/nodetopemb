#/home/abdulkadir/anaconda2/envs/embedding-tests/bin python
import sys
import gensim


def get_topics(walks_file, output_topic_file, word2topic_file, number_of_topics, number_of_nodes, passes):

    number_of_topics = int(number_of_topics)
    number_of_nodes = int(number_of_nodes)
    passes = int(passes)

    # Read the document -> number of walks
    ## It is assumed that the file consists of one line
    document = []
    nodes = []

    with open(walks_file) as f:
        for line in f:
            document.append([v for v in line.strip().split() if v])


    nodes = [i for i in range(number_of_nodes)]


    # Set the dictionary
    dict = gensim.corpora.Dictionary([[unicode(node) for node in range(number_of_nodes)]])

    # Extract the corpus
    corpus = [dict.doc2bow(doc) for doc in document]
    #corpus = [dict.doc2bow(document)]


    # Run LDA
    #id2word = {i:v for i, v in dict.items()}
    id2word = dict
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=number_of_topics, passes=passes)

    # Find the topic assignments of each word
    clusters = [[] for _ in range(number_of_topics)]

    word2topic = {}
    for word in dict.values():
        top_prob = lda.get_term_topics(dict.token2id[word], minimum_probability=-1.0)
        word2topic.update({word: max(top_prob, key=lambda item:item[1])[0]})
        clusters[word2topic[word]].append(word)

    topics = []
    for doc in document:
        for word in doc:
            topics.append(unicode(word2topic[word]))

    with open(output_topic_file, 'w') as f:
        f.write(" ".join(topics))

    for i in range(number_of_topics):
        print("The cluster {} contains {} number of nodes.".format(i, len(clusters[i])))

    # Save the word2topic file
    with open(word2topic_file, 'w') as f:
        for word in word2topic:
            f.write("{} {}\n".format(word, word2topic[word]))


if __name__ == "__main__":
    get_topics(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
