import random


def saveGraphWalks(graph, output_walks_file, num_of_paths, path_length, num_of_documents, params, method, together=True):

    data_size = num_of_paths * path_length * graph.number_of_nodes()

    print("Data size: {} = The num. of paths x path length x the num. of nodes".format(data_size))

    with open(output_walks_file, "w") as f:

        for _ in range(num_of_documents):
            document = graph.graph2doc(number_of_paths=num_of_paths,
                                       path_length=path_length,
                                       params=params, rand=random.Random(),
                                       method=method)

            if together:
                # Save documents line by line
                f.write(u"{}\n".format(u" ".join(str(node) for line in document for node in line)))
            else:
                for line in document:
                    f.write(u"{}\n".format(u" ".join(str(node) for node in line)))




