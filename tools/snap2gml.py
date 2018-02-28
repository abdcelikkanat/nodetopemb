import numpy as np
import networkx as nx


input_name = "facebook_combined.txt"
output_name = "facebook.gml"
skip_line_count = 4



input_file = "../datasets/{}".format(input_name)
output_file = "../datasets/{}".format(output_name)


g = nx.Graph()

with open(input_file, 'r') as f:
    skip_counter = 0
    for line in f:
        if skip_counter < skip_line_count:
            skip_counter += 1
            continue

        node1, node2 = line.strip().split()
        g.add_edge(int(node1), int(node2))

# Relabel nodes so that they start from 0
mapping = dict()
nodeId = 0
for node in g.nodes():
    mapping[node] = nodeId
    nodeId += 1

g = nx.relabel_nodes(g, mapping)

print("Number of nodes: {}\nNumber of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

nx.write_gml(g, output_file)

