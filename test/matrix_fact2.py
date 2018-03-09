import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt



#g = nx.read_gml("../datasets/karate.gml")
g = nx.Graph()
g.add_edges_from([[str(edge[0]), str(edge[1])] for edge in [[0, 1], [0,2], [1,3], [2,3], [3,4], [4,5], [4,6], [5,7], [6,7]]])



#P = np.asarray([[1.0 if g.has_edge(str(i), str(j)) else 0.0 for j in range(g.number_of_nodes()) ] for i in range(g.number_of_nodes())])

#P = P / np.sum(P, 0)
P = np.asarray(nx.adjacency_matrix(g).todense(), dtype=np.float)
P = P / np.sum(P)
M = P


M1 = P
M2 = np.dot(M, M)
M3 = np.dot(M, M2)
M4 = np.dot(M, M3)

AVG = np.zeros(shape=(g.number_of_nodes(), g.number_of_nodes()), dtype=np.float)
for i in range(g.number_of_nodes()):

    for j in range(i+1, g.number_of_nodes()):
        if M1[i, j] > 0.0:
            AVG[i, j] += 1.0*M1[i, j]

        if M2[i, j] > 0.0:
            AVG[i, j] += 4.0*M2[i, j]

        if M3[i, j] > 0.0:
            AVG[i, j] += 9.0*M3[i, j]

        if M4[i, j] > 0.0:
            AVG[i, j] += 16.0*M4[i, j]

        AVG[j, i] = AVG[i, j]

AVG = AVG
print(AVG)
M = np.zeros(shape=(g.number_of_nodes(), g.number_of_nodes()), dtype=np.float)
for node in range(g.number_of_nodes()):
    for nb in range(node, g.number_of_nodes()):
        #sp = float(nx.shortest_path_length(g, str(node), str(nb)))
        sp = AVG[int(node), int(nb)]
        if sp > 0 and sp < 5:
            M[int(node), int(nb)] = (float(nx.degree(g, str(node)))*float(nx.degree(g, str(nb)))) / (AVG[node, nb]*AVG[node, nb])
            M[int(nb), int(node)] = M[int(node), int(nb)]





print(M)
eigvals, eigvects = np.linalg.eigh(M)

plt.figure()
plt.plot(eigvects[:4, 0], eigvects[:4, 1], 'r.')
plt.plot(eigvects[4:, 0], eigvects[4:, 1], 'b.')
plt.show()

"""

groundTruth = np.asarray([0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1])


plt.figure()
for node in range(len(groundTruth)):
    if groundTruth[int(node)] == 0:
        plt.plot(eigvects[node, 0], eigvects[node, 1], 'r.')

    if groundTruth[int(node)] == 1:
        plt.plot(eigvects[node, 0], eigvects[node, 1], 'b.')
plt.show()

"""




"""
plt.figure()
pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos,
                       nodelist=[str(node) for node in range(g.number_of_nodes()) if groundTruth[int(node)] == 0],
                       node_color='r', node_size=100, alpha=0.8)
nx.draw_networkx_nodes(g, pos,
                       nodelist=[str(node) for node in range(g.number_of_nodes()) if groundTruth[int(node)] == 1],
                       node_color='b', node_size=100, alpha=0.8)

nx.draw_networkx_edges(g, pos,
                       edgelist=g.edges(), width=2, alpha=0.5)

plt.show()

"""