import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt



#g = nx.read_gml("../datasets/karate.gml")
g = nx.Graph()
g.add_edges_from([[str(edge[0]), str(edge[1])] for edge in [[0, 1], [0,2], [1,3], [2,3], [3,4], [4,5], [4,6], [5,7], [6,7]]])


A = np.asarray(nx.adjacency_matrix(g).todense(), dtype=np.float)
print(A)
P = (A / np.sum(A, 0)).T

P1 = P
P2 = np.dot(P, P1)
P3 = np.dot(P, P2)
P4 = np.dot(P, P3)
P5 = np.dot(P, P4)

Z = P1 + P2 + P3 + P4 + P5

Z = ((Z / np.sum(A, 0)) * np.sum(A, 0)).T

print(Z)

eigvals, eigvects = np.linalg.eigh(Z)

plt.figure()
plt.plot(eigvects[:4, 0], eigvects[:4, 1], 'r.')
plt.plot(eigvects[4:, 0], eigvects[4:, 1], 'b.')
plt.show()

print(list(g.nodes()))