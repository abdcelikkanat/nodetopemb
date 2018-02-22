import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

g = nx.read_gml("../../datasets/karate.gml")

A = nx.adjacency_matrix(g).todense()

D = np.dot(A, A)**2

#for i in range(D.shape[0]):
#    D[i,i] = 1

eigval, eigvec = np.linalg.eigh(D)

x = eigvec[:, 0]
y = eigvec[:, 1]



plt.figure(1)
plt.plot(x,y,'rx')
plt.show()