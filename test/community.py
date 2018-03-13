import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_nb_list(g):
    return [[int(nb) for nb in nx.neighbors(g, node)] for node in range(g.number_of_nodes())]

example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7], [3,4]]


edges = example1


g = nx.Graph()
g.add_edges_from(edges)

"""
plt.figure()
nx.draw(g)
plt.show()
"""

N = g.number_of_nodes()
D = 2
K = 2
eta = 0.001

F = np.random.normal(size=(N, K, D))*10.0
nb_list = get_nb_list(g)

F = np.abs(F)

num_of_iters = 2000
for iter in range(num_of_iters):

    logF = 0.0
    for node in range(N):
        ft = 0.0
        for nb in nb_list[node]:
            prod = np.sum([np.dot(F[node, k, :], F[nb, k, :]) for k in range(K)])
            ft += np.log(1.0 - np.exp(-prod) + 1e-6)

        lt = 0.0
        for lt_nb in range(N):
            if lt_nb != node and lt_nb not in nb_list[node]:
                lt += np.sum([np.dot(F[node, k, :], F[lt_nb, k, :]) for k in range(K)])

        logF += (ft - lt)

    print(logF)

    for node in range(N):


        first_term = np.zeros(shape=(K, D), dtype=np.float)
        last_term = np.zeros(shape=(K, D), dtype=np.float)
        for nb in nb_list[node]:
            prod = np.sum([np.dot(F[node, k, :], F[nb, k, :]) for k in range(K)])
            #print(prod)
            first_term += F[nb, :, :] * (np.exp(-prod) / (1.0 - np.exp(-prod)+1e-6))

            last_term += F[nb, :, :]

        Fsum = np.sum(F, axis=0)
        last_term = Fsum - last_term - F[node, :, :]

        delta_node = first_term - last_term

        # Update
        F[node, :, :] = F[node, :, :] + eta*delta_node

        F[node, :, :] = (F[node, :, :] + np.abs(F[node, :, :]))/2.0


    """
        err = 0.0
        for nb_node in g.nodes():
            prod = np.sum([np.dot(F[node, k, :], F[nb_node, k, :]) for k in range(K)])
            res = 1.0 - np.exp(-prod)


            if nb_node in nb_list[node]:
                err += np.abs(1.0 - res)
            else:
                err += np.abs(res)

        err = err / g.number_of_nodes()

        print(err)
    """
M = np.arange(0, 27).reshape((3,3,3))


print(F)

edges = [[np.dot(F[node, :, 0], F[nb, :, 0]) for nb in range(N)] for node in range(N) ]
print(np.asarray(edges))

for node in range(N):
    for k in range(K):
        print("Node: {}, Cluster: {} Val: {}".format(node, k, np.sum(F[node, k, :]**1)))

plt.figure()
plt.plot(F[:4, 0, 0], F[:4, 0, 1], 'r.')
plt.plot(F[:4, 1, 0], F[:4, 1, 1], 'b.')
plt.plot(F[4:, 0, 0], F[4:, 0, 1], 'rx')
plt.plot(F[4:, 1, 0], F[4:, 1, 1], 'bx')
plt.show()

""" """
