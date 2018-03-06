from graph.graph import Graph


myg = Graph()
myg.add_edges_from([[0,1],[2,3]])
print(myg.number_of_edges())