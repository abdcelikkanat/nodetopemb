import numpy as np
from nodetopemb import *
from twe1.link_pred.split2train_test import *
g = nx.read_gml("../datasets/karate.gml")


a = set([1, 2])
b = set([2, 3])

a.intersection(b)

print(a)