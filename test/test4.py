import numpy as np

M = np.array([[0,2,11],[4,1,7]])

print(np.argsort(M, axis=None)[::-1][:3])
print(M.flatten()[2])