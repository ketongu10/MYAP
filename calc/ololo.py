import numpy as np


a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([1, 2], dtype=np.float32)
#c = np.zeros(shape=(2, 2, 2,2), dtype=np.float32)

print(np.kron(a, b))
