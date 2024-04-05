import numpy as np


a = np.array([[[[1, 2], [3, 5]], [[1, 22], [3, 4]]], [[[1, 2], [3, 4]], [[1, 12], [3, 4]]]], dtype=np.float32)
b = np.array([[1, 2], [3, 4]], dtype=np.float32)
c = np.zeros(shape=(2, 2, 2,2), dtype=np.float32)
for i in range(2):
    for j in range(2):
        c[:, :, i, j] = a[:,:,i, j]*b[i, j]
print((a*b))
print(c)
