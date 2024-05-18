import numpy as np
from func_2d import maxwell

def set_1(f_x: np.array):
    n = len(f_x)
    dn = n//20
    f_x[n//2-dn:n//2+dn] = np.array([1 for i in range(2*dn)])
    f_x/=f_x.sum()

def set_maxwell(f, V, T):
    for i, vx in enumerate(V):
        for j, vy in enumerate(V):
            f[:,:,i, j] = maxwell(np.sqrt(vx*vx+vy*vy), T)
    return f