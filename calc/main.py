import cupy as cp
import numpy as np
from time import time
from func_1d import *

def set_1(f_x: np.array):
    n = len(f_x)
    dn = n//20
    f_x[n//2-dn:n//2+dn] = np.array([1 for i in range(2*dn)])
    f_x/=f_x.sum()

SIM_TIME = 10

N_x = 20
N_v = 2

T1 = 1
T2 = 1

dx = 1
dze_1 = np.sqrt(T2)
dze_cut = 4.8 * dze_1
dt = dx / dze_cut

V = np.linspace(-dze_cut, dze_cut, N_v)
assert 0 not in V
print(V, V[1]*dt/dx)
F_x = np.zeros((N_v, N_x), dtype=np.float32)

for v, velocity in enumerate(V):
    set_1(F_x[v])
F_x/=F_x.sum()

START_TIME = time()
for t in range(SIM_TIME):
    for v, velocity in enumerate(V):
        F_x[v] = solve_angle_np(F_x[v], velocity, dx, dt)
    reflect_right(F_x, V, T2)
    reflect_left(F_x, V, T1)
print(f"END_TIME = {time()-START_TIME} s")
#print(F_x, np.sum(F_x[4]))
