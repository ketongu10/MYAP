
import numpy as np
from time import time
from func_2d import *

def set_1(f_x: np.array):
    n = len(f_x)
    dn = n//20
    f_x[n//2-dn:n//2+dn] = np.array([1 for i in range(2*dn)])
    f_x/=f_x.sum()

SIM_TIME = 25

N_x ,  N_y = 20, 20
N_v = 2

T1 = 1
T2 = 1

dx = dy = 1
dze_1 = np.sqrt(T2)
dze_cut = 4.8 * dze_1
dt = dx / dze_cut/2

V = np.linspace(-dze_cut, dze_cut, N_v)
assert 0 not in V
print(V, V[1]*dt/dx)
F = np.zeros((N_x, N_y, N_v, N_v), dtype=np.float32)
F[0,0,1,1] = 1 #pointed gas moves up and right
F/=F.sum()

left = right = up = down = np.array([T1 for i in range(N_x)], dtype=np.float32)
def start_2d(F, V, dx, dy, dt):
    START_TIME = time()
    for t in range(SIM_TIME):
        F = solve_angle_np(F, V, V, dx, dy, dt)
        t1 = time()
        F = reflect(left, right, up, down, F, N_x, N_y, V ,V)
        print(np.where(F > 0.2), F[1,1,1,1])
    print(f"TIME = {time() - START_TIME} s, {time() - t1}")


start_2d(F, V, dx, dy, dt)




















def start_1d():
    START_TIME = time()
    for t in range(SIM_TIME):
        for v, velocity in enumerate(V):
            F_x[v] = solve_angle_np(F_x[v], velocity, dx, dt)
        reflect_right(F_x, V, T2)
        reflect_left(F_x, V, T1)
    print(f"END_TIME = {time()-START_TIME} s")
#print(F_x, np.sum(F_x[4]))

