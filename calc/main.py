
import numpy as np
from time import time
from func_2d import *
from plotter import render_animation

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

SIM_TIME = 400

N_x ,  N_y = 100, 100
N_v = 8

T1 = 1
T2 = 1

dx = dy = 1
dze_1 = np.sqrt(T2)
dze_cut = 4.8 * dze_1
dt = dx / dze_cut/2.1
V = np.linspace(-dze_cut, dze_cut, N_v)
assert 0 not in V
print(V, V[1]*dt/dx)
F = np.zeros((N_x, N_y, N_v, N_v), dtype=np.float64)
#F = set_maxwell(F, V, T1)
F[50,50,7,7] = 1 #pointed gas moves up and right
F/=F.sum()

left = right = up = down = np.array([T1 for i in range(N_x)], dtype=np.float64)
def start_2d(F, V, dx, dy, dt):
    START_TIME = time()
    saved_F = [F]
    for t in range(SIM_TIME):
        F = solve_angle_np(F, V, V, dx, dy, dt)
        t1 = time()
        F = reflect(left, right, up, down, F, N_x, N_y, V ,V)
        if t % 10 ==0:
            saved_F.append(F.copy())
    print(f"TIME = {time() - START_TIME} s, {time() - t1}")
    render_animation(saved_F, dt = 1)


start_2d(F, V, dx, dy, dt)





















