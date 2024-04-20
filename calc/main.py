
import numpy as np
from multiprocessing import freeze_support
from time import time
from func_2d import *
from pool_2d import *
from plotter import render_animation
from metrics import temperature

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

SIM_TIME = 4000
sim_step = SIM_TIME//100

N_x ,  N_y = 100, 100
N_v = 16

T1 = 1
T2 = 1

dx = dy = 1
dze_1 = np.sqrt(2*T2)
dze_cut = 4.8 * dze_1
dt = dx / dze_cut/2.1
V = np.linspace(-dze_cut, dze_cut, N_v)
VV = np.kron(np.square(V), np.square(V)).reshape(N_v, N_v)
assert 0 not in V
print(V, V[1]*dt/dx)
F = np.zeros((N_x, N_y, N_v, N_v), dtype=np.float64)
#F[5:95, 5:95, :,:] = set_maxwell(F[5:95, 5:95, :,:], V, T1)
F = set_maxwell(F, V, T1)
#F[50,50,:,:] = 1 #pointed gas moves up and right
F/=F.sum()

left = right = up = down = np.array([T1*0.9 for i in range(N_x)], dtype=np.float64)
down = left.copy()
#down[40:60] = 0.1
print(right)
def start_2d(F, V, dx, dy, dt):

    START_TIME = time()
    saved_F = [F]
    saved_T = [temperature(F, VV)]
    for t in range(SIM_TIME):
        t1 = time()
        F = solve_angle_np(F, V, V, dx, dy, dt)
        F = reflect(left, right, up, down, F, N_x, N_y, V ,V)
        if t % 10 ==0:
            saved_F.append(F.copy())
            saved_T.append(temperature(F, VV))
        if t % sim_step == 0:
            print(f"PROGRESS: {t//sim_step}% | TOTAL TIME: {(time() - START_TIME):.01f}s | STEP TIME: {(time() - t1):.03f}s")
    print(f"TIME = {(time() - START_TIME):.01f}s, {(time() - t1):.01f}s")
    render_animation(saved_F, dt=1, temp=saved_T, vmax=1/N_x/N_y*1.5)


start_2d(F, V, dx, dy, dt)





















