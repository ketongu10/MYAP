
import numpy as np
from multiprocessing import freeze_support
from time import time
from func_2d import *
from pool_2d import *
from plotter import *
from metrics import *
from init_distributions import set_1, set_maxwell
from coll_int_3d import prepare_collision_mesh, CALC_COLL_INT
from warnings import filterwarnings
import os, shutil
import json

def create_shit_dir(name):
    try:
        os.mkdir(f"../plots/{name}")
    except:
        shutil.rmtree(f"../plots/{name}")
        os.mkdir(f"../plots/{name}")

SIM_TIME = 5000
sim_step = SIM_TIME//100
float_type = np.float64
N_integral = 1000

N_x ,  N_y = 100, 50
N_v = 16

N_b = 4
b_max = np.pi/4

T1 = 1
T2 = 2.0

dx = dy = 1

dum = {}
dum["SIM_TIME"]=SIM_TIME
dum["N_integral"]=N_integral
dum["N_x"]=N_x
dum["N_y"]=N_y
dum["N_v"]=N_v
dum["N_b"]=N_b
dum["b_max"]=b_max
dum["T1"]=T1
dum["T2"]=T2
dum["dx"]=dx
dum["dy"]=dy

dze_1 = np.sqrt(T2)
dze_cut = 4.8 * dze_1
dt = dx / dze_cut/2.1
V = np.linspace(-dze_cut, dze_cut, N_v, dtype=float_type)
VV = np.zeros(shape=(N_v, N_v), dtype=float_type)#np.kron(np.square(V), np.square(V)).reshape(N_v, N_v)
for vx, VX in enumerate(np.square(V)):
    for vy, VY in enumerate(np.square(V)):
        VV[vx,vy]=VX+VY

Vxy = np.meshgrid(V, V)

assert 0 not in V
#print(V, V[1]*dt/dx)
F = np.zeros((N_x, N_y, N_v, N_v), dtype=float_type)
#F[5:95, 5:95, :,:] = set_maxwell(F[5:95, 5:95, :,:], V, T1)
F = set_maxwell(F, V, T1)
#F/=F.sum()
#F[8, 8,4,4] *= 1.3 #pointed gas moves up and right
F/=F.sum()
rho = F[1,1].sum()
chip =(4, 10, 44)#(2, 6, 13)

up = down = np.array([T1 for i in range(N_x)], dtype=float_type)
left = right = np.array([T1 for i in range(N_y)], dtype=float_type)
down = up.copy()
down[chip[2]:chip[2]+chip[1]] = T2

experiment_name = f"time={SIM_TIME}_nv={N_v}_nx={N_x}_ny={N_y}_T={T2}_njit_full_goodQ"
create_shit_dir(experiment_name)
u = 0.01
#print("F[0,0]",np.sum(F[0, 0]))
def start_2d(F, V,dx, dy, dt):
    START_TIME = time()
    saved_F = [F]
    saved_T = [temperature(F, VV, chip)]
    print(saved_T)
    saved_hV = [hydroV(F, V, Vxy)]
    saved_dQ = [dQ_on_chip(F, saved_hV[-1], VV, V, chip, u)]
    #buffered_mesh = prepare_collision_mesh(N_b, b_max, N_v, dze_cut, float_type)
    for t in range(SIM_TIME):
        t1 = time()
        F = solve_angle_np(F, V, V, dx, dy, dt)
        #F = reflect(left, right, up, down, F, N_x, N_y, V, V)
        #F = reflect_with_flow(left, right, up, down, F, N_x, N_y, V ,V, 0.01, rho)
        F = reflect_with_flow_and_chip(left, right, up, down, F, N_x, N_y, V, V, u, rho, chip)
        #CALC_COLL_INT(F, N_integral, buffered_mesh, (N_x, N_y), dt, use_all_mesh=True)

        if t % 100 ==0:
            saved_F.append(F.copy())
            saved_T.append(temperature(F, VV, chip))
            saved_hV.append(hydroV(F, V, Vxy))
            saved_dQ.append(dQ_on_chip(F, saved_hV[-1], VV, V, chip, u))
            print(saved_dQ[-1])
        if t % sim_step == 0:
            print(f"PROGRESS: {t//sim_step}% | TOTAL TIME: {(time() - START_TIME):.01f}s | STEP TIME: {(time() - t1):.03f}s")
    print(f"TIME = {(time() - START_TIME):.01f}s, {(time() - t1):.01f}s")

    render_animation_all(saved_F, dt=1, temp=saved_T, max_f=1/N_x/N_y*1.5*dx*dy, max_t=T2*1.1, filename=experiment_name, chip=chip)
    plot_dQ(saved_dQ, filename=experiment_name)
    plot_last_vH(saved_hV, filename=experiment_name)
    np.save(f"../plots/{experiment_name}/F.npy", saved_F)
    #np.save(f"../plots/{experiment_name}/T.npy", saved_T)
    np.save(f"../plots/{experiment_name}/hV.npy", saved_hV)
    np.save(f"../plots/{experiment_name}/dQ.npy", saved_dQ)
    np.save(f"../plots/{experiment_name}/V.npy", V)
    with open(f"../plots/{experiment_name}/settings.txt", "w") as f:
        json.dump(dum, f)
    print(f"FINAL Q = f{saved_dQ[-1]}")


if __name__ == '__main__':
    start_2d(F, V, dx, dy, dt)





















