import cupy as cp
import numpy as np
from time import time
from numba import njit


def maxwell(x, T):
    norm = 1 #np.sqrt(np.pi*T)
    return np.exp(-x*x/2/T)/norm

def solve_angle(f_x: cp.ndarray, dze, dx, dt):
    t = time()
    if dze > 0:
        f_x = f_x * (1 - dze / dx * dt) + cp.roll(f_x, 1, 0) * (dze / dx * dt)
        f_x[0] = 0
    elif dze < 0:
        f_x = f_x * (1 + dze / dx * dt) - cp.roll(f_x, -1, 0) * (dze / dx * dt)
        f_x[-1] = 0
    else:
        print(f"!!!! dze = 0 !!!!")
        exit()
    print(f"TIME {time()-t}")
    return f_x

def solve_angle_np(f_x: np.ndarray, dze, dx, dt):
    t = time()
    if dze > 0:
        f_x = f_x * (1 - dze / dx * dt) + np.roll(f_x, 1, 0) * (dze / dx * dt)
        f_x[0] = 0
    elif dze < 0:
        f_x = f_x * (1 + dze / dx * dt) - np.roll(f_x, -1, 0) * (dze / dx * dt)
        f_x[-1] = 0
    else:
        print(f"!!!! dze = 0 !!!!")
        exit()
    #print(f"TIME {time()-t}")
    return f_x


def reflect_right(f_x: np.ndarray, v: np.array, T):
    #right side
    v_pos = int(len(v)/2)
    exp_dist = maxwell(v[:v_pos], T)
    h_t = v[v_pos:].dot(f_x[v_pos:, -1]) / np.abs(v[:v_pos]).dot(exp_dist)
    f_x[:v_pos, -1] = h_t * exp_dist
    return f_x

def reflect_left(f_x: np.ndarray, v: np.array, T):
    #left side
    v_pos = int(len(v) / 2)
    exp_dist = maxwell(v[v_pos:], T)
    h_t = np.abs(v[:v_pos]).dot(f_x[:v_pos, 0]) / (v[v_pos:].dot(exp_dist))
    f_x[v_pos:, 0] = h_t * exp_dist
    #print("SUM_CHECK ", np.sum(f_x[v_pos:, -1]), np.sum(f_x[:v_pos, -1]))
    return f_x
