import cupy as cp
import numpy as np
from time import time
from numba import njit


def maxwell(x, T):
    norm = 1 #np.sqrt(np.pi*T)
    return np.exp(-x*x/2/T)/norm


@njit(parallel=True, cache=True)
def solve_angle_np(F: np.ndarray, Vx, Vy, dx, dy, dt):
    vx_pos = int(len(Vx) / 2)
    vy_pos = int(len(Vy) / 2)
    for vx in range(vx_pos, len(Vx)-1):
        for vy in range(vy_pos, len(Vy) - 1):
            F[:,:,vx, vy] = F[:,:,vx, vy] * (1 - Vx[vx] / dx * dt - Vy[vy] / dy * dt) + \
                            np.roll(F[:,:,vx, vy], 1, 0) * (Vx[vx] / dx * dt) + \
                            np.roll(F[:,:,vx, vy], 1, 1) * (Vy[vy] / dy * dt)
    for vx in range(vx_pos):
        for vy in range(vy_pos):
            F[:,:,vx, vy] = F[:,:,vx, vy] * (1 + Vx[vx] / dx * dt + Vy[vy] / dy * dt) - \
                            np.roll(F[:,:,vx, vy], -1, 0) * (Vx[vx] / dx * dt) - \
                            np.roll(F[:,:,vx, vy], -1, 1) * (Vy[vy] / dy * dt)

    return F


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
