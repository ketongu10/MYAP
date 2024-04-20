
import numpy as np
from time import time
from multiprocessing import Pool


def maxwell(x, T):
    return np.exp(-x*x/2/T)


def angle_xr(F: np.ndarray, Vx, Vy, dx, dy, dt, vx):
    vy_pos = int(len(Vy) / 2)

    for vy in range(vy_pos, len(Vy)):
        F[:, :, vx, vy] = F[:, :, vx, vy] * (1 - Vx[vx] / dx * dt - Vy[vy] / dy * dt) + \
                          np.roll(F[:, :, vx, vy], 1, 0) * (Vx[vx] / dx * dt) + \
                          np.roll(F[:, :, vx, vy], 1, 1) * (Vy[vy] / dy * dt)

    for vy in range(vy_pos):
        F[:, :, vx, vy] = F[:, :, vx, vy] * (1 - Vx[vx] / dx * dt + Vy[vy] / dy * dt) + \
                          np.roll(F[:, :, vx, vy], 1, 0) * (Vx[vx] / dx * dt) - \
                          np.roll(F[:, :, vx, vy], -1, 1) * (Vy[vy] / dy * dt)
    return F[:,:,vx,:]


def angle_xl(F: np.ndarray, Vx, Vy, dx, dy, dt, vx):
    vy_pos = int(len(Vy) / 2)

    for vy in range(vy_pos, len(Vy)):
        F[:, :, vx, vy] = F[:, :, vx, vy] * (1 + Vx[vx] / dx * dt - Vy[vy] / dy * dt) - \
                          np.roll(F[:, :, vx, vy], -1, 0) * (Vx[vx] / dx * dt) + \
                          np.roll(F[:, :, vx, vy], 1, 1) * (Vy[vy] / dy * dt)

    for vy in range(vy_pos):
        F[:, :, vx, vy] = F[:, :, vx, vy] * (1 + Vx[vx] / dx * dt + Vy[vy] / dy * dt) - \
                          np.roll(F[:, :, vx, vy], -1, 0) * (Vx[vx] / dx * dt) - \
                          np.roll(F[:, :, vx, vy], -1, 1) * (Vy[vy] / dy * dt)

    return F[:,:,vx,:]


def solve_angle_pool(F: np.ndarray, Vx, Vy, dx, dy, dt):
    vx_pos = int(len(Vx) / 2)

    with Pool(8) as p:
        args = [(F, Vx, Vy, dx, dy, dt, vx) for vx in range(vx_pos, len(Vx))]
        F_ = p.map(angle_xr, args)
        F[:, :, vx_pos:, :] = F_[:, :, vx_pos:, :]

        args = [(F, Vx, Vy, dx, dy, dt, vx) for vx in range(vx_pos)]
        F_ = p.map(angle_xl, args)
        F[:, :, :vx_pos, :] = F_[:, :, :vx_pos, :]



    return F