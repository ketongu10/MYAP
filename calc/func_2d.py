
import numpy as np
from time import time
from multiprocessing import Pool
from metrics import temperature

def maxwell_start(x, T):
    return x*x*np.exp(-x*x/2/T)

def maxwell(x, T):
    return np.exp(-x*x/2/T)

def maxwell_with_flow(x, v, T):
    return np.exp(-(x-v) * (x-v) / 2 / T)

def solve_angle_np(F: np.ndarray, Vx, Vy, dx, dy, dt):
    vx_pos = int(len(Vx) / 2)
    vy_pos = int(len(Vy) / 2)

    # ++
    for vx in range(vx_pos, len(Vx)):
        for vy in range(vy_pos, len(Vy)):
            F[:,:,vx, vy] = F[:,:,vx, vy] * (1 - Vx[vx] / dx * dt - Vy[vy] / dy * dt) + \
                            np.roll(F[:,:,vx, vy], 1, 0) * (Vx[vx] / dx * dt) + \
                            np.roll(F[:,:,vx, vy], 1, 1) * (Vy[vy] / dy * dt)

    # +-

    for vx in range(vx_pos, len(Vx)):
        for vy in range(vy_pos):
            F[:, :, vx, vy] = F[:, :, vx, vy] * (1 - Vx[vx] / dx * dt + Vy[vy] / dy * dt) + \
                              np.roll(F[:, :, vx, vy], 1, 0) * (Vx[vx] / dx * dt) - \
                              np.roll(F[:, :, vx, vy], -1, 1) * (Vy[vy] / dy * dt)
    # -+
    for vx in range(vx_pos):
        for vy in range(vy_pos, len(Vy)):
            F[:, :, vx, vy] = F[:, :, vx, vy] * (1 + Vx[vx] / dx * dt - Vy[vy] / dy * dt) - \
                              np.roll(F[:, :, vx, vy], -1, 0) * (Vx[vx] / dx * dt) + \
                              np.roll(F[:, :, vx, vy], 1, 1) * (Vy[vy] / dy * dt)
    # --
    for vx in range(vx_pos):
        for vy in range(vy_pos):
            F[:,:,vx, vy] = F[:,:,vx, vy] * (1 + Vx[vx] / dx * dt + Vy[vy] / dy * dt) - \
                            np.roll(F[:,:,vx, vy], -1, 0) * (Vx[vx] / dx * dt) - \
                            np.roll(F[:,:,vx, vy], -1, 1) * (Vy[vy] / dy * dt)

    return F

def reflect(left, right, up, down, F, N_x, N_y, Vx, Vy):
    n_vx = len(Vx)
    n_vy = len(Vy)
    vx_pos = int(n_vx / 2)
    vy_pos = int(n_vy / 2)



    # up
    for x in range(N_x):
        # exp_dist = maxwell(Vy[:vy_pos], up[x])
        # exp_f = np.abs(Vy[:vy_pos]).dot(exp_dist)
        # h_t = np.tensordot(Vy[vy_pos:], (F[x, -1, :, vy_pos:]), ([0], [1])) / exp_f
        #
        # F[x, -1, :, :vy_pos] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)

        exp_dist_x = maxwell(Vx, up[x])
        exp_dist_y = maxwell(Vy[:vy_pos], up[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[:vy_pos]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[vy_pos:]), (F[x, -1, :, vy_pos:]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, -1, :, :vy_pos] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)



    # down
    for x in range(N_x):
        # exp_dist = maxwell(Vy[vy_pos:], down[x])
        # exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist)
        # h_t = np.tensordot(np.abs(Vy[:vy_pos]), (F[x,0,:,:vy_pos]), ([0], [1])) / exp_f
        # F[x, 0, :, vy_pos:] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)

        exp_dist_x = maxwell(Vx, down[x])
        exp_dist_y = maxwell(Vy[vy_pos:], down[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[:vy_pos]), (F[x, 0, :, :vy_pos]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, 0, :, vy_pos:] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)


    # right
    for y in range(0, N_y):
        # exp_dist = maxwell(Vx[:vx_pos], right[y])
        # exp_f = np.abs(Vx[:vx_pos]).dot(exp_dist)
        # h_t = np.tensordot(Vx[vx_pos:], (F[-1, y, vx_pos:, :]), ([0], [0])) / exp_f
        # F[-1, y, :vx_pos, :] = np.kron(exp_dist, h_t).reshape(vx_pos, n_vy)

        exp_dist_x = maxwell(Vx[:vx_pos], right[y])
        exp_dist_y = maxwell(Vy, right[y])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vx[:vx_pos]).dot(exp_dist_x)
        h_t = (np.tensordot(np.abs(Vx[vx_pos:]), (F[-1, y, vx_pos:, :]), ([0], [0]))).sum() / (exp_f).sum()
        F[-1, y, :vx_pos, :] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)


    # left
    for y in range(0, N_y):
        # exp_dist = maxwell(Vx[vx_pos:], left[y])
        # exp_f = np.abs(Vx[vx_pos:]).dot(exp_dist)
        # h_t = np.tensordot(np.abs(Vx[:vx_pos]), (F[0, y, :vx_pos, :]), ([0], [0])) / exp_f
        # F[0, y, vx_pos:, :] = np.kron(exp_dist, h_t).reshape(vx_pos, n_vy)

        exp_dist_x = maxwell(Vx[vx_pos:], left[y])
        exp_dist_y = maxwell(Vy, left[y])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vx[vx_pos:]).dot(exp_dist_x)
        h_t = (np.tensordot(np.abs(Vx[:vx_pos]), (F[0, y, :vx_pos, :]), ([0], [0]))).sum() / (exp_f).sum()
        F[0, y, vx_pos:, :] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)

    return F

def reflect_with_flow(left, right, up, down, F, N_x, N_y, Vx, Vy, f_in, rho):
    n_vx = len(Vx)
    n_vy = len(Vy)
    vx_pos = int(n_vx / 2)
    vy_pos = int(n_vy / 2)

    # right
    for y in range(0, N_y):
        F[-1, y, :vx_pos, :] *= 0.01

    # left
    for y in range(0, N_y):
        exp_dist_x = maxwell_with_flow(Vx[vx_pos:], f_in,left[y])
        exp_dist_y = maxwell(Vy, left[y])
        exp_dist_x/= exp_dist_x.sum()
        exp_dist_y/=exp_dist_y.sum()
        F[0, y, vx_pos:, :] = rho * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)

    # up
    for x in range(N_x):
        # exp_dist = maxwell(Vy[:vy_pos], up[x])
        # exp_f = np.abs(Vy[:vy_pos]).dot(exp_dist)
        # h_t = np.tensordot(Vy[vy_pos:], (F[x, -1, :, vy_pos:]), ([0], [1])) / exp_f
        # # print(x, h_t)
        # F[x, -1, :, :vy_pos] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)

        exp_dist_x = maxwell(Vx, up[x])
        exp_dist_y = maxwell(Vy[:vy_pos], up[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[:vy_pos]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[vy_pos:]), (F[x, -1, :, vy_pos:]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, -1, :, :vy_pos] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)


    # down
    for x in range(N_x):
        # exp_dist = maxwell(Vy[vy_pos:], down[x])
        # exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist)
        # h_t = np.tensordot(np.abs(Vy[:vy_pos]), (F[x, 0, :, :vy_pos]), ([0], [1])) / exp_f
        # F[x, 0, :, vy_pos:] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)

        exp_dist_x = maxwell(Vx, down[x])
        exp_dist_y = maxwell(Vy[vy_pos:], down[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[:vy_pos]), (F[x, 0, :, :vy_pos]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, 0, :, vy_pos:] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)

    return F


def reflect_with_flow_and_chip(left, right, up, down, F, N_x, N_y, Vx, Vy, f_in, rho, chip):
    h_chip, w_chip, x_start = chip
    n_vx = len(Vx)
    n_vy = len(Vy)
    vx_pos = int(n_vx / 2)
    vy_pos = int(n_vy / 2)





    # up
    for x in range(N_x):
        # exp_dist = maxwell(Vy[:vy_pos], up[x])
        # exp_f = np.abs(Vy[:vy_pos]).dot(exp_dist)
        # h_t = np.tensordot(Vy[vy_pos:], (F[x, -1, :, vy_pos:]), ([0], [1])) / exp_f
        # # print(x, h_t)
        # F[x, -1, :, :vy_pos] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)

        exp_dist_x = maxwell(Vx, up[x])
        exp_dist_y = maxwell(Vy[:vy_pos], up[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[:vy_pos]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[vy_pos:]), (F[x, -1, :, vy_pos:]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, -1, :, :vy_pos] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)

    # down
    for x in range(N_x):
        # exp_dist = maxwell(Vy[vy_pos:], down[x])
        # exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist)
        # h_t = np.tensordot(np.abs(Vy[:vy_pos]), (F[x, 0, :, :vy_pos]), ([0], [1])) / exp_f
        # F[x, 0, :, vy_pos:] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)

        exp_dist_x = maxwell(Vx, down[x])
        exp_dist_y = maxwell(Vy[vy_pos:], down[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[:vy_pos]), (F[x, 0, :, :vy_pos]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, 0, :, vy_pos:] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)




    # right
    for y in range(0, N_y):
        #F[-1, y, :vx_pos, :] *= 0.01
        # exp_dist_x = maxwell_with_flow(Vx, f_in, left[y])
        # exp_dist_y = maxwell(Vy, left[y])
        # exp_dist_x /= exp_dist_x.sum()
        # exp_dist_y /= exp_dist_y.sum()
        # F[-1, y, :, :] = rho * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, n_vy)

        exp_dist_x = maxwell_with_flow(Vx[:vx_pos], f_in, left[y])
        exp_dist_y = maxwell(Vy, left[y])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        F[-1, y, :vx_pos, :] = rho * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)

    # left
    for y in range(0, N_y):
        exp_dist_x = maxwell_with_flow(Vx[vx_pos:], f_in, left[y])

        exp_dist_y = maxwell(Vy, left[y])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        F[0, y, vx_pos:, :] = rho * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)

        # exp_dist_x = maxwell_with_flow(Vx, f_in, left[y])
        # exp_dist_y = maxwell(Vy, left[y])
        # exp_dist_x /= exp_dist_x.sum()
        # exp_dist_y /= exp_dist_y.sum()
        # F[0, y, :, :] = rho * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, n_vy)
    # print(F[:,:,])



    # chip

    # up

    for x in range(x_start, x_start+w_chip):
        # exp_dist = maxwell(Vy[vy_pos:], down[x])
        # exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist)
        # h_t = np.tensordot(np.abs(Vy[:vy_pos]), (F[x, 0, :, :vy_pos]), ([0], [1])) / exp_f
        # F[x, h_chip, :, vy_pos:] = np.kron(h_t, exp_dist).reshape(n_vx, vy_pos)
        # F[x, :h_chip, :, :] = 0

        exp_dist_x = maxwell(Vx, down[x])
        exp_dist_y = maxwell(Vy[vy_pos:], down[x])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vy[vy_pos:]).dot(exp_dist_y)
        h_t = (np.tensordot(np.abs(Vy[:vy_pos]), (F[x, h_chip, :, :vy_pos]), ([0], [1]))).sum() / (exp_f).sum()
        F[x, h_chip, :, vy_pos:] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(n_vx, vy_pos)
        F[x, :h_chip, :, :] = 0



    # left
    for y in range(h_chip):
        # exp_dist = maxwell(Vx[:vx_pos], right[y])
        # exp_f = np.abs(Vx[:vx_pos]).dot(exp_dist)
        # h_t = np.tensordot(Vx[vx_pos:], (F[x_start, y, vx_pos:, :]), ([0], [0])) / exp_f
        # F[x_start, y, :vx_pos, :] = np.kron(exp_dist, h_t).reshape(vx_pos, n_vy)

        exp_dist_x = maxwell(Vx[:vx_pos], down[x_start+1])
        exp_dist_y = maxwell(Vy, down[x_start])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vx[:vx_pos]).dot(exp_dist_x)
        h_t = (np.tensordot(np.abs(Vx[vx_pos:]), (F[x_start-1, y, vx_pos:, :]), ([0], [0]))).sum() / (exp_f).sum()
        F[x_start-1, y, :vx_pos, :] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)

    # right
    for y in range(h_chip):
        # exp_dist = maxwell(Vx[vx_pos:], left[y])
        # exp_f = np.abs(Vx[vx_pos:]).dot(exp_dist)
        # h_t = np.tensordot(np.abs(Vx[:vx_pos]), (F[x_start+w_chip, y, :vx_pos, :]), ([0], [0])) / exp_f
        # F[x_start+w_chip, y, vx_pos:, :] = np.kron(exp_dist, h_t).reshape(vx_pos, n_vy)

        exp_dist_x = maxwell(Vx[vx_pos:], down[x_start+1])
        exp_dist_y = maxwell(Vy, down[x_start+1])
        exp_dist_x /= exp_dist_x.sum()
        exp_dist_y /= exp_dist_y.sum()
        exp_f = np.abs(Vx[vx_pos:]).dot(exp_dist_x)
        h_t = (np.tensordot(np.abs(Vx[:vx_pos]), (F[x_start+w_chip, y, :vx_pos, :]), ([0], [0]))).sum() / (exp_f).sum()
        F[x_start+w_chip, y, vx_pos:, :] = h_t * np.kron(exp_dist_x, exp_dist_y).reshape(vx_pos, n_vy)


    return F


def prepare_space(N_x, N_y, chip):
    h_chip, w_chip, x_start = chip
    space = []