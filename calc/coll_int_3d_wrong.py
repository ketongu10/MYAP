import numpy as np
from numba import njit
from time import time
from init_distributions import set_maxwell
from multiprocessing import Pool

import seaborn as sns
import matplotlib.pyplot as plt


#@njit(cache=True)
def shind(ind, mid):
    i = int(ind)
    return (i+mid, i+mid-1) if i<0 else (i+mid-1, i+mid)


#@njit(cache=True)
def calc_coll_integral(F: np.ndarray, V: np.array, float_type):
    B = np.linspace(-0.1, 0.1, 10)
    dB = B[1]-B[0]
    mesh = np.meshgrid(V, V, V, V, B)
    print(mesh)
    mesh_inputs = mesh.ravel()
    mesh_out = gen_outs(mesh_inputs) #v1,v2
    fit_out = fit(mesh_out, real_mesh) #i1x, i1y, i2x, i2y, j1x, j1y, j2x, j2y - indexes on real mesh
    N_mesh = len(mesh_inputs)
    N_TO_CALT_INT = 100
    for vx, vy in real_mesh:
        chosen_inds = np.random.randint(0, N_mesh, size=N_TO_CALT_INT)
        I = 0
        for i in chosen_inds:
            I += main_integral_function(F, mesh_inputs[i], fit_out[i],  real_mesh, vx, vy)






def gen_outs(mesh):
    out = np.ndarray(shape=mesh.shape, dtype=mesh.dtype)
    out = calc_out_velocities(mesh)
    return out







#@njit(cache=True)
def calc_out_velocities(input_data):
    v, v1, b_ = input_data[:2], input_data[2:4], input_data[4]
    g = v1-v

    if g[0] < 0:
        beta = np.pi+np.arctan(g[1]/g[0])
    elif g[0] == 0:
        beta = np.sign(g[1])*np.pi/2
    else:
        beta = np.arctan(g[1] / g[0])
    _S = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]]) #??? arctan
    alpha = np.arcsin(b_)
    gg = np.linalg.norm(g)
    ca, sa = np.cos(alpha), np.sin(alpha)
    _v, _v1 = np.array([gg*ca*ca, gg*sa*ca]), np.array([gg*sa*sa, -gg*sa*ca])
    v_, v1_ = v+_S.dot(_v), v+_S.dot(_v1)

    return np.hstack((v_, v1_))

test1 = 0
def fit_to_base_mesh(dV_base, v_mean, v_max, output):
    global test1
    v_, v1_ = output[:2], output[2:]
    v_ind, v1_ind = v_ / dV_base, v1_ / dV_base
    (v_indx_l, v_indx_m), (v_indy_l, v_indy_m) = shind(v_ind[0], v_mean), shind(v_ind[1], v_mean)
    (v1_indx_l, v1_indx_m), (v1_indy_l, v1_indy_m) = shind(v1_ind[0], v_mean), shind(v1_ind[1], v_mean)
    if (v_indx_l > v_max or v_indx_m > v_max or v_indy_l > v_max or v_indy_m > v_max or
            v1_indx_l > v_max or v1_indx_m > v_max or v1_indy_l > v_max or v1_indy_m > v_max or
            v_indx_l < 0 or v_indx_m < 0 or v_indy_l < 0 or v_indy_m < 0 or
            v1_indx_l < 0 or v1_indx_m < 0 or v1_indy_l < 0 or v1_indy_m < 0):
        #print(v_, v1_)
        test1+=1


    return np.array([v_indx_l, v_indy_l, v_indx_m, v_indy_m, v1_indx_l, v1_indy_l, v1_indx_m, v1_indy_m], dtype=np.int8)

test2 = 0
def calculate_r_energy_koef(input_vel, out_vel):
    global test2
    v1_less, v1_more, v2_less, v2_more = out_vel[:2], out_vel[2:4], out_vel[4:6], out_vel[6:]
    v1, v2 = input_vel[:2], input_vel[2:4]
    E0 = v1.dot(v1) + v2.dot(v2)
    E_less = v1_less.dot(v1_less) + v2_less.dot(v2_less)
    E_more = v1_more.dot(v1_more) + v2_more.dot(v2_more)
    r = 0.5 if (E_more - E_less) == 0 else (E0 - E_less) / (E_more - E_less)
    if r > 1 or r < 0:
        #print(r, v1, v2 ,v1_less, v1_more, v2_less, v2_more)
        test2+=1
    return r


if __name__ == '__main__':
    print("COLLISION INTEGRAL TESTS STARTED")




    t0 = time()

    dze_cut = 1
    # V_mesh for F
    N_v_base = 8
    V_base = np.linspace(-dze_cut, dze_cut, N_v_base)
    V_base_indexes = np.zeros(shape=(N_v_base, N_v_base, 2), dtype=np.int8)
    dV_base = V_base[1] - V_base[0]
    v_max = N_v_base - 1
    v_mean = int(N_v_base / 2)
    for i in range(N_v_base):
        for j in range(N_v_base):
            V_base_indexes[i, j] = np.array([i, j])

    N_x = N_y = 1
    float_type = np.float32

    F = np.zeros((N_x, N_y, N_v_base, N_v_base), dtype=float_type)
    F[...] = np.random.rand(N_x, N_y, N_v_base, N_v_base)
    F/=F.sum()
    print(F)

    # V_mesh for calculating J(f)
    N_v_cub = 16
    V_cub = np.linspace(-dze_cut, dze_cut, N_v_cub)
    V_cub_indexes = np.zeros(shape=(N_v_cub, N_v_cub, 2), dtype=np.int8)

    N_b = 8
    B = np.linspace(-np.pi/4, np.pi/4, N_b)
    dB = B[1] - B[0]



    V_input = np.zeros(shape=(N_b*N_v_cub**4, 5), dtype=np.float32)
    V_input_indexes_cub = np.zeros(shape=(N_b * N_v_cub ** 4, 4), dtype=np.int8)
    V_input_indexes_base = np.zeros(shape=(N_b * N_v_cub ** 4, 4), dtype=np.int8)
    V_out = np.zeros(shape=(N_b*N_v_cub**4, 4), dtype=np.float32)
    V_fit = np.zeros(shape=(N_b*N_v_cub**4, 8), dtype=np.int8)
    r_energy_koef = np.zeros(shape=(N_b*N_v_cub**4,), dtype=np.float32)
    for i in range(N_v_cub):
        for j in range(N_v_cub):
            for n in range(N_v_cub):
                for m in range(N_v_cub):
                    for k in range(N_b):
                        ind = k+m*N_b+n*N_v_cub*N_b+j*N_v_cub*N_v_cub*N_b+i*N_v_cub*N_v_cub*N_v_cub*N_b
                        V_input[ind] = np.array([V_cub[i], V_cub[j], V_cub[n], V_cub[m], B[k]])
                        V_input_indexes_cub[ind] = np.array([i, j, n, m])

    for ind, prepared in enumerate(V_input):
        V_out[ind] = calc_out_velocities(prepared)
        V_input_indexes_base[ind] = fit_to_base_mesh(dV_base, v_mean, v_max, prepared)[[2,3,6,7]]

    for ind, prepared in enumerate(V_out):
        V_fit[ind] = fit_to_base_mesh(dV_base, v_mean, v_max, prepared)
        try:
            r_energy_koef[ind] = calculate_r_energy_koef(V_input[ind], V_base[V_fit[ind]])
        except:
            pass
    print(f"INIT={time()-t0}")
    print(test1,test2,N_b*N_v_cub**4)
    t0 = time()
    N_integral = 10
    I = np.zeros(shape=(N_x, N_y, N_v_base, N_v_base), dtype=float_type)
    for x in range(N_x):
        for y in range(N_y):
            input_samples = np.random.randint(0, N_b * N_v_cub ** 4, size=N_integral)
            for sample in input_samples:

                r = r_energy_koef[sample]
                C = 1 * np.linalg.norm(V_input[sample, :2] - V_input[sample, 2:4])
                Omega = F[x, y, V_fit[sample, 0], V_fit[sample, 1]] * F[x, y, V_fit[sample, 4], V_fit[sample, 5]] * \
                        np.power(
                            (F[x, y, V_fit[sample, 2], V_fit[sample, 3]] * F[x, y, V_fit[sample, 6], V_fit[sample, 7]] /
                             (F[x, y, V_fit[sample, 0], V_fit[sample, 1]] * F[
                                 x, y, V_fit[sample, 4], V_fit[sample, 5]])), r) - \
                        F[x, y, V_input_indexes_base[sample, 0], V_input_indexes_base[sample, 1]] * \
                        F[x, y, V_input_indexes_base[sample, 2], V_input_indexes_base[sample, 3]]

                I[x, y, V_input_indexes_base[sample, 0], V_input_indexes_base[sample, 1]] += C * Omega
                I[x, y, V_input_indexes_base[sample, 2], V_input_indexes_base[sample, 3]] += C * Omega
                I[x, y, V_fit[sample, 2], V_fit[sample, 3]] -= C * Omega * r
                I[x, y, V_fit[sample, 6], V_fit[sample, 7]] -= C * Omega * r
                I[x, y, V_fit[sample, 0], V_fit[sample, 1]] -= C * Omega * (1 - r)
                I[x, y, V_fit[sample, 4], V_fit[sample, 5]] -= C * Omega * (1 - r)



    print(f"CALC={time()-t0}")






