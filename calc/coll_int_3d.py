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
    return (i+mid, i+mid-1) if ind<0 else (i+mid-1, i+mid)

def calc_out_velocities(input_data, v_cut):
    v, v1, b_ = input_data[:2], input_data[2:4], input_data[4]
    g = v1-v

    if g[0] < 0:
        beta = np.pi+np.arctan(g[1]/g[0])
    elif g[0] == 0:
        return None
    else:
        beta = np.arctan(g[1] / g[0])
    _S = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]]) #??? arctan
    alpha = np.arcsin(b_)
    gg = np.linalg.norm(g)
    ca, sa = np.cos(alpha), np.sin(alpha)
    _v, _v1 = np.array([gg*ca*ca, gg*sa*ca]), np.array([gg*sa*sa, -gg*sa*ca])
    v_, v1_ = v+_S.dot(_v), v+_S.dot(_v1)
    if np.linalg.norm(v_) > v_cut or np.linalg.norm(v1_) > v_cut:
        return None
    return np.hstack((v_, v1_))


def fit_to_base_mesh(dV_base, v_mean, v_max, output, input, V_base):

    v_, v1_ = output[:2], output[2:]
    v_ind, v1_ind = v_ / dV_base, v1_ / dV_base
    (v_indx_l, v_indx_m), (v_indy_l, v_indy_m) = shind(v_ind[0], v_mean), shind(v_ind[1], v_mean)
    if (v_indx_l > v_max or v_indx_m > v_max or v_indy_l > v_max or v_indy_m > v_max or
            v_indx_l < 0 or v_indx_m < 0 or v_indy_l < 0 or v_indy_m < 0):

        return None
    else:
        v1_xl, v1_yl = input[:2]+input[2:4] - V_base[[v_indx_l, v_indy_l]]
        v1_xm, v1_ym = input[:2] + input[2:4] - V_base[[v_indx_m, v_indy_m]]
        (v1_indx_l, v1_indx_m), (v1_indy_l, v1_indy_m) = (shind(v1_xl/dV_base, v_mean)[1], shind(v1_xm/dV_base, v_mean)[1]), \
                                                         (shind(v1_yl/dV_base, v_mean)[1], shind(v1_ym/dV_base, v_mean)[1])


        if (v1_indx_l > v_max or v1_indx_m > v_max or v1_indy_l > v_max or v1_indy_m > v_max or
            v1_indx_l < 0 or v1_indx_m < 0 or v1_indy_l < 0 or v1_indy_m < 0):
            return None
        else:
            return np.array([v_indx_l, v_indy_l, v_indx_m, v_indy_m, v1_indx_l, v1_indy_l, v1_indx_m, v1_indy_m], dtype=np.int8)



def calculate_r_energy_koef(input_vel, out_vel):
    global test2
    v1_less, v1_more, v2_less, v2_more = out_vel[:2], out_vel[2:4], out_vel[4:6], out_vel[6:]
    v1, v2 = input_vel[:2], input_vel[2:4]
    E0 = v1.dot(v1) + v2.dot(v2)
    E_less = v1_less.dot(v1_less) + v2_less.dot(v2_less)
    E_more = v1_more.dot(v1_more) + v2_more.dot(v2_more)
    r = 0.5 if (E_more - E_less) == 0 else (E0 - E_less) / (E_more - E_less)
    if r > 1 or r < 0:
        return None
    return r



def prepare_collision_mesh(N_b, b_max, N_v_base, v_cut, float_type):


    V_base = np.linspace(-v_cut, v_cut, N_v_base)
    dV_base = V_base[1] - V_base[0]
    v_max = N_v_base - 1
    v_mean = int(N_v_base / 2)

    B = np.linspace(-b_max, b_max, N_b, dtype=float_type)
    dB = B[1] - B[0]


    V_input = np.zeros(shape=(N_b*N_v_base**4, 5), dtype=float_type)
    V_input_indexes_base = np.zeros(shape=(N_v_base**4*N_b, 4), dtype=np.int8)
    V_fit = np.zeros(shape=(N_b*N_v_base**4, 8), dtype=np.int8)
    r_energy_koef = np.zeros(shape=(N_b*N_v_base**4,), dtype=float_type)
    for i in range(N_v_base):
        for j in range(N_v_base):
            for n in range(N_v_base):
                for m in range(N_v_base):
                    for k in range(N_b):
                        ind = k+m*N_b+n*N_v_base*N_b+j*N_v_base*N_v_base*N_b+i*N_v_base*N_v_base*N_v_base*N_b
                        V_input[ind] = np.array([V_base[i], V_base[j], V_base[n], V_base[m], B[k]])
                        V_input_indexes_base[ind] = np.array([i, j, n, m])

    # for ind, prepared in enumerate(V_input):
    #     V_out[ind] = calc_out_velocities(prepared)
    new_N = 0
    for ind, prepared in enumerate(V_input):
        v_out = calc_out_velocities(prepared, 1)
        if v_out is not None:
            fitted = fit_to_base_mesh(dV_base, v_mean, v_max, v_out, V_input[ind], V_base)
            if fitted is not None:
                r = calculate_r_energy_koef(V_input[ind], V_base[fitted])
                if r is not None:
                    V_fit[new_N] = fitted
                    r_energy_koef[new_N] = r
                    V_input[new_N] = V_input[ind]
                    V_input_indexes_base[new_N] = V_input_indexes_base[ind]
                    new_N+=1

    return V_input, V_input_indexes_base, V_fit, r_energy_koef, new_N

def CALC_COLL_INT(F, N_integral, buffered_mesh, space_shape, dt, is_test=False, use_all_mesh=False):
    V_input, V_input_indexes_base, V_fit, r_energy_koef, new_N = buffered_mesh
    N_x, N_y = space_shape
    if is_test:
        I = np.zeros(shape=F.shape, dtype=F.dtype)
        dtau = 1
    else:
        I = F
        dtau = dt
    actual_N = new_N if use_all_mesh else N_integral

    for x in range(N_x):
        for y in range(N_y):
            input_samples = np.random.randint(0, new_N, size=actual_N)  # [i for i in range(new_N)] #

            for sample in input_samples:
                r = r_energy_koef[sample]
                C = 1 * np.linalg.norm(V_input[sample, :2] - V_input[sample, 2:4]) * dtau*10
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

    return I



if __name__ == '__main__':
    print("COLLISION INTEGRAL TESTS STARTED")




    t0 = time()

    N_integral=50000
    dt = 1


    N_x = N_y = 1
    N_v_base = 4
    N_b = 4
    v_cut = 1
    b_max = np.pi/4
    float_type = np.float32

    F = np.zeros((N_x, N_y, N_v_base, N_v_base), dtype=float_type)
    F[...] = 1 #np.random.rand(N_x, N_y, N_v_base, N_v_base)
    F[...,:3, :3] = 2
    F/=F.sum()
    print(F)

    buffered_mesh = prepare_collision_mesh(N_b, b_max, N_v_base, v_cut, float_type)
    CALC_COLL_INT(F, N_integral, buffered_mesh, (N_x, N_y), dt, is_test=False, use_all_mesh=True)
    print(F)



    print(f"CALC={time()-t0}")
    plot = sns.heatmap(F[0, 0], square=True, cbar=True)

    plot.invert_yaxis()
    plt.savefig("../plots/loh_ebaniy.png")
    plt.show()






