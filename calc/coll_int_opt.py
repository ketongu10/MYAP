import numpy as np
from numba import njit
from time import time
from init_distributions import set_maxwell
from multiprocessing import Pool


#@njit(cache=True)
def shind(ind, mid):
    i = int(ind)
    return (i+mid, i+mid-1) if i<0 else (i+mid-1, i+mid)


#@njit(cache=True)
def calc_coll_integral(F: np.ndarray, V: np.array, float_type):
    #ARGS
    bs = [-0.1, 0.1]
    db = 1
    dv = V[-1]-V[-2]
    V_sph = 4*V[-1]*V[-1] #???
    v_max = len(V)-1
    v_pos = int(len(V)/2)
    additional_args = [bs, dv, v_max, v_pos, float_type, F, V]

    #CREATING ARGS FOR POOL
    mesh_V = []
    for ix, vx in enumerate(V):
        for iy, vy in enumerate(V):
            mesh_V.append([(ix, iy), np.array([vx, vy]), additional_args])

    #POOL
    with Pool(8) as p:
        ret = p.map(calc_int_for_v, mesh_V)

    # INTEGRAL WE WANTED TO FIND
    I_total = np.zeros(F.shape, dtype=float_type)
    for i, (ids, vel, other) in enumerate(mesh_V):
        ix, iy = ids
        I_total[:, :, ix, iy] = ret[i]
    I_total *= db * 2 * np.pi * V_sph

    return I_total


def calc_int_for_v(args: list):
    [(ix, iy), v, (bs, dv, v_max, v_pos, float_typ, F, V)] = args
    #FUCKING NUMBA DOESN'T WORK WITH TUPLE FUUUUUUUCK!!!!!!!!
    return with_fucking_numba(ix, iy, v, bs, dv, v_max, v_pos, float_typ, F, V)

#@njit(cache=True)
def with_fucking_numba(ix, iy, v, bs, dv, v_max, v_pos, float_typ, F, V):
    I_for_v = np.zeros(F.shape[:2], dtype=float_typ)
    for jx, v1x in enumerate(V):
        for jy, v1y in enumerate(V):
            v1 = np.array([v1x, v1y])
            I = np.zeros(F.shape[:2], dtype=float_typ)
            if v[0] == v1[0] and v[1] == v1[1]:
                continue
            for b in bs:
                v_, v1_ = calc_out_velocities(v, v1, b, dv)
                v_ind, v1_ind = v_ / dv, v1_ / dv
                (v_indx_l, v_indx_m), (v_indy_l, v_indy_m) = shind(v_ind[0], v_pos), shind(v_ind[1], v_pos)
                (v1_indx_l, v1_indx_m), (v1_indy_l, v1_indy_m) = shind(v1_ind[0], v_pos), shind(v1_ind[1], v_pos)
                if (v_indx_l > v_max or v_indx_m > v_max or v_indy_l > v_max or v_indy_m > v_max or
                        v1_indx_l > v_max or v1_indx_m > v_max or v1_indy_l > v_max or v1_indy_m > v_max or
                        v_indx_l < 0 or v_indx_m < 0 or v_indy_l < 0 or v_indy_m < 0 or
                        v1_indx_l < 0 or v1_indx_m < 0 or v1_indy_l < 0 or v1_indy_m < 0):
                    continue
                v_less, v1_less = np.array([V[v_indx_l], V[v_indy_l]]), np.array([V[v1_indx_l], V[v1_indy_l]])
                v_more, v1_more = np.array([V[v_indx_m], V[v_indy_m]]), np.array([V[v1_indx_m], V[v1_indy_m]])
                E0 = v.dot(v) + v1.dot(v1)
                E1 = v_less.dot(v_less) + v1_less.dot(v1_less)
                E2 = v_more.dot(v_more) + v1_more.dot(v1_more)
                r = 0.5 if (E2 - E1) == 0 else (E0 - E1) / (E2 - E1)
                I += (np.power(F[:, :, v_indx_l, v_indy_l] * F[:, :, v1_indx_l, v1_indy_l], 1 - r) * \
                      np.power(F[:, :, v_indx_m, v_indy_m] * F[:, :, v1_indx_m, v1_indy_m], r) - \
                      F[:, :, ix, iy] * F[:, :, jx, jy]) * b
                # print(I, E0, E1, E2)
            I *= np.linalg.norm(v - v1)
            I_for_v += I
    return I_for_v







#@njit(cache=True)
def calc_out_velocities(v, v1, b_, dv=1):
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



    #print(f"beta={beta}")
    #print(f"S={_S}")
    #print(f"cos, sin= {ca, sa}")
    #print(f"_v, _v1 = {_v, _v1}")
    #print(f"v_, v1_ = {v_, v1_}")
    #print(f"ASSERT VELOCITY: {(v+v1)} =?= {(v_+v1_)}")
    #print(f"ASSERT ENERGY: {(v.dot(v)+v1.dot(v1))} =?= {(v_.dot(v_)+v1_.dot(v1_))}")
    return v_, v1_


#a = calc_out_velocities(np.array([1, -1]), np.array([-0.5, 20]), np.arcsin(-0.25))

if __name__ == '__main__':
    print("COLLISION INTEGRAL TESTS STARTED")
    dze_cut = 1
    N_v = 16
    V = np.linspace(-dze_cut, dze_cut, N_v)
    N_x = N_y = 100
    float_type = np.float64
    F = np.zeros((N_x, N_y, N_v, N_v), dtype=float_type)
    F = set_maxwell(F, V, 1)
    #F[0, 0, 4:8, 4:6] = 1

    F/=F.sum()
    t0 = time()
    print(calc_coll_integral(F, V, float_type))
    print(f"TIME= {time()-t0}")

