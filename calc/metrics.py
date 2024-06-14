import numpy as np


def temperature(F, VV, chip):
    h_chip, w_chip, x_start = chip
    T = np.ndarray(shape=(F.shape[0], F.shape[1]) ,dtype=np.float32)
    T = np.sum((F * VV), axis=(2, 3))
    T = T / np.sum(F, axis=(2,3))/2
    T[x_start:x_start+w_chip, 0:h_chip] = 0
    T = T.swapaxes(0, 1)
    gradT = np.gradient(T)
    gradT[0][0:h_chip,x_start-1:x_start+w_chip+1] = 0
    gradT[1][0:h_chip,x_start-1:x_start+w_chip+1] = 0

    gradT[0][h_chip,x_start:x_start+w_chip] = 0
    gradT[1][h_chip,x_start:x_start+w_chip] = 0
    #print(gradT[0][x_start, h_chip], gradT[1][x_start, h_chip])

    return T, gradT


def hydroV(F, V, Vxy):
    _F = F.swapaxes(0, 1)
    vvx = np.sum(_F*Vxy[0], axis=(2,3))
    vvy = np.sum(_F*Vxy[1], axis=(2,3))
    # Vx, Vy = np.ndarray(shape=F.shape[:2], dtype=F.dtype), np.ndarray(shape=F.shape[:2], dtype=F.dtype)
    # for i,vx in enumerate(V):
    #     for j, vy in enumerate(V):
    #         Vx+=F[:,:,i,j]*vx
    #         Vy += F[:, :, i, j] * vy
    #
    # print(vvx[0,0], vvy[0,0], Vx[0,0], Vy[0,0])

    #print(vvx[:,-1])
    #print(vvx[:,0])
    return (vvx,vvy)

def dQ(F, VV, v_pos):
    Q_out = (F[-1,:,v_pos:,:]*VV[v_pos:,:]).sum()
    Q_in = (F[0,:,v_pos:,:]*VV[v_pos:,:]).sum()
    return Q_out-Q_in

def dQ_on_chip(F, hydroV, VV, V, chip, u):
    h_chip, w_chip, x_start = chip

    q = 0
    for x in range(x_start-1, x_start+w_chip+1):
        for i, vy in enumerate(V):
            q += (F[x, h_chip+1, :, i] * (VV[:, i] - u * u)).sum() * vy / 2
        # q += (F[x, h_chip+1]*(VV-u*u)).sum()*hydroV[1][h_chip+1, x]/2

    for y in range(0, h_chip):
        for i, vx in enumerate(V):
            q += -(F[x_start - 1, y, i,:] * (VV[i,:] - u * u)).sum() * vx / 2
            q += (F[x_start+w_chip+1, y, i, :] * (VV[i, :] - u * u)).sum() * vx / 2
        # q += (F[x_start-1, y]*(VV-u*u)).sum()*hydroV[0][y,x_start-1]*(-1)/2
        # q += (F[x_start+w_chip+1, y]*(VV-u*u)).sum()*hydroV[0][y,x_start+w_chip+1]/2

    print("vH 50, hchip+1", hydroV[0][h_chip+1,50], hydroV[1][h_chip+1,50])
    print("vH x_start-1, 2", hydroV[0][2,x_start-1], hydroV[1][2,x_start-1])
    print("vH x_start+w_chip+1, 2", hydroV[0][2,x_start+w_chip+1], hydroV[1][2,x_start+w_chip+1])
    return q


