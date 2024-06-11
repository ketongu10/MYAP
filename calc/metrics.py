import numpy as np


def temperature(F, VV, chip):
    h_chip, w_chip, x_start = chip
    T = np.ndarray(shape=(F.shape[0], F.shape[1]) ,dtype=np.float32)
    T = np.sum((F * VV), axis=(2, 3))
    T = T / np.sum(F, axis=(2,3))
    T[x_start:x_start+w_chip, 0:h_chip] = 0
    T = T.swapaxes(0, 1)
    gradT = np.gradient(T)
    gradT[0][0:h_chip,x_start-1:x_start+w_chip+1] = 0
    gradT[1][0:h_chip,x_start-1:x_start+w_chip+1] = 0

    gradT[0][h_chip,x_start:x_start+w_chip] = 0
    gradT[1][h_chip,x_start:x_start+w_chip] = 0
    #print(gradT[0][x_start, h_chip], gradT[1][x_start, h_chip])

    return T, gradT


def hydroV(F, Vxy):
    _F = F.swapaxes(0, 1)
    vx = np.sum(_F*Vxy[0], axis=(2,3))
    vy = np.sum(_F*Vxy[1], axis=(2,3))
    return (vy, vx)

def dQ(F, VV, v_pos):
    Q_out = (F[-1,:,v_pos:,:]*VV[v_pos:,:]).sum()
    Q_in = (F[0,:,v_pos:,:]*VV[v_pos:,:]).sum()
    return Q_out-Q_in

def dQ_on_chip(F, hydroV, VV, chip):
    h_chip, w_chip, x_start = chip
    q = 0
    for x in range(x_start-1, x_start+w_chip+1):
        q += (F[x, h_chip+2]*VV).sum()*np.linalg.norm([hydroV[0][h_chip+2,x], hydroV[1][h_chip+2,x]])

    for y in range(0, h_chip):
        q += (F[x_start-2, y]*VV).sum()*np.linalg.norm([hydroV[0][y,x_start-2], hydroV[1][y,x_start-2]])
        q += (F[x_start+w_chip+2, y]*VV).sum()*np.linalg.norm([hydroV[0][y,x_start+w_chip+2], hydroV[1][y,x_start+w_chip+2]])
    return q


