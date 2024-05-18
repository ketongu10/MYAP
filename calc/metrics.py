import numpy as np


def temperature(F, VV):
    T = np.ndarray(shape=(F.shape[0], F.shape[1]) ,dtype=np.float32)
    T = np.sum((F * VV), axis=(2, 3))
    T = T / np.sum(F, axis=(2,3))
    T = T.swapaxes(0, 1)
    gradT = np.gradient(T)

    return T, gradT


def hydroV(F, Vxy):
    _F = F.swapaxes(0, 1)
    vx = np.sum(_F*Vxy[0], axis=(2,3))
    vy = np.sum(_F*Vxy[1], axis=(2,3))
    return (vy, vx)

