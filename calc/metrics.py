import numpy as np


def temperature(F, VV):
    T = np.ndarray(shape=(F.shape[0], F.shape[1]) ,dtype=np.float32)
    T = np.sum((F * VV), axis=(2, 3)).swapaxes(0, 1)
    gradT = np.gradient(T)
    return T, gradT


def dQ(F1, F2):
    pass

