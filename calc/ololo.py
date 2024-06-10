from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy as np
import matplotlib.pyplot as plt
def maxwell(x, T):
    return x*x*np.exp(-x*x/2/T)
T = 2
vt = np.sqrt(T)
v_max = 4.8*vt


n = 100

a = np.linspace(-v_max, v_max, n)
w = maxwell(a, T)
print((a*a*w/3).sum()/w.sum())
plt.plot(w)
plt.show()