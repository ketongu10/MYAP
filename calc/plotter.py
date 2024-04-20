import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation




def save_slide(F, name):
    data = np.sum(F, axis=(2, 3)).swapaxes(0, 1)
    plot = sns.heatmap(data, square=True, cbar=False)
    plot.invert_yaxis()


def init(F, dt, ax1, max):
      sns.heatmap(np.zeros((100, 100)), vmax=max, square=True, cbar=True)

def animate(i, F, dt, ax, max):
    print(f"Plotting {i} frame")
    ax.cla()
    data = np.sum(F[i*dt], axis=(2,3)).swapaxes(0, 1)
    plot = sns.heatmap(data, square=True,vmax=max, cbar=False)
    plot.invert_yaxis()

def animate_with_T(i, F,T, dt, ax1, ax2, max):
    print(f"Plotting {i} frame")
    ax1.cla()
    ax2.cla()
    data = np.sum(F[i*dt], axis=(2,3)).swapaxes(0, 1)
    plotF = sns.heatmap(data, square=True, vmax=max, cbar=False, ax=ax1)
    plotF.invert_yaxis()
    plotT = sns.heatmap(T[i][0], square=True, vmax=max, cbar=False, ax=ax2)
    plotT.invert_yaxis()

def render_animation(F, dt=1, temp=None, vmax=1):
    if temp:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        anim = animation.FuncAnimation(fig, animate_with_T, fargs=(F, temp, dt, ax1, ax2, vmax), frames=len(F)//dt, repeat=False)
    else:
        fig, ax1 = plt.subplots()
        anim = animation.FuncAnimation(fig, animate,init_func=init, fargs=(F, dt, ax1, vmax), frames=len(F)//dt, repeat=False)
    anim.save("../plots/temp.gif", writer="ffmpeg")

