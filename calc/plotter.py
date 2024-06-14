import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re



def save_slide(F, name):
    data = np.sum(F, axis=(2, 3)).swapaxes(0, 1)
    plot = sns.heatmap(data, square=True, cbar=False)
    plot.invert_yaxis()




def animate(i, F, dt, ax, max):
    print(f"Plotting {i} frame")
    ax.cla()
    data = np.sum(F[i*dt], axis=(2,3)).swapaxes(0, 1)
    plot = sns.heatmap(data, square=True,vmax=max, cbar=False)
    plot.invert_yaxis()

def animate_relax(i, F, dt, ax, max):
    print(f"Plotting {i} frame")
    ax.cla()
    data = F[i] #.swapaxes(0, 1)
    print(data)
    plot = sns.heatmap(data, square=True,vmax=max, cbar=False)
    plot.invert_yaxis()


inited = False
def animate_with_T(i, F,T, dt, ax1, ax2, max_f, max_t):
    global inited
    x, y = F[i].shape[0:2]
    print(f"Plotting {i} frame")
    ax1.cla()
    ax2.cla()
    data = np.sum(F[i*dt], axis=(2,3)).swapaxes(0, 1)
    ax1.title.set_text(f"f={np.sum(data):.2f}")
    ax2.title.set_text(f"T={np.sum(T[i][0])/x/y:.2f}")
    plotF = sns.heatmap(data, square=True,  cbar=(not inited), ax=ax1) #vmax=max_f, vmin=max_f*0.5,
    plotF.invert_yaxis()

    plotT = sns.heatmap(T[i][0], square=True, cbar=(not inited), ax=ax2) #
    plotT.invert_yaxis()

    if not inited:
        inited = True


def animate_grads(i, F,T, dt, ax1,ax2, max_f, max_t):

    print(f"Plotting {i} grad")

    ax1.cla()
    ax2.cla()

    ax1.title.set_text(f"grad(f)")
    ax2.title.set_text(f"grad(T)")
    data = np.sum(F[i * dt], axis=(2, 3)).swapaxes(0, 1)
    gr_f = np.gradient(data)

    plotF = ax1.quiver(gr_f[1], gr_f[0], data, cmap='inferno')
    #ax1.invert_yaxis()
    plotT = ax2.quiver(T[i][1][1], T[i][1][0], T[i][0], cmap='inferno')
    #ax2.invert_yaxis()

inited = False
def animate_all(i, F,T, dt, axs, max_f, max_t, chip=None):
    global inited
    x, y = F[i].shape[0:2]
    print(f"Plotting {i} frame")
    for ax in axs:
        ax.cla()
    data = np.sum(F[i*dt], axis=(2,3)).swapaxes(0, 1)
    axs[0].title.set_text(f"f={np.sum(data):.2f}")
    axs[1].title.set_text(f"T={np.sum(T[i][0])/x/y:.2f}")
    axs[2].title.set_text(f"grad(f)")
    axs[3].title.set_text(f"grad(T)")

    plotF = sns.heatmap(data, square=True,  cbar=(not inited), ax=axs[0]) #vmax=max_f, vmin=max_f*0.5,
    plotF.invert_yaxis()

    plotT = sns.heatmap(T[i][0], square=True, cbar=(not inited), ax=axs[1]) #
    plotT.invert_yaxis()

    gr_f = np.gradient(data)
    if chip is not None:
        h_chip, w_chip, x_start = chip
        gr_f[0][0:h_chip, x_start - 1:x_start + w_chip + 1] = 0
        gr_f[1][0:h_chip, x_start - 1:x_start + w_chip + 1] = 0

        gr_f[0][h_chip, x_start:x_start + w_chip] = 0
        gr_f[1][h_chip, x_start:x_start + w_chip] = 0

    gradF = axs[2].quiver(gr_f[1], gr_f[0], data, cmap='inferno')
    gradT = axs[3].quiver(T[i][1][1], T[i][1][0], T[i][0], cmap='inferno')


    if not inited:
        inited = True

def animate_hydro(i, F, H, dt, ax1, ax2, max_f, max_t):

    print(f"Plotting {i} grad")

    ax1.cla()
    ax2.cla()

    ax1.title.set_text(f"grad(f)")
    ax2.title.set_text(f"grad(T)")
    data = np.sum(F[i * dt], axis=(2, 3)).swapaxes(0, 1)
    gr_f = np.gradient(data)

    plotF = ax1.quiver(gr_f[1], gr_f[0], data, cmap='inferno')
    plotT = ax2.quiver(H[i][0], H[i][1], cmap='inferno')

def render_animation(F, dt=1, temp=None, max_f=1, max_t=1):
    if temp:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        anim = animation.FuncAnimation(fig, animate_with_T, fargs=(F, temp, dt, ax1, ax2, max_f, max_t), frames=len(F)//dt, repeat=False)
    else:
        fig, ax1 = plt.subplots()
        anim = animation.FuncAnimation(fig, animate, fargs=(F, dt, ax1, max_f), frames=len(F)//dt, repeat=False)
    anim.save("../plots/tempru.gif", writer="ffmpeg")


def render_grads_animation(F, dt=1, temp=None, max_f=1, max_t=1):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,9))
    anim = animation.FuncAnimation(fig, animate_grads, fargs=(F, temp, dt, ax1, ax2, max_f, max_t), frames=len(F)//dt, repeat=False)
    anim.save("../plots/temp_gradru.gif", writer="ffmpeg")


def render_hydro_animation(F, dt=1, hydro=None, max_f=1, max_h=1):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,9))
    anim = animation.FuncAnimation(fig, animate_hydro, fargs=(F, hydro, dt, ax1, ax2, max_f, max_h), frames=len(F)//dt, repeat=False)
    anim.save("../plots/temp_hydroru.gif", writer="ffmpeg")




def render_animation_all(F, dt=1, temp=None, max_f=1, max_t=1, filename="last", chip=None):
    if temp:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(32, 16))
        axs = (ax1, ax2, ax3, ax4)
        print(axs)
        anim = animation.FuncAnimation(fig, animate_all, fargs=(F, temp, dt, axs, max_f, max_t, chip), frames=len(F)//dt, repeat=False)
        anim.save(f"../plots/{filename}/anim.gif", writer="ffmpeg")
        fig.savefig(f"../plots/{filename}/last_frame.png")
    else:
        fig, ax1 = plt.subplots()
        anim = animation.FuncAnimation(fig, animate, fargs=(F, dt, ax1, max_f), frames=len(F)//dt, repeat=False)
        anim.save(f"../plots/{filename}.gif", writer="ffmpeg")
        fig.savefig(f"../plots/{filename}.png")

def plot_last_vH(vH, filename):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.cla()
    hydro = ax.quiver(vH[-1][1], vH[-1][0], cmap='inferno')
    fig.savefig(f"../plots/{filename}/vH.png")


def plot_dQ(dQs, filename):
    print(dQs)
    fig, ax = plt.subplots()
    ax.cla()
    ax.plot(dQs)
    fig.savefig(f"../plots/{filename}/dQ.png")

if __name__ == "__main__":
    dQs = []


    for dir in os.listdir("../plots/NEW_T"):
        if "T=1.5" in dir:
            print(dir)
            dQs.append(np.load(os.path.join("../plots/NEW_T", dir, "dQ.npy")))
    #dQs[0] -= dQs[1]
    #print(dQs)
            patternT = r"_T=([0-9.]+)"
            patternu = r"_u=([0-9.]+)"
            match_T = re.search(patternT, dir)
            match_u = re.search(patternu, dir)
            plt.plot(dQs[-1], label=f"T={float(match_T.group(1))}  u={float(match_u.group(1))}")
    plt.title("Уносимое тепло от скорости потока")
    plt.xlabel("time, steps")
    plt.ylabel("Q")
    plt.legend()
    plt.savefig("../plots/last.png")
    plt.show()

