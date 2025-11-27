import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from tqdm import tqdm
import FD, LBM

# Current simulation data:
L = 200
Nx, Ny = 200, 200
T_end = 50000

# u_arr = np.zeros((500, 200, 200, 2))
def vorticity(u_arr, L, Nx, Ny):
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    w_arr = np.zeros((u_arr.shape[:-1]))
    for t in range(w_arr.shape[0]): #iterate over all 500 times. u_arr[t] has shape (Nx, Ny, 2).
        u_x = u_arr[t][:, :, 0]
        u_y = u_arr[t][:, :, 1]    
        w1 = np.zeros((Nx, Ny))
        w2 = np.zeros((Nx, Ny))
        for i in range(Nx):
            w1[:, i] = FD.finiteDifferences(u_y[:, i], x)
            w2[i, :] = FD.finiteDifferences(u_x[i, :], y)
        w_arr[t] = w1 - w2

    return w_arr

def makeUani(u_arr, t_arr, L, Nx, Ny, save = 1):
    '''
    u_arr must be the complete (T, Nx, Ny, 2)-dim array containing all velocity fields from a simulation.
    Method is identical as in the LBM class. 
    '''
    snaps_u = u_arr
    times = t_arr
    x = np.linspace(0, L, Nx)[::8]
    y = np.linspace(0, L, Ny)[::8]
    xx, yy = np.meshgrid(x, y, indexing="ij") 
    interval = times[1] - times[0]

    fig, ax = plt.subplots(figsize = (6,6))
    ax.set_title(f"Velocity field u, t = {times[0]}")

    u_init = snaps_u[0]
    speed = np.sqrt(u_init[::8, ::8, 0]**2 + u_init[::8, ::8, 1]**2)
    im = ax.quiver(xx, yy, u_init[::8, ::8, 0], u_init[::8, ::8, 1], speed, cmap = "cividis",
                       scale=3,          # smaller → larger arrows
                        width=0.005,       # thicker shaft
                        headwidth=4,       # wider arrowhead
                        headlength=6,      # longer arrowhead
                        headaxislength=4.5,
                        minlength=0.5,     # remove tiny noisy arrows
                        pivot='mid',       # center the arrows
                        alpha=0.9 ) # 
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)

    def update(i):
        u = snaps_u[i]
        speed = np.sqrt(u[::8, ::8, 0]**2 + u[::8, ::8, 1]**2)

        im.set_UVC(u[::8, ::8, 0], u[::8, ::8, 1], speed)
        ax.set_title(f"Velocity field u, t = {times[i]}")

    ani = FuncAnimation(fig, update, frames = len(snaps_u), interval = interval)
    if save:
        ani.save("u_snaps.gif", fps = "24", dpi = 200)

    plt.show()
    plt.close()

def makeWani(w_arr, t_arr, L, save = 1):
    snaps_w = w_arr
    times = t_arr
    interval = times[1] - times[0]

    fig, ax = plt.subplots(figsize = (6, 5))
    im = ax.imshow(
        snaps_w[0].T, #transpose to get correct axes?
        origin = "lower",
        cmap = "cividis",
        extent = [0, L, 0, L],
        vmin = -np.max(snaps_w[0]),
        vmax = np.max(snaps_w[0])
    )
    fig.colorbar(im, ax = ax)
    ax.set_title(f"t = {times[0]}")

    def update(i):
        im.set_data(snaps_w[i].T)
        ax.set_title(f"t = {times[i]}")
        return [im]
    
    ani = FuncAnimation(fig, update, frames = len(snaps_w), interval = interval)
    if save:
        ani.save("w_snaps.gif", fps = "24", dpi = 200)
    
    plt.show()
    plt.close()



test1 = LBM.LBM(200, 200, 0.001, 200, 0.01, debug = 0) 
interval = 100
test1.fullSim(2000, diagnostic = 1, plot = ["C"], interval = interval, findEta = 1)

# index_end = 10000
# T_end = 50000
# u_arr = np.load("u_array50k.npy")
# u_arr = u_arr[:index_end, ...]
# times_arr = np.arange(0, T_end, 100)
# # makeUani(u_arr, times_arr, L, Nx, Ny)

# # w_arr = vorticity(u_arr, L, Nx, Ny)
# # makeWani(w_arr, times_arr, L)

# Nx, Ny = 13, 12
# ta = np.ones((Nx, Ny))
# ta[7,8] = 0
# print(np.argmin(ta))
# ind = np.argmin(ta)
# print(ta[ind // Nx, ind % Ny])