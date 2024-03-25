
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def visualize(stl_milp, sampling_time):
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]
    
    stl_px = [var.x for var in stl_milp.variables['px'].values()]
    stl_py = [var.x for var in stl_milp.variables['py'].values()]
    stl_vx = [var.x for var in stl_milp.variables['vx'].values()]
    stl_vy = [var.x for var in stl_milp.variables['vy'].values()]
    stl_u_ax = [var.x for var in stl_milp.variables['u_ax'].values()]
    stl_u_ay = [var.x for var in stl_milp.variables['u_ay'].values()]
    stl_jx = [var.x for var in stl_milp.variables['jx'].values()]
    stl_jy = [var.x for var in stl_milp.variables['jy'].values()]

    x_ped = [var.x for var in stl_milp.variables['x_ped'].values()]
    y_ped = [var.x for var in stl_milp.variables['y_ped'].values()]
    
    fig, axs = plt.subplots(4, 2, figsize=(10, 7))
    fig.suptitle('STL-Control Synthesis')

    axs[0][0].plot(t, stl_px, '-r', label=r'x', 
                   linewidth=3, marker='s', markersize=7)     
    axs[0][0].plot(t, x_ped, '-b', label=r'x_ped', 
                   linewidth=3, marker='s', markersize=7)        
    axs[0][0].set_title('x vs t')
    axs[0][0].grid()
    axs[0][0].legend(prop={'size': 10})
    axs[0][0].xaxis.set_tick_params(labelsize=12)
    axs[0][0].tick_params(labelsize=10)

    axs[0][1].plot(t, stl_py, '-r', label=r'y', 
                   linewidth=3, marker='s', markersize=7)
    axs[0][1].plot(t, y_ped, '-b', label=r'y_ped',
                     linewidth=3, marker='s', markersize=7)
    axs[0][1].set_title('y vs t')
    axs[0][1].grid()
    axs[0][1].legend(prop={'size': 10})
    axs[0][1].tick_params(labelsize=10)

    axs[1][0].plot(t, stl_vx, '-r', label=r'vx', 
                   linewidth=3, marker='s', markersize=7)
    axs[1][0].set_title('v_x vs t')
    axs[1][0].grid()
    axs[1][0].legend(prop={'size': 10})
    axs[1][0].tick_params(labelsize=10)

    axs[1][1].plot(t, stl_vy, '-r', label=r'vy', 
                   linewidth=3, marker='s', markersize=7)
    axs[1][1].set_title('v_y vs t')
    axs[1][1].grid()
    axs[1][1].legend(prop={'size': 10})
    axs[1][1].tick_params(labelsize=10)

    axs[2][0].plot(t[0:-1] , stl_u_ax, '-r', label=r'ax', 
                   linewidth=3, marker='s', markersize=7)
    axs[2][0].set_title('a_x vs t')
    axs[2][0].grid()
    axs[2][0].legend(prop={'size': 10})
    axs[2][0].tick_params(labelsize=10)

    axs[2][1].plot(t[0:-1], stl_u_ay, '-r', label=r'ay', 
                   linewidth=3, marker='s', markersize=7)
    axs[2][1].set_title('a_y vs t')
    axs[2][1].grid()
    axs[2][1].legend(prop={'size': 10})
    axs[2][1].tick_params(labelsize=10)

    axs[3][0].plot(t[0:-1], stl_jx, '-r', label=r'jx', 
                   linewidth=3, marker='s', markersize=7)
    axs[3][0].set_title('j_x vs t')
    axs[3][0].grid()
    axs[3][0].legend(prop={'size': 10})
    axs[3][0].tick_params(labelsize=10)

    axs[3][1].plot(t[0:-1], stl_jy, '-r', label=r'jy',
                     linewidth=3, marker='s', markersize=7)
    axs[3][1].set_title('j_y vs t')
    axs[3][1].grid()
    axs[3][1].legend(prop={'size': 10})
    axs[3][1].tick_params(labelsize=10)
    fig.tight_layout()
    plt.show()

def visualize2d(stl_milp, sampling_time):
    # create 2d plot of the pedestrian and the car
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]

    stl_px = [var.x for var in stl_milp.variables['px'].values()]
    stl_py = [var.x for var in stl_milp.variables['py'].values()]
    x_ped = [var.x for var in stl_milp.variables['x_ped'].values()]
    y_ped = [var.x for var in stl_milp.variables['y_ped'].values()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('STL-Control Synthesis')

    ax.plot(stl_px, stl_py, '-r', label=r'car',
            linewidth=3, marker='s', markersize=7)
    ax.plot(x_ped, y_ped, '-b', label=r'pedestrian',
            linewidth=3, marker='s', markersize=7)
    ax.set_title('x vs y')
    ax.grid()
    ax.legend(prop={'size': 10})
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    plt.show()


# Create an animation that advances with time of the trajectory of the car and the pedestrian
def visualize_animation(stl_milp, sampling_time, weight_list=None):
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]

    stl_px = [var.x for var in stl_milp.variables['px'].values()]
    stl_py = [var.x for var in stl_milp.variables['py'].values()]
    x_ped = [var.x for var in stl_milp.variables['x_ped'].values()]
    y_ped = [var.x for var in stl_milp.variables['y_ped'].values()]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('STL-Control Synthesis')
    horizon = t[-1]/sampling_time
    ax.set_title(f'x vs y, horizon: {int(horizon)}, robustness: {stl_milp.variables[stl_milp.formula][0][1].x:.2f}')
    ax.grid()
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plot a green square at setpoint
    ax.plot(stl_px[-1], stl_py[-1], 'gs', label=r'setpoint')
    # plot a black line at sidewalk
    ax.axhline(y=0, color='k', linestyle='-', label='sidewalk')
    ax.axhline(y=5, color='k', linestyle='-', label='sidewalk')
    ax.axhline(y=2.5, color='k', linestyle='--', label='sidewalk')

    line, = ax.plot(stl_px, stl_py, '-r', label=r'car', linewidth=3, marker='s', markersize=7)
    line2, = ax.plot(x_ped, y_ped, '-b', label=r'pedestrian',linewidth=3, marker='s', markersize=7)

    # Show initial and final coordinates for pedestrian by adding text of (x,y) at the points
    ax.plot(x_ped[0], y_ped[0], 'bo', label='initial pedestrian')
    ax.plot(x_ped[-1], y_ped[-1], 'bo', label='final pedestrian')
    ax.text(x_ped[0], y_ped[0], f'({x_ped[0]:.2f},{y_ped[0]:.2f})', fontsize=8)
    ax.text(x_ped[-1], y_ped[-1], f'({x_ped[-1]:.2f},{y_ped[-1]:.2f})', fontsize=8)

    if weight_list is not None:
        # print weights
        ax.text(0.05, 0.95, f'w1: {weight_list[0]} \n w2: {weight_list[1]} \n p1: {weight_list[2]} \n p2: {weight_list[3]}', 
                transform=ax.transAxes, fontsize=6, verticalalignment='top')

    def animate(i):
        line.set_xdata(stl_px[:i])
        line.set_ydata(stl_py[:i])
        line2.set_xdata(x_ped[:i])
        line2.set_ydata(y_ped[:i])
        return line, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=100, repeat=False)
    plt.show()
    return ani

def plot_var(stl_milp, var: str, sampling_time):
    t = [k * sampling_time for k in stl_milp.variables[var].keys()]
    stl_var = [var.x for var in stl_milp.variables[var].values()]

    fig_var, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title('Variable vs time')
    ax.plot(t, stl_var, '-r', label=var)
    ax.grid()
    ax.legend(prop={'size': 10})
    fig_var.tight_layout()
    plt.show()


# save "visualize_animation" as video
def save_vid(ani, filename):
    # Check if filename exists in the system, otherwise add _1, _2, etc. before the extension
    if os.path.exists(filename):
        i = 0
        while os.path.exists(filename):
            i += 1
            filename_parts = filename.split('.')
            # check if filename has a _1, _2 already
            if filename_parts[-2][-2] == '_':
                filename_parts[-2] = filename_parts[-2][:-2] + f'_{i}'
            else:
                filename_parts[-2] = filename_parts[-2] + f'_{i}'
            filename = '.'.join(filename_parts)
    ani.save(filename, writer='Pillow', fps=10)
    print(f'Video saved as {filename}')
    plt.show()