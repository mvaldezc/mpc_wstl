
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def visualize(stl_milp, sampling_time):
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]
    
    px = [var.x for var in stl_milp.variables['px'].values()]
    py = [var.x for var in stl_milp.variables['py'].values()]
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

    axs[0][0].plot(t, px, '-r', label=r'x', 
                   linewidth=3, marker='s', markersize=7)     
    axs[0][0].plot(t, x_ped, '-b', label=r'x_ped', 
                   linewidth=3, marker='s', markersize=7)        
    axs[0][0].set_title('x vs t')
    axs[0][0].grid()
    axs[0][0].legend(prop={'size': 10})
    axs[0][0].xaxis.set_tick_params(labelsize=12)
    axs[0][0].tick_params(labelsize=10)

    axs[0][1].plot(t, py, '-r', label=r'y', 
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

    px = [var.x for var in stl_milp.variables['px'].values()]
    py = [var.x for var in stl_milp.variables['py'].values()]
    x_ped = [var.x for var in stl_milp.variables['x_ped'].values()]
    y_ped = [var.x for var in stl_milp.variables['y_ped'].values()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('STL-Control Synthesis')

    ax.plot(px, py, '-r', label=r'car',
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
def visualize_animation(stl_milp, sampling_time, weight_list=None, carla=False):
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]

    px = [var.x for var in stl_milp.variables['px'].values()]
    py = [var.x for var in stl_milp.variables['py'].values()]
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
    ax.plot(px[-1], py[-1], 'gs', label=r'setpoint')
    # plot a black line at sidewalk
    if not carla:
        ax.axhline(y=0, color='k', linestyle='-', label='sidewalk')
        ax.axhline(y=5, color='k', linestyle='-')
        ax.axhline(y=2.5, color='k', linestyle='--', label='sidewalk')
    else:
        ax.axhline(y=0, color='k', linestyle='-')
        ax.axhline(y=1, color='k', linestyle='-')
        ax.axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
        ax.axhline(y=7.95, color='k', linestyle='-')
        ax.axhline(y=8.05, color='k', linestyle='-')
        ax.axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
        ax.axhline(y=15, color='k', linestyle='-')
        ax.axhline(y=16, color='k', linestyle='-')

    line, = ax.plot(px, py, '-r', label=r'car', linewidth=3, marker='s', markersize=7)
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
        line.set_xdata(px[:i])
        line.set_ydata(py[:i])
        line2.set_xdata(x_ped[:i])
        line2.set_ydata(y_ped[:i])
        return line, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(t)+1, interval=100, repeat=False)
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

def plot_multi_vars(stl_milp, var_list: list, sampling_time):
    t = [k * sampling_time for k in stl_milp.variables[var_list[0]].keys()]

    fig_var, axs = plt.subplots(len(var_list), 1, figsize=(5, 8))
    fig_var.suptitle('STL-Control Synthesis')

    for i, var in enumerate(var_list):
        stl_var = [var.x for var in stl_milp.variables[var].values()]
        axs[i].set_title(f'{var} vs time')
        axs[i].plot(t, stl_var, '-r', label=var)
        axs[i].grid()
        axs[i].legend(prop={'size': 10})
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


# Create an animation that advances with time of the trajectory of the car and the pedestrian
def visualize_demonstration(px, py, x_ped, y_ped, t):

    horizon = t.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('STL-Control Synthesis')
    ax.set_title(f'x vs y, horizon: {int(horizon)}, Time: {t[-1]:.2f}')
    ax.grid()
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plot a green square at setpoint
    ax.plot(px[-1], py[-1], 'gs', label=r'setpoint')
    # plot a black line at sidewalk
    ax.axhline(y=0, color='k', linestyle='-')
    ax.axhline(y=1, color='k', linestyle='-')
    ax.axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=7.95, color='k', linestyle='-')
    ax.axhline(y=8.05, color='k', linestyle='-')
    ax.axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=15, color='k', linestyle='-')
    ax.axhline(y=16, color='k', linestyle='-')

    line, = ax.plot(px, py, '-r', label=r'car', linewidth=3, marker='s', markersize=7)
    line2, = ax.plot(x_ped, y_ped, '-b', label=r'pedestrian',linewidth=3, marker='s', markersize=7)

    # Show initial and final coordinates for pedestrian by adding text of (x,y) at the points
    ax.plot(x_ped[0], y_ped[0], 'bo', label='initial pedestrian')
    ax.plot(x_ped[-1], y_ped[-1], 'bo', label='final pedestrian')
    ax.text(x_ped[0], y_ped[0], f'({x_ped[0]:.2f},{y_ped[0]:.2f})', fontsize=8)
    ax.text(x_ped[-1], y_ped[-1], f'({x_ped[-1]:.2f},{y_ped[-1]:.2f})', fontsize=8)

    def animate(i):
        line.set_xdata(px[:i])
        line.set_ydata(py[:i])
        line2.set_xdata(x_ped[:i])
        line2.set_ydata(y_ped[:i])
        return line, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(t)+1, interval=100, repeat=False)
    plt.show()
    return ani

def visualize_demo_and_stl(px_demo, py_demo, stl_milp, sampling_time):
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]

    px = [var.x for var in stl_milp.variables['px'].values()]
    py = [var.x for var in stl_milp.variables['py'].values()]
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
    ax.plot(px[-1], py[-1], 'ms', label=r'setpoint')
    # plot a black line at sidewalk
    ax.axhline(y=0, color='k', linestyle='-')
    ax.axhline(y=1, color='k', linestyle='-')
    ax.axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=7.95, color='k', linestyle='-')
    ax.axhline(y=8.05, color='k', linestyle='-')
    ax.axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=15, color='k', linestyle='-')
    ax.axhline(y=16, color='k', linestyle='-')

    line, = ax.plot(px, py, '-r', label=r'car', linewidth=3, marker='s', markersize=7, alpha = 0.7)
    line2, = ax.plot(x_ped, y_ped, '-b', label=r'pedestrian',linewidth=3, marker='s', markersize=7, alpha = 0.6)
    line3, = ax.plot(px_demo, py_demo, '-g', label=r'demonstration',linewidth=3, marker='s', markersize=7, alpha = 0.3)

    # Show initial and final coordinates for pedestrian by adding text of (x,y) at the points
    ax.plot(x_ped[0], y_ped[0], 'bo')
    ax.plot(x_ped[-1], y_ped[-1], 'bo')
    ax.text(x_ped[0], y_ped[0], f'({x_ped[0]:.2f},{y_ped[0]:.2f})', fontsize=8)
    ax.text(x_ped[-1], y_ped[-1], f'({x_ped[-1]:.2f},{y_ped[-1]:.2f})', fontsize=8)

    def animate(i):
        line.set_xdata(px[:i])
        line.set_ydata(py[:i])
        line2.set_xdata(x_ped[:i])
        line2.set_ydata(y_ped[:i])
        line3.set_xdata(px_demo[:i])
        line3.set_ydata(py_demo[:i])
        return line, line2, line3

    ani = animation.FuncAnimation(fig, animate, frames=len(t)+1, interval=100, repeat=False)
    plt.legend()
    plt.show()
    return ani


def visualize_mpc(stl_milps, px, py, x_ped, y_ped, t, px_demo, py_demo):
    horizon = t.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('STL-Control Synthesis')
    ax.set_title(f'x vs y, horizon: {int(horizon)}, Time: {t[-1]:.2f}')
    ax.grid()
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plot a green square at setpoint
    ax.plot(px[-1], py[-1], 'gs', label=r'setpoint')
    # plot a black line at sidewalk
    ax.axhline(y=0, color='k', linestyle='-')
    ax.axhline(y=1, color='k', linestyle='-')
    ax.axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=7.95, color='k', linestyle='-')
    ax.axhline(y=8.05, color='k', linestyle='-')
    ax.axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=15, color='k', linestyle='-')
    ax.axhline(y=16, color='k', linestyle='-')

    px_list = []
    py_list = []
    # px_ped_list = []
    # py_ped_list = []
    for i in range(len(stl_milps)):
        px_i = [var.x for var in stl_milps[i].variables['px'].values()]
        py_i = [var.x for var in stl_milps[i].variables['py'].values()]
        px_list.append(px_i)
        py_list.append(py_i)
        # px_ped_i = [var.x for var in stl_milps[i].variables['x_ped'].values()]
        # py_ped_i = [var.x for var in stl_milps[i].variables['y_ped'].values()]
        # px_ped_list.append(px_ped_i)
        # py_ped_list.append(py_ped_i)
    line4, = ax.plot(px_list[0], py_list[0], '-m', linewidth=2, marker='s', markersize=7, alpha = 0.2)
    line2, = ax.plot(x_ped, y_ped, '-b', label=r'pedestrian',linewidth=3, marker='s', markersize=7)
    line3, = ax.plot(px_demo, py_demo, '-g', label=r'demonstration',linewidth=3, marker='s', markersize=7, alpha = 1.0)
    line, = ax.plot(px, py, '-r', label=r'car', linewidth=3, marker='s', markersize=7, alpha = 0.8)
    # line5, = ax.plot(px_ped_list[0], py_ped_list[0], '-b', linewidth=2, marker='s', markersize=7, alpha = 0.1)


    # Show initial and final coordinates for pedestrian by adding text of (x,y) at the points
    ax.plot(x_ped[0], y_ped[0], 'bo', label='initial pedestrian')
    ax.plot(x_ped[-1], y_ped[-1], 'bo', label='final pedestrian')
    ax.text(x_ped[0], y_ped[0], f'({x_ped[0]:.2f},{y_ped[0]:.2f})', fontsize=8)

    def animate(i):
        if i < len(px_list):
            line4.set_xdata(px_list[i])
            line4.set_ydata(py_list[i])
            # line5.set_xdata(px_ped_list[i])
            # line5.set_ydata(py_ped_list[i])
        line2.set_xdata(x_ped[:i])
        line2.set_ydata(y_ped[:i])
        line3.set_xdata(px_demo[:i])
        line3.set_ydata(py_demo[:i])
        line.set_xdata(px[:i])
        line.set_ydata(py[:i])
        return line, line2, line3
    
    ani = animation.FuncAnimation(fig, animate, frames=len(t)+1, interval=100, repeat=False)
    # set limits
    ax.set_xlim([-10, 130])
    ax.set_ylim([-2, 18])
    plt.show()
    return ani

def plot_multi_vars_mpc(var_name_list: list,var_list: list, sampling_time, var_list_demo: list = None):
    t = [k * sampling_time for k in range(var_list[0].shape[0])]

    fig_var, axs = plt.subplots(len(var_name_list), 1, figsize=(5, 8))
    fig_var.suptitle('STL-Control Synthesis')

    for i, var in enumerate(var_name_list):
        axs[i].set_title(f'{var} vs time')
        axs[i].plot(t, var_list[i], '-r', label=var)
        if var_list_demo is not None:
            axs[i].plot(t, var_list_demo[i], '-g', label=var + '_demo')
        axs[i].grid()
        axs[i].legend(prop={'size': 10})
    fig_var.tight_layout()
    plt.show()

def visualize_multiple(px_demo, py_demo, stl_milp_1, stl_milp_2, stl_milp_3, region, sampling_time):
    t = [k * sampling_time for k in stl_milp_1.variables['px'].keys()]

    x_ped = [var.x for var in stl_milp_1.variables['x_ped'].values()]
    y_ped = [var.x for var in stl_milp_1.variables['y_ped'].values()]

    px_1 = [var.x for var in stl_milp_1.variables['px'].values()]
    py_1 = [var.x for var in stl_milp_1.variables['py'].values()]

    px_2 = [var.x for var in stl_milp_2.variables['px'].values()]
    py_2 = [var.x for var in stl_milp_2.variables['py'].values()]

    px_3 = [var.x for var in stl_milp_3.variables['px'].values()]
    py_3 = [var.x for var in stl_milp_3.variables['py'].values()]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    horizon = t[-1]/sampling_time
    ax.grid()
    ax.tick_params(labelsize=20)
    ax.set_xlabel('x', fontsize=22)
    ax.set_ylabel('y', fontsize=22)

    # plot a black line at sidewalk
    ax.axhline(y=0, color='k', linestyle='-')
    ax.axhline(y=1, color='k', linestyle='-')
    ax.axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=7.95, color='k', linestyle='-')
    ax.axhline(y=8.05, color='k', linestyle='-')
    ax.axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=15, color='k', linestyle='-')
    ax.axhline(y=16, color='k', linestyle='-')

    rho1 = stl_milp_1.variables[stl_milp_1.formula][0][1].x
    rho2 = stl_milp_2.variables[stl_milp_2.formula][0][1].x
    rho3 = stl_milp_3.variables[stl_milp_3.formula][0][1].x

    # Region contains the final region limits as [x_min, x_max, y_min, y_max]
    ax.fill_between([region[0], region[1]], region[2], region[3], color='m', alpha=0.3, label='destination')

    line2, = ax.plot(x_ped, y_ped, '-b', label=r'pedestrian',linewidth=3, marker='s', markersize=7, alpha = 0.7)
    line3, = ax.plot(px_demo, py_demo, '-g', label=r'demonstration',linewidth=3, marker='s', markersize=7, alpha = 0.7)
    line, = ax.plot(px_1, py_1, '-r', label=r'car W1 $\rho$='+f'{rho1:.3f}', linewidth=3, marker='s', markersize=7, alpha = 0.5)
    line4, = ax.plot(px_2, py_2, '-m', label=r'car W2 $\rho$='+f'{rho2:.3f}', linewidth=3, marker='s', markersize=7, alpha = 0.5)
    line5, = ax.plot(px_3, py_3, '-c', label=r'car W3 $\rho$='+f'{rho3:.3f}', linewidth=3, marker='s', markersize=7, alpha = 0.5)

    # Show initial and final coordinates for pedestrian by adding text of (x,y) at the points
    ax.plot(x_ped[0], y_ped[0], 'bo')
    ax.plot(x_ped[-1], y_ped[-1], 'bo')
    ax.text(x_ped[0], y_ped[0], f'({x_ped[0]:.2f},{y_ped[0]:.2f})', fontsize=8)
    ax.text(x_ped[-1], y_ped[-1], f'({x_ped[-1]:.2f},{y_ped[-1]:.2f})', fontsize=8)

    def animate(i):
        line2.set_xdata(x_ped[:i])
        line2.set_ydata(y_ped[:i])
        line3.set_xdata(px_demo[:i])
        line3.set_ydata(py_demo[:i])
        line.set_xdata(px_1[:i])
        line.set_ydata(py_1[:i])
        line4.set_xdata(px_2[:i])
        line4.set_ydata(py_2[:i])
        line5.set_xdata(px_3[:i])
        line5.set_ydata(py_3[:i])
        return line, line2, line3

    ani = animation.FuncAnimation(fig, animate, frames=len(t)+1, interval=100, repeat=False)
    plt.legend(fontsize=22)
    ax.set_xlim([-2, 140])
    ax.set_ylim([0.5, 5])
    plt.show()
    return ani

def visualize_grid(px_demo, py_demo, stl_milp_1, stl_milp_2, stl_milp_3, region, sampling_time, lambdas):

    colors = {'ped': 'black', 'demo': '#4575b4', 'car1': '#d73027', 'car2': '#91bfdb', 'car3': '#fc8d59', 'destination': '#fee090'}

    subsample = 10
    ts = []
    x_peds = []
    y_peds = []
    px_1s = []
    py_1s = []
    px_2s = []
    py_2s = []
    px_3s = []
    py_3s = []
    for i in range(len(stl_milp_1)):
        ts.append([k * sampling_time for k in stl_milp_1[i][0].variables['px'].keys()])

        x_peds.append([var.x for var in stl_milp_1[i][0].variables['x_ped'].values()])
        y_peds.append([var.x for var in stl_milp_1[i][0].variables['y_ped'].values()])

        px_1s.append([var.x for var in stl_milp_1[i][0].variables['px'].values()])
        py_1s.append([var.x for var in stl_milp_1[i][0].variables['py'].values()])

        px_2s.append([var.x for var in stl_milp_2[i][0].variables['px'].values()])
        py_2s.append([var.x for var in stl_milp_2[i][0].variables['py'].values()])

        px_3s.append([var.x for var in stl_milp_3[i][0].variables['px'].values()])
        py_3s.append([var.x for var in stl_milp_3[i][0].variables['py'].values()])

    # change font to times new roman
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(2, 2, figsize=(18, 4.5))

    ax[0,0].set_ylabel('y    ', fontsize=18)
    ax[0,0].yaxis.label.set_rotation(0)
    ax[1,0].set_ylabel('y    ', fontsize=18)
    ax[1,0].yaxis.label.set_rotation(0)
    ax[1,0].set_xlabel('x', fontsize=18)
    ax[1,1].set_xlabel('x', fontsize=18)

    axes = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]

    axes[0].set_title(f'a) $\lambda$={lambdas[0]}', fontsize=16)
    axes[1].set_title(f'b) $\lambda$={lambdas[1]}', fontsize=16)
    axes[2].set_title(f'c) $\lambda$={lambdas[2]}', fontsize=16)
    axes[3].set_title(f'd) $\lambda$={lambdas[3]}, J=0', fontsize=16)
    
    for i in range(len(stl_milp_1)):
        axes[i].grid()
        axes[i].tick_params(labelsize=12)
        # set y ticks off but keep y grid
        if(i == 1  or i == 3):
            axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        
        if (i == 0 or i == 1):
            axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # plot a black line at sidewalk
        axes[i].axhline(y=0, color='k', linestyle='-')
        axes[i].axhline(y=1, color='k', linestyle='-')
        axes[i].axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
        axes[i].axhline(y=7.95, color='k', linestyle='-')
        axes[i].axhline(y=8.05, color='k', linestyle='-')
        axes[i].axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
        axes[i].axhline(y=15, color='k', linestyle='-')
        axes[i].axhline(y=16, color='k', linestyle='-')
        # Region contains the final region limits as [x_min, x_max, y_min, y_max]
        axes[i].fill_between([region[0], region[1]], region[2], region[3], color=colors['destination'], alpha=0.3, label='Destination')

    rho1 = []
    rho2 = []
    rho3 = []
    for i in range(len(stl_milp_1)):
        rho1.append(stl_milp_1[i][0].variables[stl_milp_1[i][0].formula][0][1].x)
        rho2.append(stl_milp_2[i][0].variables[stl_milp_2[i][0].formula][0][1].x)
        rho3.append(stl_milp_3[i][0].variables[stl_milp_3[i][0].formula][0][1].x)

    lines_ped = []
    lines_demo = []
    lines_1 = []
    lines_2 = []
    lines_3 = []
    for i in range(len(stl_milp_1)):
        line_ped, = axes[i].plot(x_peds[i], y_peds[i], '-b', label=r'Pedestrian', linewidth=2, marker='o', markersize=5, alpha = 1.0)
        line_demo, = axes[i].plot(px_demo, py_demo, '-g', label=r'Demonstration', linewidth=2, marker='s', markersize=5, alpha = 1.0)
        line1, = axes[i].plot(px_1s[i], py_1s[i], '-r', label=r'Car W1', linewidth=2, marker='^', markersize=5, alpha = 1.0)
        line2, = axes[i].plot(px_2s[i], py_2s[i], '-m', label=r'Car W2', linewidth=2, marker='D', markersize=6, alpha = 1.0)
        line3, = axes[i].plot(px_3s[i], py_3s[i], '-c', label=r'Car W3', linewidth=2, marker='*', markersize=5, alpha = 1.0)
        # change to photocopy safe
        line_ped.set_color(colors['ped'])
        line_demo.set_color(colors['demo'])
        line1.set_color(colors['car1'])
        line2.set_color(colors['car2'])
        line3.set_color(colors['car3'])
        lines_ped.append(line_ped)
        lines_demo.append(line_demo)
        lines_1.append(line1)
        lines_2.append(line2)
        lines_3.append(line3)

        axes[i].plot(x_peds[i][-3], y_peds[i][-3], 'o', color=colors['ped'])
        axes[i].text(x_peds[i][-3], y_peds[i][-3], f'  pedestrian\n  end', color=colors['ped'], fontsize=9)

        # add $\rho$='+f'{rho:.3f} as text
        axes[i].text(0.01, 0.42, fr'$\rho_{{W1}}$={rho1[i]:.3f}', transform=axes[i].transAxes, fontsize=11, verticalalignment='top')
        axes[i].text(0.01, 0.32, fr'$\rho_{{W2}}$={rho2[i]:.3f}', transform=axes[i].transAxes, fontsize=11, verticalalignment='top')
        axes[i].text(0.01, 0.22, fr'$\rho_{{W3}}$={rho3[i]:.3f}', transform=axes[i].transAxes, fontsize=11, verticalalignment='top')

    sub_text = subsample*4
    def animate(i):
        for j in range(len(stl_milp_1)):
            lines_ped[j].set_xdata(x_peds[j][:i:subsample])
            lines_ped[j].set_ydata(y_peds[j][:i:subsample])
            lines_1[j].set_xdata(px_1s[j][:i:subsample])
            lines_1[j].set_ydata(py_1s[j][:i:subsample])
            lines_2[j].set_xdata(px_2s[j][:i:subsample])
            lines_2[j].set_ydata(py_2s[j][:i:subsample])
            lines_3[j].set_xdata(px_3s[j][:i:subsample])
            lines_3[j].set_ydata(py_3s[j][:i:subsample])
            lines_demo[j].set_xdata(px_demo[:i:subsample])
            lines_demo[j].set_ydata(py_demo[:i:subsample])
            # add temporal markers every 10 steps
            dx1 = [0.0, 2.5, 1.0, 2.0]
            dy1 = [-0.25, -0.06, -0.25, -0.2]
            dx2 = [0.0, 0.0, -3.0, 0.0]
            dy2 = [-0.25, -0.3, 0.1, 0.1]
            dx3 = [0.0, 2.0, 2.0, 0.0]
            dy3 = [-0.25, 0.09, 0.0, 0.1]
            dxdemo = [6.0, 0.0, -3.0, 0.0]
            dydemo = [0.0, 0.15, -0.25, 0.1]
            dxped = [20.0, 40.0, 60.0, 2.0]
            dyped = [20.0, 40.0, 60.0, 0.0]
            # dx1 = [0.0, 2.5, 0.0, 2.0]
            # dy1 = [-0.25, -0.06, -0.25, -0.1]
            # dx2 = [0.0, 0.0, -3.5, -5.0]
            # dy2 = [-0.25, -0.3, 0.1, -0.4]
            # dx3 = [0.0, 2.0, 3.0, -1.0]
            # dy3 = [-0.25, 0.09, 0.0, 0.2]
            # dxdemo = [2.0, 0.0, -2.5, 0.0]
            # dydemo = [0.2, 0.15, 0.18, 0.1]
            # dxped = [20.0, 40.0, 60.0, 2.0]
            # dyped = [20.0, 40.0, 60.0, -0.1]
            if i % sub_text == 0:
                axes[j].text(x_peds[j][i]+dxped[int(i/sub_text)], y_peds[j][i]+dyped[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['ped'], weight='bold')
                axes[j].text(px_1s[j][i]+dx1[int(i/sub_text)], py_1s[j][i]+dy1[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['car1'], weight='bold')
                axes[j].text(px_2s[j][i]+dx2[int(i/sub_text)], py_2s[j][i]+dy2[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['car2'], weight='bold')
                axes[j].text(px_3s[j][i]+dx3[int(i/sub_text)], py_3s[j][i]+dy3[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['car3'], weight='bold')
                axes[j].text(px_demo[i]+dxdemo[int(i/sub_text)], py_demo[i]+dydemo[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['demo'], weight='bold')

        return lines_ped, lines_demo, lines_1, lines_2, lines_3
    
    handles1, labels1 = axes[0].get_legend_handles_labels()
    legend = fig.legend(
    handles=handles1,
    labels=labels1,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=6,
    fontsize=15,
    labelspacing=0.0, 
    columnspacing=2
    )

    ani = animation.FuncAnimation(fig, animate, frames=len(ts[0])+1, interval=sampling_time*1000/subsample/2, repeat=False)
    # plt.legend(fontsize=16)
    for i in range(len(stl_milp_1)):
        axes[i].set_xlim([-2, 140])
        axes[i].set_ylim([0.5, 5])
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    plt.show()
    return ani

def visualize_just_one(px_demo, py_demo, stl_milp_1, stl_milp_2, stl_milp_3, region, sampling_time, lambdas):

    colors = {'ped': 'black', 'demo': '#4575b4', 'car1': '#d73027', 'car2': '#91bfdb', 'car3': '#fc8d59', 'destination': '#fee090'}

    subsample = 10
    ts = []
    x_peds = []
    y_peds = []
    px_1s = []
    py_1s = []
    px_2s = []
    py_2s = []
    px_3s = []
    py_3s = []
    for i in range(len(stl_milp_1)):
        ts.append([k * sampling_time for k in stl_milp_1[i][0].variables['px'].keys()])

        x_peds.append([var.x for var in stl_milp_1[i][0].variables['x_ped'].values()])
        y_peds.append([var.x for var in stl_milp_1[i][0].variables['y_ped'].values()])

        px_1s.append([var.x for var in stl_milp_1[i][0].variables['px'].values()])
        py_1s.append([var.x for var in stl_milp_1[i][0].variables['py'].values()])

        px_2s.append([var.x for var in stl_milp_2[i][0].variables['px'].values()])
        py_2s.append([var.x for var in stl_milp_2[i][0].variables['py'].values()])

        px_3s.append([var.x for var in stl_milp_3[i][0].variables['px'].values()])
        py_3s.append([var.x for var in stl_milp_3[i][0].variables['py'].values()])

    # change font to times new roman
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

    ax.set_ylabel('y    ', fontsize=18)
    ax.yaxis.label.set_rotation(0)
    ax.set_xlabel('x', fontsize=18)
    ax.set_title(f'$\lambda$={lambdas[0]}', fontsize=16)

    ax.grid()
    ax.tick_params(labelsize=12)

    # plot a black line at sidewalk
    ax.axhline(y=0, color='k', linestyle='-')
    ax.axhline(y=1, color='k', linestyle='-')
    ax.axhline(y=4.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=7.95, color='k', linestyle='-')
    ax.axhline(y=8.05, color='k', linestyle='-')
    ax.axhline(y=11.5, color='k', linestyle=(0, (5, 10)))
    ax.axhline(y=15, color='k', linestyle='-')
    ax.axhline(y=16, color='k', linestyle='-')
    # Region contains the final region limits as [x_min, x_max, y_min, y_max]
    ax.fill_between([region[0], region[1]], region[2], region[3], color=colors['destination'], alpha=0.3, label='Destination')

    rho1 = []
    rho2 = []
    rho3 = []
    for i in range(len(stl_milp_1)):
        rho1.append(stl_milp_1[i][0].variables[stl_milp_1[i][0].formula][0][1].x)
        rho2.append(stl_milp_2[i][0].variables[stl_milp_2[i][0].formula][0][1].x)
        rho3.append(stl_milp_3[i][0].variables[stl_milp_3[i][0].formula][0][1].x)

    lines_ped = []
    lines_demo = []
    lines_1 = []
    lines_2 = []
    lines_3 = []
    for i in range(len(stl_milp_1)):
        line_ped, = ax.plot(x_peds[i], y_peds[i], '-b', label=r'Pedestrian', linewidth=2, marker='o', markersize=5, alpha = 1.0)
        line_demo, = ax.plot(px_demo, py_demo, '-g', label=r'Demonstration', linewidth=2, marker='s', markersize=5, alpha = 1.0)
        line1, = ax.plot(px_1s[i], py_1s[i], '-r', label=r'Car W1', linewidth=2, marker='^', markersize=5, alpha = 1.0)
        line2, = ax.plot(px_2s[i], py_2s[i], '-m', label=r'Car W2', linewidth=2, marker='D', markersize=6, alpha = 1.0)
        line3, = ax.plot(px_3s[i], py_3s[i], '-c', label=r'Car W3', linewidth=2, marker='*', markersize=5, alpha = 1.0)
        # change to photocopy safe
        line_ped.set_color(colors['ped'])
        line_demo.set_color(colors['demo'])
        line1.set_color(colors['car1'])
        line2.set_color(colors['car2'])
        line3.set_color(colors['car3'])
        lines_ped.append(line_ped)
        lines_demo.append(line_demo)
        lines_1.append(line1)
        lines_2.append(line2)
        lines_3.append(line3)

        ax.plot(x_peds[i][-3], y_peds[i][-3], 'o', color=colors['ped'])
        ax.text(x_peds[i][-3], y_peds[i][-3], f'  pedestrian\n  end', color=colors['ped'], fontsize=9)

        # add $\rho$='+f'{rho:.3f} as text
        ax.text(0.01, 0.4, fr'$\rho_{{W1}}$={rho1[i]:.3f}', transform=ax.transAxes, fontsize=11, verticalalignment='top')
        ax.text(0.01, 0.3, fr'$\rho_{{W2}}$={rho2[i]:.3f}', transform=ax.transAxes, fontsize=11, verticalalignment='top')
        ax.text(0.01, 0.2, fr'$\rho_{{W3}}$={rho3[i]:.3f}', transform=ax.transAxes, fontsize=11, verticalalignment='top')

    sub_text = subsample*4
    def animate(i):
        for j in range(len(stl_milp_1)):
            lines_ped[j].set_xdata(x_peds[j][:i:subsample])
            lines_ped[j].set_ydata(y_peds[j][:i:subsample])
            lines_1[j].set_xdata(px_1s[j][:i:subsample])
            lines_1[j].set_ydata(py_1s[j][:i:subsample])
            lines_2[j].set_xdata(px_2s[j][:i:subsample])
            lines_2[j].set_ydata(py_2s[j][:i:subsample])
            lines_3[j].set_xdata(px_3s[j][:i:subsample])
            lines_3[j].set_ydata(py_3s[j][:i:subsample])
            lines_demo[j].set_xdata(px_demo[:i:subsample])
            lines_demo[j].set_ydata(py_demo[:i:subsample])
            # add temporal markers every 10 steps
            # dx1 = [0.0, 2.5, 1.0, 2.0]
            # dy1 = [-0.25, -0.06, -0.25, -0.2]
            # dx2 = [0.0, 0.0, -3.0, 0.0]
            # dy2 = [-0.25, -0.3, 0.1, 0.1]
            # dx3 = [0.0, 2.0, 2.0, 0.0]
            # dy3 = [-0.25, 0.09, 0.0, 0.1]
            # dxdemo = [6.0, 0.0, -3.0, 0.0]
            # dydemo = [0.0, 0.15, -0.25, 0.1]
            # dxped = [20.0, 40.0, 60.0, 2.0]
            # dyped = [20.0, 40.0, 60.0, 0.0]
            dx1 = [0.0, 2.5, 0.0, 2.0]
            dy1 = [-0.25, -0.06, -0.25, -0.1]
            dx2 = [0.0, 0.0, -3.5, -5.0]
            dy2 = [-0.25, -0.3, 0.1, -0.4]
            dx3 = [0.0, 2.0, 3.0, -1.0]
            dy3 = [-0.25, 0.09, 0.0, 0.2]
            dxdemo = [2.0, 0.0, -2.5, 0.0]
            dydemo = [0.2, 0.15, 0.18, 0.1]
            dxped = [20.0, 40.0, 60.0, 2.0]
            dyped = [20.0, 40.0, 60.0, -0.1]
            if i % sub_text == 0:
                ax.text(x_peds[j][i]+dxped[int(i/sub_text)], y_peds[j][i]+dyped[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['ped'], weight='bold')
                ax.text(px_1s[j][i]+dx1[int(i/sub_text)], py_1s[j][i]+dy1[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['car1'], weight='bold')
                ax.text(px_2s[j][i]+dx2[int(i/sub_text)], py_2s[j][i]+dy2[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['car2'], weight='bold')
                ax.text(px_3s[j][i]+dx3[int(i/sub_text)], py_3s[j][i]+dy3[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['car3'], weight='bold')
                ax.text(px_demo[i]+dxdemo[int(i/sub_text)], py_demo[i]+dydemo[int(i/sub_text)], f't={ts[j][i]:.0f}', fontsize=10, color=colors['demo'], weight='bold')

        return lines_ped, lines_demo, lines_1, lines_2, lines_3
    
    handles1, labels1 = ax.get_legend_handles_labels()
    legend = fig.legend(
    handles=handles1,
    labels=labels1,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    fontsize=15,
    labelspacing=0.0, 
    columnspacing=2
    )

    ani = animation.FuncAnimation(fig, animate, frames=len(ts[0])+1, interval=sampling_time*1000/subsample/2, repeat=False)
    # plt.legend(fontsize=16)
    for i in range(len(stl_milp_1)):
        ax.set_xlim([-2, 140])
        ax.set_ylim([0.5, 5])
    fig .tight_layout()
    fig.subplots_adjust(top=0.75)
    
    plt.show()
    return ani
