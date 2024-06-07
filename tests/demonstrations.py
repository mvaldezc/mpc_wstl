import numpy as np
import torch
import pickle

def read_demonstration(filename : str, T : float = 0.2):
    # read csv in carla_settings/demonstrations
    data = np.genfromtxt(filename, delimiter=',')[1:,:]
    x = data[:,0]
    y = data[:,1]
    v = data[:,2]
    th = data[:,3]
    t = data[:,4]

    # coordinate transformations (-143, -4) is (0,2.5), (-144, -4) is (1,2.5)
    x = -x - 143
    y = y + 6.5
    t = t - t[0]
    th = (th+180)*np.pi/180 # shift -180 to 0 and convert to radians
    th = -np.arctan2(np.sin(th), np.cos(th)) # wrap angle to [-pi, pi] and flip it

    # find zero 
    len = x.shape[0]
    x_prev = x[0]
    zero_idx = 0

    for i in range(1, len):
        if x_prev < 0 and x[i] >= 0:
            zero_idx = i
            break
        x_prev = x[i]

    # remove data before zero and subsample
    sub = int(T//0.025)
    x = x[zero_idx:][::sub]
    y = y[zero_idx:][::sub]
    th = th[zero_idx:][::sub]
    v = v[zero_idx:][::sub]
    t = t[:len - zero_idx][::sub]

    if data[0].shape[0] == 7:
        x_ped = data[:,5]
        y_ped = data[:,6]
        x_ped = -x_ped - 143
        y_ped = y_ped + 6.5
        x_ped = x_ped[zero_idx:][::sub]
        y_ped = y_ped[zero_idx:][::sub]
        return x, y, v, th, t, x_ped, y_ped
    
    return x, y, v, th, t

def read_demonstration_broken(filename : str, T : float = 0.1):
    # read csv in carla_settings/demonstrations
    data = np.genfromtxt(filename, delimiter=',')[1:,:]
    x = data[:,0]
    y = data[:,1]
    v = data[:,8]
    th = data[:,9]
    t = data[:,10]

    # coordinate transformations (-143, -4) is (0,2.5), (-144, -4) is (1,2.5)
    x = -x - 143
    y = y + 6.5
    t = t - t[0]
    th = (th+180)*np.pi/180 # shift -180 to 0 and convert to radians
    th = -np.arctan2(np.sin(th), np.cos(th)) # wrap angle to [-pi, pi] and flip it

    # find zero 
    len = x.shape[0]
    x_prev = x[0]
    zero_idx = 0

    for i in range(1, len):
        if x_prev < 0 and x[i] >= 0:
            zero_idx = i
            break
        x_prev = x[i]

    # remove data before zero and subsample
    sub = 1
    x = x[zero_idx:][::sub]
    y = y[zero_idx:][::sub]
    th = th[zero_idx:][::sub]
    v = v[zero_idx:][::sub]
    t = t[:len - zero_idx][::sub]

    if data[0].shape[0] == 7:
        x_ped = data[:,5]
        y_ped = data[:,6]
        x_ped = -x_ped - 143
        y_ped = y_ped + 6.5
        x_ped = x_ped[zero_idx:][::sub]
        y_ped = y_ped[zero_idx:][::sub]
        return x, y, v, th, t, x_ped, y_ped
    
    return x, y, v, th, t

def read_pro_demonstrations(num:int=0):
    data_name = f"misc/pedestrian_trajectories.pkl"
    with open(data_name, "rb") as f:
        data = pickle.load(f)
    ego_data = []
    ado_data = []
    for k in range(len(data["ego_trajectory"])):
        try:
            ego_data.append(torch.tensor(np.array(data["ego_trajectory"][k]))[:, 1:3])
            ado_data.append(torch.tensor(np.array(data["ado_trajectory"][k]))[:, 1:3])
        except KeyError:
            continue

    N = len(ego_data)

    max_length = max([len(ego_data[k]) for k in range(N)])

    vels = []
    speeds = []
    yaws = []

    for k in range(N):
        velocity = -(ego_data[k][1:, :] - ego_data[k][:-1, :]) / 0.1
        velocity0 = -(ego_data[k][0]-torch.tensor([-142,4.0])) / 0.1
        velocity = torch.cat((velocity0.unsqueeze(0), velocity), axis=0)
        vels.append(velocity)
        speed = torch.linalg.norm(velocity, axis=-1, keepdim=True)
        speeds.append(speed.squeeze())
        
        yaw = torch.atan2(velocity[:, 1], velocity[:, 0])
        yaws.append(yaw)

    x = ego_data[num][:, 0]
    y = ego_data[num][:, 1]
    v = speeds[num]
    th = yaws[num]

    t = np.arange(0, len(x)*0.1, 0.1)

    x = -x - 142
    y = -y + 6.75

    x_ped = ado_data[num][:, 0]
    y_ped = ado_data[num][:, 1]

    x_ped = -x_ped - 147
    y_ped = -y_ped + 6.5

    return x, y, v, th, t, x_ped, y_ped