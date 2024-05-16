import numpy as np

def read_demonstration(filename : str):
    # read csv in carla_settings/demonstrations
    data = np.genfromtxt(filename, delimiter=',')[1:,:]
    x = data[:,0]
    y = data[:,1]
    th = data[:,2]
    v = data[:,3]
    t = data[:,4]

    # coordinate transformations (-143, -4) is (0,2.5), (-144, -4) is (1,2.5)
    x = -x - 143
    y = y + 6.5
    t = t - t[0]

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

    x = x[zero_idx:][::8]
    y = y[zero_idx:][::8]
    th = th[zero_idx:][::8]
    v = v[zero_idx:][::8]
    t = t[:len][::8]

    return x, y, th, v, t