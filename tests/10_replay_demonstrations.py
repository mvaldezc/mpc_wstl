from visualize import visualize_demonstration
import numpy as np
from demonstrations import read_demonstration, read_pro_demonstrations

# Define the pedestrian model
class pedestrian:
    def __init__(self):
        self.x_ped = 116.0 #+ 5*np.random.rand()
        self.y_ped = 17.0
    def __call__(self, t):
        if t <= 20: #15: #16.45:
            vel = 0.9#1.2#1.1
            self.x_ped = self.x_ped #+ 0.1*np.random.randn(1)
            self.y_ped = 17.0 - vel*t
        return [self.x_ped, self.y_ped] 
ped = pedestrian()

# Read demonstration
# x, y, v, th, t = read_demonstration('../carla_settings/demonstrations/trajectory-a_5.csv')
x, y, v, th, t, x_ped, y_ped = read_pro_demonstrations(0)
print(y)

# rollout pedestrian dynamics
# pedestrian_position = np.array([ped(t[i]) for i in range(t.shape[0])])
# visualize_demonstration(x, y, pedestrian_position[:,0], pedestrian_position[:,1], t)
visualize_demonstration(x, y, x_ped, y_ped, t)

# change pickle version
import pickle
with open("misc/pedestrian_trajectories.pkl", "rb") as f:
        data = pickle.load(f)
with open("misc/pedestrian_trajectories_downgraded.pkl", "wb") as f:
    pickle.dump(data, f, protocol=4)

# np.savetxt('../carla_settings/preference_synthesis/carla_traj_demo_0.csv', np.vstack((x, y)).T, delimiter=',')