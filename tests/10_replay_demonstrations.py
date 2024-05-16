from visualize import visualize_demonstration
import numpy as np
from demonstrations import read_demonstration

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
x, y, v, th, t = read_demonstration('../carla_settings/demonstrations/trajectory-a.csv')

# rollout pedestrian dynamics
pedestrian_position = np.array([ped(t[i]) for i in range(t.shape[0])])
visualize_demonstration(x, y, pedestrian_position[:,0], pedestrian_position[:,1], t)