import numpy as np
import torch
from typing import Callable, Optional, Tuple, Type

class BicycleModel:
    def __init__(self, dt : float = 0.05):
        self.dt = dt
        self.length = 3.0
        self.length_rear = 1.5

    def get_params(self) -> Tuple[float, float]:
        return self.length, self.length_rear

    def dynamics_ode(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the dynamics of the bicycle model x_dot = f(x, u). Supports batching.
    
        Args:
            x: (B,4) state [x, y, v, theta]
            u: (B,2) control input [a, delta]
        Returns:
            x_dot: (B,4) derivative of state
        """
        x = torch.reshape(x, (-1, 4))
        u = torch.reshape(u, (-1, 2))
        
        # Get batch size
        B = x.shape[0]
        
        # Get physical properties
        L, lr = self.get_params()

        # Get state and control
        pos_x, pos_y, v, theta = x[:,0], x[:,1], x[:,2], x[:,3]
        a, delta = u[:,0], u[:,1]

        # Equations of motion
        beta = torch.atan2(lr * torch.tan(delta), torch.tensor(L))

        pos_x_dot = v * torch.cos(theta + beta)
        pos_y_dot = v * torch.sin(theta + beta)
        v_dot = a
        theta_dot = v * torch.sin(beta) / lr

        x_dot = torch.zeros((B, 4))
        x_dot[:,0] = pos_x_dot
        x_dot[:,1] = pos_y_dot
        x_dot[:,2] = v_dot
        x_dot[:,3] = theta_dot

        return x_dot

    def integrate_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Computes x_t+1 = f(x_t, u_t) using rk4 on analytic dynamics model
        Supports batching.

        Args:
            x_t: (B,4) state [x, y, v, theta]
            u_t: (B,2) control input [a, delta]

        Returns:
            x_t+1: (B, 4) next state
        """
        x = torch.reshape(x, (-1, 4))
        u = torch.reshape(u, (-1, 2))

        k1 = self.dt * self.dynamics_ode(x, u)
        k2 = self.dt * self.dynamics_ode(x + k1 / 2, u)
        k3 = self.dt * self.dynamics_ode(x + k2 / 2, u)
        k4 = self.dt * self.dynamics_ode(x + k3, u)

        next_state = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Wrap angles
        next_theta = torch.chunk(next_state, 4, dim=1)[3]
        next_theta_w = torch.atan2(torch.sin(next_theta), torch.cos(next_theta))
        next_state = torch.cat((next_state[:,:3], next_theta_w), dim=1)

        return next_state
    
    def eval_autograd_jacobian(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the continuous-time jacobians A and B at the given state and input using autograd on analytic dynamics

        Args:
            state: (1, 4) state [x, y, v, theta]
            control: (1, 2) control input [a, delta]

        Returns:
            A: (4, 4) process jacobian wrt state
            B: (4, 1) process jacobian wrt input 
        """
        x = torch.reshape(x, (-1, 4))
        u = torch.reshape(u, (-1, 2))

        # Linearize dynamics
        J = torch.autograd.functional.jacobian(self.dynamics_ode,(x, u))
        
        A = J[0].squeeze()
        B = J[1].squeeze()

        return A, B
    
    def eval_analytic_jacobian(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the continuous-time jacobians A, B at the given state and input analytically

        Args:
            x: (1, 4) state [x, y, v, theta]
            u: (1, 2) control input [a, delta]

        Returns:
            A: (4, 4) process jacobian wrt state
            B: (4, 2) process jacobian wrt input

        """
        x = torch.reshape(x, (-1, 4))
        u = torch.reshape(u, (-1, 2))

        # Get physical properties
        L, lr = self.get_params()

        # Get state and control
        pos_x, pos_y, v, theta = x[:,0], x[:,1], x[:,2], x[:,3]
        a, delta = u[:,0], u[:,1]

        # Linearize dynamics

        beta = torch.atan2(lr * torch.tan(delta), torch.tensor(L))

        A = torch.tensor([[0, 0, torch.cos(theta + beta), -v * torch.sin(theta + beta)],
                          [0, 0, torch.sin(theta + beta),  v * torch.cos(theta + beta)],
                          [0, 0, 0, 0],
                          [0, 0, torch.sin(beta)/lr, 0]])
        
        B = torch.tensor([[0, -lr/L * v * torch.sin(theta + beta) * torch.cos(beta)**2 * (1 + torch.tan(delta)**2)],
                          [0,  lr/L * v * torch.cos(theta + beta) * torch.cos(beta)**2 * (1 + torch.tan(delta)**2)],
                          [1, 0],
                          [0, v / L * torch.cos(beta)**3 * (1 + torch.tan(delta)**2)]])
        
        return A, B
    
    def discretize_linear_dynamics(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize the linearized dynamics A, B using exact discretization
        
        Args:
            A: (4, 4) process jacobian wrt state
            B: (4, 2) process jacobian wrt input

        Returns:
            Ad: (4, 4) discrete-time process jacobian wrt state
            Bd: (4, 2) discrete-time process jacobian wrt input
        """
        T = self.dt

        Ad = torch.matrix_exp(A * self.dt)
        Bd = (T * torch.eye(4) + 1/2 * T**2 * A + 1/6 * T**3 * torch.matmul(A, A)) @ B

        rem = (T * torch.eye(4) + 1/2 * T**2 * A + 1/6 * T**3 * torch.matmul(A, A))
        return Ad, Bd, rem
    
    def discretize_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linearize and discretize the dynamics xdot = f(x, u) using exact discretization

        Args:
            x: (B,4) state [x, y, v, theta]
            u: (B,2) control input [a, delta]
        Returns:
            A_d: (4, 4) discrete-time process jacobian wrt state
            B_d: (4, 2) discrete-time process jacobian wrt input
        """


        x = torch.reshape(x, (-1, 4))
        u = torch.reshape(u, (-1, 2))

        # Get physical properties
        L, lr = self.get_params()
        T = self.dt

        # Get state and control
        pos_x, pos_y, v, theta = x[:,0], x[:,1], x[:,2], x[:,3]
        a, delta = u[:,0], u[:,1]

        beta = torch.atan2(lr * torch.tan(delta), torch.tensor(L))

        Ad = torch.tensor([[1, 0, T * torch.cos(theta + beta) - T**2/2 * v * torch.sin(beta) * torch.sin(beta + theta), -T * v * torch.sin(beta + theta)],
                           [0, 1, T * torch.sin(beta + theta) + T**2/2 * v * torch.sin(beta) * torch.cos(beta + theta),  T * v * torch.cos(beta + theta)],
                           [0, 0, 1, 0],
                           [0, 0, T / lr * torch.sin(beta), 1]])
        
        Bd = torch.tensor([[T**2/2 * torch.cos(beta + theta) - T**3/(6*lr) * v * torch.sin(beta) * torch.sin(beta + theta), -lr/L * T * v * torch.cos(beta)**2 * torch.sin(beta + theta) * (1 + torch.tan(delta)**2) - T**2/(2*L) * v**2 * torch.cos(beta)**3 * torch.sin(beta + theta) * (1 + torch.tan(delta)**2)],
                           [T**2/2 * torch.sin(beta + theta) + T**3/(6*lr) * v * torch.sin(beta) * torch.cos(beta + theta),  lr/L * T * v * torch.cos(beta)**2 * torch.cos(beta + theta) * (1 + torch.tan(delta)**2) + T**2/(2*L) * v**2 * torch.cos(beta)**3 * torch.cos(beta + theta) * (1 + torch.tan(delta)**2)],
                           [T, 0],
                           [T**2/(2*lr) * torch.sin(beta), T/L * v * torch.cos(beta)**3 * (1 + torch.tan(delta)**2)]])
        
        return Ad, Bd
