from antlr4 import InputStream, CommonTokenStream
import time
from visualize import visualize_mpc, plot_multi_vars_mpc
from demonstrations import read_demonstration
import gurobipy as grb
import sys
import numpy as np
from typing import Callable
sys.path.append('../../pytelo/')

from wstl.wstl import WSTLAbstractSyntaxTreeExtractor
from wstl.wstl2milp import wstl2milp
from wstl.wstlLexer import wstlLexer
from wstl.wstlParser import wstlParser

from bicycleModel import BicycleModel
import torch
import matplotlib.pyplot as plt

def wstl_synthesis_control(
                formula : str,
                weights : dict, 
                pedestrian : Callable[[float], list],
                f0: np.ndarray,
                A : np.ndarray,
                B : np.ndarray,
                T : float,
                vars_lb : dict,
                vars_ub : dict,
                control_lb : dict, 
                control_ub : dict,
                t0 : float,
                x_0 : dict,
                demo : dict,
                alpha: np.ndarray, 
                beta : np.ndarray,
                zeta : np.ndarray,
                lambd : float):
    """
    Synthesize a controller for a given wSTL formula and a linear system

    Inputs
    ----------
    formula :  wSTL formula
    weights :  weights for the formula
    pedestrian : pedestrian model, receives time and returns a list with the pedestrian positions
    f0 : nonlinear function at linearization point
    A : system matrix for the discrete time linear system
    B : input matrix for the discrete time linear system
    T : sampling time
    vars_lb : lower bound for the state variables
    vars_ub : upper bound for the state variables
    control_lb : lower bound for the control inputs
    control_ub : upper bound for the control inputs
    t0 : initial time
    x_0 : initial condition for the state variables
    demo : demonstration
    alpha : weight for the state cost
    betax : weight for the control cost
    zeta : weight for the jerk cost
    lambd : weight for the robustness term

    Returns
    -------
    wstl_milp : gurobi model
    """

    # Create Abstract Syntax Tree    
    lexer = wstlLexer(InputStream(formula))
    tokens = CommonTokenStream(lexer)
    parser = wstlParser(tokens)
    t = parser.wstlProperty()
    ast = WSTLAbstractSyntaxTreeExtractor(weights).visit(t)
    
    # Create Gurobi object
    wstl_milp = wstl2milp(ast)
   
    # Get the time horizon
    time_horizon = demo.shape[0]

    # Define the variables for the state, control inputs and jerk
    px = dict()
    py = dict()
    v = dict()
    theta = dict()
    u_a = dict()
    u_delta = dict()
    j = dict()
    sr = dict()

    # Define variables for pedestrian
    x_ped = dict()
    y_ped = dict()
    x_dist = dict()
    y_dist = dict()
    distance = dict()  # define 

    # Define variables for the cost
    delta_px = dict() # define state - setpoint
    delta_py = dict()
    delta_v = dict()
    delta_theta = dict()
    delta_px_abs = dict() # define abs(state-setpoint)
    delta_py_abs = dict()
    delta_v_abs = dict() 
    delta_theta_abs = dict() 
    u_a_abs = dict() # define magnitude of control inputs
    u_delta_abs = dict()
    j_abs = dict() # define magnitude of jerk
    sr_abs = dict() # define magnitude of steering rate

    # Couple predicate variables with constraint variables
    for k in range(time_horizon):  
        # State variables
        name = "px_{}".format(k) 
        px[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['px'], 
                                       ub=vars_ub['px'], name=name)                             
        name = "py_{}".format(k)
        py[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['py'], 
                                       ub=vars_ub['py'], name=name)
        name = "v_{}".format(k) 
        v[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['v'], 
                                       ub=vars_ub['v'], name=name)                             
        name = "theta_{}".format(k)
        theta[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['theta'], 
                                       ub=vars_ub['theta'], name=name)
        
        # Control inputs
        if k < time_horizon - 1:    
            name = "u_a_{}".format(k)
            u_a[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['u_a'], 
                                            ub=control_ub['u_a'], name=name)
            name = "u_delta_{}".format(k)
            u_delta[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['u_delta'], 
                                            ub=control_ub['u_delta'], name=name)
            name = "u_a_abs_{}".format(k)
            u_a_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            name = "u_delta_abs_{}".format(k)
            u_delta_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)

            name = "j_{}".format(k)
            j[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['j'], ub=control_ub['j'], name=name)
            name = "j_abs_{}".format(k)
            j_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            name = "sr_{}".format(k)
            sr[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['sr'], ub=control_ub['sr'], name=name)
            name = "sr_abs_{}".format(k)
            sr_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)

        # Error variables
        name = "delta_px_{}".format(k)
        delta_px[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-130, ub=130, name=name)
        name = "delta_py_{}".format(k)
        delta_py[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-20, ub=20, name=name)
        name = "delta_v_{}".format(k)
        delta_v[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-130, ub=130, name=name)
        name = "delta_theta_{}".format(k)
        delta_theta[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-20, ub=20, name=name)
        name = "delta_px_abs_{}".format(k)
        delta_px_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
        name = "delta_py_abs_{}".format(k)
        delta_py_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
        name = "delta_v_abs_{}".format(k)
        delta_v_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
        name = "delta_theta_abs_{}".format(k)
        delta_theta_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
        
        # Pedestrian variables
        name = "x_ped_{}".format(k)
        x_ped[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-5, ub=125, name=name)
        name = "y_ped_{}".format(k)
        y_ped[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-20, ub=20, name=name)
        name = "x_dist_{}".format(k)
        x_dist[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-125, ub=125, name=name)
        name = "y_dist_{}".format(k)
        y_dist[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-20, ub=20, name=name)
        name = "distance_{}".format(k)
        distance[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            
    # Indicate which variables are related
    wstl_milp.variables['px'] = px
    wstl_milp.variables['py'] = py
    wstl_milp.variables['v'] = v
    wstl_milp.variables['theta'] = theta
    wstl_milp.variables['u_a'] = u_a
    wstl_milp.variables['u_delta'] = u_delta
    wstl_milp.variables['j'] = j
    wstl_milp.variables['x_ped'] = x_ped
    wstl_milp.variables['y_ped'] = y_ped
    wstl_milp.variables['x_dist'] = x_dist
    wstl_milp.variables['y_dist'] = y_dist
    wstl_milp.variables['distance'] = distance
    wstl_milp.variables['delta_px'] = delta_px
    wstl_milp.variables['delta_py'] = delta_py
    wstl_milp.variables['delta_v'] = delta_v
    wstl_milp.variables['delta_theta'] = delta_theta
    wstl_milp.variables['delta_px_abs'] = delta_px_abs
    wstl_milp.variables['delta_py_abs'] = delta_py_abs
    wstl_milp.variables['delta_v_abs'] = delta_v_abs
    wstl_milp.variables['delta_theta_abs'] = delta_theta_abs
    wstl_milp.variables['u_a_abs'] = u_a_abs
    wstl_milp.variables['u_delta_abs'] = u_delta_abs
    wstl_milp.variables['j_abs'] = j_abs
    wstl_milp.variables['sr'] = sr
    wstl_milp.variables['sr_abs'] = sr_abs

    # system constraints x[k+1] = A X[k]+ B U[k]
    for k in range(time_horizon-1):
        wstl_milp.model.addConstr(px[k+1] == A[0][0] * px[k]   +  A[0][1] * py[k]      +
                                             A[0][2] * (v[k]-v[0])    +  A[0][3] * (theta[k]-theta[0])   +
                                             B[0][0] * u_a[k]  +  B[0][1] * u_delta[k] + f0[0])

        wstl_milp.model.addConstr(py[k+1] == A[1][0] * px[k]   +  A[1][1] * py[k]      +
                                             A[1][2] * (v[k]-v[0])    +  A[1][3] * (theta[k]-theta[0])   +
                                             B[1][0] * u_a[k]  +  B[1][1] * u_delta[k] + f0[1])
        
        wstl_milp.model.addConstr(v[k+1] ==  A[2][0] * px[k]   +  A[2][1] * py[k]      +
                                             A[2][2] * (v[k]-v[0])    +  A[2][3] * (theta[k]-theta[0])   +
                                             B[2][0] * u_a[k]  +  B[2][1] * u_delta[k] + f0[2])

        wstl_milp.model.addConstr(theta[k+1] == A[3][0] * px[k]  +  A[3][1] * py[k]      +
                                                A[3][2] * (v[k]-v[0])   +  A[3][3] * (theta[k]-theta[0])   +
                                                B[3][0] * u_a[k] +  B[3][1] * u_delta[k] + f0[3])

    # Jerk constraints
    wstl_milp.model.addConstr(j[0] == (u_a[0] - 0)/T)
    wstl_milp.model.addConstrs(j[k] == (u_a[k] - u_a[k-1])/T for k in range(1, time_horizon-1))
    wstl_milp.model.addConstr(sr[0] == (u_delta[0] - 0)/T)
    wstl_milp.model.addConstrs(sr[k] == (u_delta[k] - u_delta[k-1])/T for k in range(1, time_horizon-1))

    # Initial conditions as additional constraints
    wstl_milp.model.addConstr(px[0] == x_0[0])
    wstl_milp.model.addConstr(py[0] == x_0[1])
    wstl_milp.model.addConstr(v[0] == x_0[2])
    wstl_milp.model.addConstr(theta[0] == x_0[3])

    final_cost_on = 0
    # Setpoint constraints
    if time_horizon < 41:
        final_cost_on = 1
        # wstl_milp.model.addConstr(px[time_horizon-1] == demo[time_horizon-1][0])
        # wstl_milp.model.addConstr(py[time_horizon-1] == demo[time_horizon-1][1])
        # wstl_milp.model.addConstr(v[time_horizon-1] == 0)
        # wstl_milp.model.addConstr(theta[time_horizon-1] == 0)

    # Pedestrian constraints
    for k in range(time_horizon):
        wstl_milp.model.addConstr(x_ped[k] == pedestrian(k*T+t0)[0])
        wstl_milp.model.addConstr(y_ped[k] == pedestrian(k*T+t0)[1])
        wstl_milp.model.addConstr(x_dist[k] == px[k] - x_ped[k])
        wstl_milp.model.addConstr(y_dist[k] == py[k] - y_ped[k])
        wstl_milp.model.addConstr(distance[k] == grb.norm([x_dist[k], y_dist[k]], 1))

    # add the specification (STL) constraints and objective function
    z_formula, rho_formula = wstl_milp.translate(satisfaction=True)

    # State error cost
    wstl_milp.model.addConstrs(delta_px[k] == demo[k][0] - px[k] for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(delta_py[k] == demo[k][1] - py[k] for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(delta_v[k] == demo[k][2] -  v[k] for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(delta_theta[k] == demo[k][3] - theta[k] for k in range(time_horizon-1))
    wstl_milp.model.addConstr(delta_px[time_horizon-1] == demo[time_horizon-1][0] - px[time_horizon-1])
    wstl_milp.model.addConstr(delta_py[time_horizon-1] == demo[time_horizon-1][1] - py[time_horizon-1])
    wstl_milp.model.addConstr(delta_v[time_horizon-1] == demo[time_horizon-1][2] -  v[time_horizon-1])
    wstl_milp.model.addConstr(delta_theta[time_horizon-1] == demo[time_horizon-1][3] - theta[time_horizon-1])
    wstl_milp.model.addConstrs(delta_px_abs[k] == grb.abs_(delta_px[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_py_abs[k] == grb.abs_(delta_py[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_v_abs[k] == grb.abs_(delta_v[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_theta_abs[k] == grb.abs_(delta_theta[k]) for k in range(time_horizon))
    state_cost = sum(alpha[0]*delta_px_abs[k] + alpha[1]*delta_py_abs[k] 
                   + alpha[2]*delta_v_abs[k]  + alpha[3]*delta_theta_abs[k] for k in range(time_horizon-1))
    
    final_cost = final_cost_on*alpha[4]*(alpha[0]*delta_px_abs[time_horizon-1] + alpha[1]*delta_py_abs[time_horizon-1] + alpha[2]*delta_v_abs[time_horizon-1] + alpha[3]*delta_theta_abs[time_horizon-1])
    

    # Control magnitude cost
    wstl_milp.model.addConstrs(u_a_abs[k] == grb.abs_(u_a[k]) for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(u_delta_abs[k] == grb.abs_(u_delta[k]) for k in range(time_horizon-1))
    control_cost = sum(beta[0]*u_a_abs[k] + beta[1]*u_delta_abs[k] for k in range(time_horizon-1))

    # Jerk magnitude cost
    wstl_milp.model.addConstrs(j_abs[k] == grb.abs_(j[k]) for k in range(time_horizon-1))
    jerk_cost = sum(zeta[0]*j_abs[k] for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(sr_abs[k] == grb.abs_(sr[k]) for k in range(time_horizon-1))
    steering_cost = sum(zeta[1]*sr_abs[k] for k in range(time_horizon-1))

    wstl_milp.model.addConstr(rho_formula >= 0)

    print("LAMBDA", lambd*(1-0.5*(41.0-time_horizon)/41.0))
    
    wstl_milp.model.setObjective(lambd*(1-0.5*(41.0-time_horizon)/41.0)*rho_formula - state_cost - control_cost - jerk_cost - steering_cost - final_cost, grb.GRB.MAXIMIZE) #- control_cost - jerk_cost - steering_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()

    return wstl_milp, rho_formula, z_formula

class Pedestrian:
    def __init__(self):
        self.x_ped = 116.0 #+ 5*np.random.rand()
        self.y_ped = 17.0
    def __call__(self, t):
        if t <= 20:#15:#20:
            vel = 0.9#1.2#0.9
            self.x_ped = self.x_ped #+ 0.1*np.random.randn(1)
            self.y_ped = 17.0 - vel*t
        return [self.x_ped, self.y_ped] 

class MPC():
    def __init__(self, dt : float, x0 : np.ndarray, model : BicycleModel, pedestrian : Pedestrian, demo : np.ndarray, 
                 formula : str, weights : dict, bounds : dict, alpha : np.ndarray, beta : np.ndarray, zeta : np.ndarray, 
                 lambd : float, horizon : int) -> None:
        self.dt = dt
        self.x = np.array([x0]).reshape(1,4)
        self.u = np.array([0, 0]).reshape(1,2)
        self.model = model
        self.ped = pedestrian
        self.demo = demo
        self.formula = formula
        self.weights = weights
        self.vars_lb = bounds['vars_lb']
        self.vars_ub = bounds['vars_ub']
        self.control_lb = bounds['control_lb']
        self.control_ub = bounds['control_ub']
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.lambd = lambd
        self.horizon = horizon
        self.robustness = []
        self.end = False
        self.horizon_traj = []
        self.t = 0.0
        self.k = 0

    def compute_horizon(self, waypoints):
        '''
        Compute the control inputs for the whole horizon.
        Args:
            waypoints: array with reference waypoints
        '''
        # Get current state and last input
        xt = self.x[-1]
        u_nom = self.u[0]#self.u[-1]

        xt_lin = np.array([0, 0, xt[2], xt[3]])
        # xt_lin[2] += 0.5 if (xt[2] < 0.5 and xt[2] > 0.0) else 0.0 # avoid zero velocity

        # Approximate dynamics with linearization
        Ad, Bd = self.model.discretize_dynamics(torch.tensor(xt_lin), torch.tensor(u_nom))
        Ad, Bd = Ad.detach().numpy(), Bd.detach().numpy()
        f0 = self.model.integrate_dynamics(torch.tensor(xt_lin), torch.tensor(u_nom)).detach().numpy().reshape(4)

        # Compute optimization problem
        wstl_milp, rho_formula, z_formula = wstl_synthesis_control(self.formula, self.weights, self.ped, f0, Ad, Bd, self.dt, 
                                                                   self.vars_lb, self.vars_ub, self.control_lb, self.control_ub, 
                                                                   self.t, xt, waypoints, self.alpha, self.beta, self.zeta, self.lambd)
        try:
            self.robustness.append(rho_formula.x)
            self.horizon_traj.append(wstl_milp)
        except:
            self.end = True
            return
        
        # Get the first input
        u0 = np.array([wstl_milp.model.getVarByName('u_a_0').x, wstl_milp.model.getVarByName('u_delta_0').x]).reshape(1,2)
        self.u = np.concatenate((self.u, u0), 0)

    def compute_control(self):
        '''
        Plans the reference waypoints and control inputs for the next time step.
        '''
        xt = self.x[-1].reshape(1,4)
        # k = self.get_closest_waypoint(xt, self.demo)
        # print(np.linalg.norm(xt[0,0:2] - self.demo[-1,0:2]))
        # if np.linalg.norm(xt[0,0:2] - np.array([120,2.5])) < 1.0 or k == self.demo.shape[0]-1:#or self.t > 22.0:
        if self.k == self.demo.shape[0]-1:
            self.end = True
            return
        if self.k + self.horizon+1 <= self.demo.shape[0]: # if planning horizon is shorter than the waypoints
            # ref = np.concatenate([xt, self.demo[k+1:k+self.horizon+1]], 0)
            ref = self.demo[self.k:self.k+self.horizon+1]
        else:
            # extra_waypoints = self.demo[-1].reshape(1,4)
            # ref = np.concatenate([xt, self.demo[k+1:], np.tile(extra_waypoints, (self.horizon+1-self.demo.shape[0]+k, 1))], 0)
            # print("REF:", ref)
            # ref = np.concatenate([xt, self.demo[k+1:]], 0)
            ref = self.demo[self.k:]
            # ref = np.concatenate([self.demo[k:], np.tile(extra_waypoints, (self.horizon+1-self.demo.shape[0]+k, 1))], 0)
            # ref = ref[:40]
        self.compute_horizon(ref)
    
    def simulate(self):
        xt = self.x[-1].reshape(1,4)
        u0 = self.u[-1].reshape(1,2)
        x_next = self.model.integrate_dynamics(torch.tensor(xt), torch.tensor(u0)).detach().numpy().reshape(1,4)
        self.x = np.concatenate((self.x, x_next), 0)
        self.k += 1
        self.t = self.k*self.dt

    def get_closest_waypoint(self, current_state, waypoints):
        '''
        Find the closest waypoint to the current state.
        '''
        min_dist = 1e6
        if current_state.shape[1]==4:
            current_state = np.delete(current_state, [2,3])
        if waypoints.shape[1]==4: 
            waypoints= np.delete(waypoints, [2,3], axis=1)
        for i in reversed(range(waypoints.shape[0])):
            dist = np.linalg.norm(np.array(current_state-waypoints[i,:]))
            if dist < min_dist:
                min_dist = dist
                idx = i
        return idx  


if __name__ == '__main__':  

    # Read demonstration
    x_demo, y_demo, v_demo, th_demo, t_demo = read_demonstration('../carla_settings/demonstrations/trajectory-a_5.csv')

    T = 0.2 # sampling time
    horizon = 40

    # Define wSTL specification
    weights = {"w1" : lambda k : 1.0, 
               "w2" : lambda k : 1.0, 
               "p1" : lambda k : [1.0, 1.0][k],
               "p2" : lambda k : [1.0, 1.0][k],
               }

    phi_rule = f"G[0,{horizon}]^w1 (distance >= 2)"
    phi_confort = f"G[0,{horizon-1}]^w2 &&^p2(u_a <= 10, j <= 30)"
    phi = f"&&^p1 ({phi_rule}, {phi_confort})"

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 1, 'v': 0, 'theta': -3.14159}
    vars_ub = {'px': 125, 'py': 6.5, 'v': 30, 'theta': 3.14159}
    control_lb = {'u_a': -10, 'u_delta': -1, 'j': -80, 'sr': -1}
    control_ub = {'u_a': 15, 'u_delta': 1, 'j': 80, 'sr': 1}
    bounds = {'vars_lb': vars_lb, 'vars_ub': vars_ub, 'control_lb': control_lb, 'control_ub': control_ub}

    # Define the initial conditions
    x_0 = [0, 2.5, 0, 0]

    # Define the vehicle model
    model = BicycleModel(dt=T)

    # Define the pedestrian model
    ped = Pedestrian()

    # Define the weights for the cost function
    # beta = np.array([3.0, 6.0])
    # zeta =  np.array([3.0, 2.0])
    alpha = np.array([2.0, 4.0, 0.01, 10.0, 1000.0])

    beta = np.array([0.0, 0.0])
    zeta =  np.array([0.0, 1.0])
    lambd = 500

    # Define the demonstration
    demo = np.array([x_demo, y_demo, v_demo, th_demo]).T

    # Translate WSTL to MILP and retrieve integer variable for the formula
    mpc = MPC(T, x_0, model, ped, demo, phi, weights, bounds, 
              alpha, beta, zeta, lambd, horizon)
    
    start = time.time()
    while True:
        mpc.compute_control()
        print(f'{mpc.t:.2f} s, {mpc.x[-1]}')
        if mpc.end == True:
            break
        mpc.simulate()
    end = time.time()

    mpc.u = np.delete(mpc.u, 0, axis=0)
    time_needed = mpc.u.shape[0]*T
    total_horizon = mpc.u.shape[0]

    print(f"Robustness:", mpc.robustness)
    print("Time needed:", time_needed)
    print("Time horizon:", total_horizon)
    print("Time elapsed:", end-start)

    # Visualize the results

    state_name_var = ['px', 'py', 'v', 'theta']
    state_var = [mpc.x[:,0], mpc.x[:,1], mpc.x[:,2], mpc.x[:,3]]
    control_name_var = ['u_a', 'u_delta']
    control_var = [mpc.u[:,0], mpc.u[:,1]]
    state_var_demo = [x_demo, y_demo, v_demo, th_demo]

    plot_multi_vars_mpc(state_name_var, state_var, T, state_var_demo)
    plot_multi_vars_mpc(control_name_var, control_var, T)

    t = np.linspace(0, time_needed, total_horizon+1)
    pedestrian_position = np.array([ped(t[i]) for i in range(mpc.x.shape[0])])
    ani = visualize_mpc(mpc.horizon_traj,mpc.x[:,0], mpc.x[:,1], pedestrian_position[:,0], pedestrian_position[:,1], t, x_demo, y_demo)


    # Create a csv file with the trajectory of the car for the whole horizon, were each row is a different time containing x and y
    # x_traj = np.zeros(horizon)
    # y_traj = np.zeros(horizon)
    # v_traj = np.zeros(horizon)
    # th_traj = np.zeros(horizon)
    # for i in range(horizon):
    #     x_traj[i] = stl_milp.model.getVarByName('px_' + str(i)).x
    #     y_traj[i] = stl_milp.model.getVarByName('py_' + str(i)).x
    #     v_traj[i] = stl_milp.model.getVarByName('v_' + str(i)).x
    #     th_traj[i] = stl_milp.model.getVarByName('theta_' + str(i)).x
    #np.savetxt('../carla_settings/preference_synthesis/carla_traj.csv', np.vstack((x_traj, y_traj, v_traj)).T, delimiter=',')