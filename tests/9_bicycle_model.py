from antlr4 import InputStream, CommonTokenStream
import time
from visualize import visualize_animation, plot_multi_vars
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
                x_0 : dict,
                sp : dict,
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
    x_0 : initial condition for the state variables
    setpoint : setpoint for the state variables
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
    time_horizon = int(ast.bound()) + 1

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
    wstl_milp.model.addConstr(px[0] == x_0['px'])
    wstl_milp.model.addConstr(py[0] == x_0['py'])
    wstl_milp.model.addConstr(v[0] == x_0['v'])
    wstl_milp.model.addConstr(theta[0] == x_0['theta'])

    # Setpoint constraints
    wstl_milp.model.addConstr(px[time_horizon-1] == sp['px'])
    wstl_milp.model.addConstr(py[time_horizon-1] == sp['py'])
    wstl_milp.model.addConstr(v[time_horizon-1] == sp['v'])
    wstl_milp.model.addConstr(theta[time_horizon-1] == sp['theta'])

    # Pedestrian constraints
    for k in range(time_horizon):
        wstl_milp.model.addConstr(x_ped[k] == pedestrian(k*T)[0])
        wstl_milp.model.addConstr(y_ped[k] == pedestrian(k*T)[1])
        wstl_milp.model.addConstr(x_dist[k] == px[k] - x_ped[k])
        wstl_milp.model.addConstr(y_dist[k] == py[k] - y_ped[k])
        wstl_milp.model.addConstr(distance[k] == grb.norm([x_dist[k], y_dist[k]], 1))

    # add the specification (STL) constraints and objective function
    z_formula, rho_formula = wstl_milp.translate(satisfaction=True)

    # State error cost
    wstl_milp.model.addConstrs(delta_px[k] == sp['px'] - px[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_py[k] == sp['py'] - py[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs( delta_v[k] == sp['v'] -  v[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_theta[k] == sp['theta'] - theta[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_px_abs[k] == grb.abs_(delta_px[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_py_abs[k] == grb.abs_(delta_py[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_v_abs[k] == grb.abs_(delta_v[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_theta_abs[k] == grb.abs_(delta_theta[k]) for k in range(time_horizon))
    state_cost = sum(alpha[0]*delta_px_abs[k] + alpha[1]*delta_py_abs[k] 
                   + alpha[2]*delta_v_abs[k]  + alpha[3]*delta_theta_abs[k] for k in range(time_horizon))

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
    
    wstl_milp.model.setObjective(lambd*rho_formula - state_cost - control_cost - jerk_cost - steering_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()
    return wstl_milp, rho_formula, z_formula


if __name__ == '__main__':  

    T = 0.2 # sampling time
    time_length = 18 #+ 2*np.random.rand()
    horizon = int(time_length/T)+1

    # Define wSTL specification
    def w1_f(k):
        if k < horizon/2:
            return 0.05
        return 1.0
    weights = {"w1" : lambda k : w1_f(k), 
               "w2" : lambda k : 1.0, 
               "p1" : lambda k : [0.5, 1.0][k],
               "p2" : lambda k : [1.0, 1.0][k],
               }

    phi_rule = f"G[0,{horizon}]^w1 (distance >= 2)"
    phi_confort = f"G[0,{horizon-1}]^w2 &&^p2(u_a <= 10, j <= 30)"
    phi = f"&&^p1 ({phi_rule}, {phi_confort})"

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 0, 'v': -40, 'theta': -1}
    vars_ub = {'px': 125, 'py': 5, 'v': 40, 'theta': 1}
    control_lb = {'u_a': -2, 'u_delta': -5, 'j': -40, 'sr': -1}
    control_ub = {'u_a': 15, 'u_delta': 5, 'j': 40, 'sr': 1}

    # Define the initial and final conditions
    x_0 = {'px': 0, 'py': 2.5, 'v': 0, 'theta': 0}
    x_f = {'px': 120, 'py': 2.5, 'v': 0, 'theta': 0}

    # Linearization point
    x0 = torch.tensor([0, 0, 0.1, 0]).reshape(1,4)
    u0 = torch.tensor([0, 0]).reshape(1,2)

    # Define the matrices for linear system 
    model = BicycleModel(dt=T)
    Ad, Bd = model.discretize_dynamics(x0, u0)
    Ad = Ad.detach().numpy()
    Bd = Bd.detach().numpy()
    f0 = model.integrate_dynamics(x0, u0).detach().numpy().reshape(4)

    # Define the pedestrian model
    class pedestrian:
        def __init__(self):
            self.x_ped = 116.0 #+ 5*np.random.rand()
            self.y_ped = 17.0
        def __call__(self, t):
            if t <= 15: #16.45:
                vel = 1.2#1.1
                self.x_ped = self.x_ped #+ 0.1*np.random.randn(1)
                self.y_ped = 17.0 - vel*t
            return [self.x_ped, self.y_ped] 
    ped = pedestrian()

    alpha = np.array([0.001, 0.005, 0.05, 0.001])
    beta = np.array([0.0001, 0.005])
    zeta =  np.array([0.01, 0.0001])
    lambd = 10

    # Translate WSTL to MILP and retrieve integer variable for the formula
    stl_start = time.time()
    stl_milp, rho_formula, z = wstl_synthesis_control(phi, weights, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                    control_ub, x_0, x_f, alpha, beta, zeta, lambd)
    stl_end = time.time()
    stl_time = stl_end - stl_start

    print(f"Robustness: {rho_formula.x}")
    print('Time needed:', stl_time)
    print("Time horizon:", horizon)
    print("z:", z.x)

    # Visualize the results
    plot_multi_vars(stl_milp, ['px', 'py', 'v', 'theta'], T)
    plot_multi_vars(stl_milp, ['u_a', 'u_delta'], T)
    ani = visualize_animation(stl_milp, T, carla=True)
    # save_vid(ani, "anim/carla.gif")

    # Create a csv file with the trajectory of the car for the whole horizon, were each row is a different time containing x and y
    x_traj = np.zeros(horizon)
    y_traj = np.zeros(horizon)
    v_traj = np.zeros(horizon)
    for i in range(horizon):
        x_traj[i] = stl_milp.model.getVarByName('px_' + str(i)).x
        y_traj[i] = stl_milp.model.getVarByName('py_' + str(i)).x
        v_traj[i] = stl_milp.model.getVarByName('v_' + str(i)).x
    # np.savetxt('../carla_settings/preference_synthesis/carla_traj.csv', np.vstack((x_traj, y_traj, v_traj)).T, delimiter=',')