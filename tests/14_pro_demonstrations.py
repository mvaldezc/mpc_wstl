from antlr4 import InputStream, CommonTokenStream
import time
from visualize import plot_multi_vars, save_vid, plot_multi_vars_mpc,  visualize_grid, visualize_just_one
from demonstrations import read_pro_demonstrations, read_demonstration_broken
import gurobipy as grb
import sys
import numpy as np
from typing import Callable
from dataclasses import dataclass
sys.path.append('../../pytelo/')

from wstl.wstl import WSTLAbstractSyntaxTreeExtractor
from wstl.wstl2milp import wstl2milp
from wstl.wstlLexer import wstlLexer
from wstl.wstlParser import wstlParser

from bicycleModel import BicycleModel
import torch
import readWeights

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
                lin : dict,
                demo : dict,
                alpha: np.ndarray, 
                beta : np.ndarray,
                zeta : np.ndarray,
                lambd : float,
                stateFlag : bool = True):
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
                                             A[0][2] * (v[k]-lin['v'])    +  A[0][3] * (theta[k]-lin['th'])   +
                                             B[0][0] * u_a[k]  +  B[0][1] * u_delta[k] + f0[0])

        wstl_milp.model.addConstr(py[k+1] == A[1][0] * px[k]   +  A[1][1] * py[k]      +
                                             A[1][2] * (v[k]-lin['v'])    +  A[1][3] * (theta[k]-lin['th'])   +
                                             B[1][0] * u_a[k]  +  B[1][1] * u_delta[k] + f0[1])
        
        wstl_milp.model.addConstr(v[k+1] ==  A[2][0] * px[k]   +  A[2][1] * py[k]      +
                                             A[2][2] * (v[k]-lin['v'])    +  A[2][3] * (theta[k]-lin['th'])   +
                                             B[2][0] * u_a[k]  +  B[2][1] * u_delta[k] + f0[2])

        wstl_milp.model.addConstr(theta[k+1] == A[3][0] * px[k]  +  A[3][1] * py[k]      +
                                                A[3][2] * (v[k]-lin['v'])   +  A[3][3] * (theta[k]-lin['th'])   +
                                                B[3][0] * u_a[k] +  B[3][1] * u_delta[k] + f0[3])

    # Jerk constraints
    wstl_milp.model.addConstr(j[0] == (u_a[0] - 0)/T)
    wstl_milp.model.addConstrs(j[k] == (u_a[k] - u_a[k-1])/T for k in range(1, time_horizon-1))
    # wstl_milp.model.addConstr(sr[0] == (u_delta[0] - 0)/T)
    # wstl_milp.model.addConstrs(sr[k] == (u_delta[k] - u_delta[k-1])/T for k in range(1, time_horizon-1))

    # Initial conditions as additional constraints
    wstl_milp.model.addConstr(px[0] == x_0['px'])
    wstl_milp.model.addConstr(py[0] == x_0['py'])
    wstl_milp.model.addConstr(v[0] == x_0['v'])
    wstl_milp.model.addConstr(theta[0] == x_0['theta'])

    # Setpoint constraints
    # wstl_milp.model.addConstr(px[time_horizon-1] == demo['px'][time_horizon-1])
    # wstl_milp.model.addConstr(py[time_horizon-1] == demo['py'][time_horizon-1])
    # wstl_milp.model.addConstr(v[time_horizon-1] == demo['v'][time_horizon-1])
    # wstl_milp.model.addConstr(theta[time_horizon-1] == demo['th'][time_horizon-1])

    # Pedestrian constraints
    for k in range(time_horizon):
        wstl_milp.model.addConstr(x_ped[k] == pedestrian(k*T)[0])
        wstl_milp.model.addConstr(y_ped[k] == pedestrian(k*T)[1])
        wstl_milp.model.addConstr(x_dist[k] == px[k] - x_ped[k])
        wstl_milp.model.addConstr(y_dist[k] == py[k] - y_ped[k])
        wstl_milp.model.addConstr(distance[k] == grb.norm([x_dist[k], y_dist[k]], 1))

    # add the specification (STL) constraints and objective function
    z_formula, rho_formula = wstl_milp.translate(satisfaction=True)

    wstl_milp.model.addConstr(rho_formula >= 0.001)

    # State error cost
    wstl_milp.model.addConstrs(delta_px[k] == demo['px'][k] - px[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_py[k] == demo['py'][k] - py[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_v[k] == demo['v'][k] -  v[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_theta[k] == demo['th'][k] - theta[k] for k in range(time_horizon))
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
    # wstl_milp.model.addConstrs(sr_abs[k] == grb.abs_(sr[k]) for k in range(time_horizon-1))
    # steering_cost = sum(zeta[1]*sr_abs[k] for k in range(time_horizon-1))

    wstl_milp.model.addConstr(rho_formula >= 0)
    
    if stateFlag:
        wstl_milp.model.setObjective(lambd*rho_formula - state_cost, grb.GRB.MAXIMIZE)
    else:
        wstl_milp.model.setObjective(lambd*rho_formula, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()
    return wstl_milp, rho_formula, z_formula


if __name__ == '__main__':  

    # Read demonstration
    x_demo, y_demo, v_demo, th_demo, t_demo, x_ped, y_ped = read_pro_demonstrations(2)
    # x_demo, y_demo, v_demo, th_demo, t_demo = read_demonstration_broken('../carla_settings/demonstrations/1010_trajectory-a.csv')
    demo = {'px': x_demo.squeeze(), 'py': y_demo.squeeze(), 'v': v_demo.squeeze(), 'th': th_demo.squeeze()}

    # print(demo['px'])

    horizon = x_demo.shape[0]-1

    T = 0.1 # sampling time

    # Define wSTL specification
    W1 = {
        "w1" : lambda k : 1.0, 
        "w2" : lambda k : 1.0, 
        "w3" : lambda k : 1.0,
        "w4" : lambda k : 1.0,
        "w5" : lambda k : 1.0,
        "p1" : lambda k : [1.0, 1.0][k],
        "p2" : lambda k : [1.0, 1.0][k],
        "p3" : lambda k : [1.0, 1.0][k],
        "p4" : lambda k : [1.0, 1.0][k]
        }
    
    W2 = readWeights.read_learned_weights(readWeights.W2)
    W3 = readWeights.read_learned_weights(readWeights.W3)

    phi_rule = f"G[0,{horizon}]^w1 (distance >= 2)"
    phi_confort = f"&&^p1((G[0,{horizon-1}]^w2 (u_a <= 10)), (G[0,{horizon-1}]^w3 (j <= 30)))"
    phi_destination = f"F[{horizon-20},{horizon-10}]^w5 G[0,10]^w4 &&^p3(px >= 98, px <= 140)"
    phi = f"&&^p4 (&&^p2 ({phi_rule}, {phi_confort}), {phi_destination}))"

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 1, 'v': 0, 'theta': -np.pi}
    vars_ub = {'px': 150, 'py': 4.5, 'v': 30, 'theta': np.pi}
    control_lb = {'u_a': -10, 'u_delta': -1, 'j': -80, 'sr': -1}
    control_ub = {'u_a': 15, 'u_delta': 1, 'j': 80, 'sr': 1}

    # Define the initial conditions
    x_0 = {'px': 0, 'py': 2.5, 'v': 0.1, 'theta': 0}

    # Linearization point
    x_lin = torch.tensor([0, 0, 0.5, 0]).reshape(1,4) # x, y doesn't matter for linearization
    u_lin = torch.tensor([0, 0]).reshape(1,2)
    lin = {'v': x_lin[0,2], 'th': x_lin[0,3]}

    # Define the matrices for linear system 
    model = BicycleModel(dt=T)
    Ad, Bd = model.discretize_dynamics(x_lin, u_lin)
    Ad, Bd = Ad.detach().numpy(), Bd.detach().numpy()
    f0 = model.integrate_dynamics(x_lin, u_lin).detach().numpy().reshape(4)

    # Define the pedestrian model
    # @dataclass
    # class pedestrian:
    #     def __call__(self, t):
    #         return [x_ped[int(t/T)], y_ped[int(t/T)]]
    class pedestrian:
        def __init__(self):
            self.x_ped = 116.0 #+ 5*np.random.rand()
            self.y_ped = 17.0
        def __call__(self, t):
            if t <= 16:#20:
                vel = 1.2#0.9
                self.x_ped = self.x_ped #+ 0.1*np.random.randn(1)
                self.y_ped = 17.0 - vel*t
            return [self.x_ped, self.y_ped] 
       
    ped = pedestrian()

    alpha = np.array([0.2, 0.4, 0.05, 0.5])
    beta = np.array([0.00, 0.00])
    zeta =  np.array([0.00, 0.00])
    lambd = 1.0


    # Translate WSTL to MILP and retrieve integer variable for the formula
    stl_ones = []
    stl_twos = []
    stl_threes = []
    lambdas = [0.0, 100.0, 1000.0, 1.0]
    stateFlags = [True, True, True, False]
    # lambdas = [1000.0]
    # stateFlags = [True]
    for i in range(len(lambdas)):

        stl_ones.append(wstl_synthesis_control(phi, W1, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                        control_ub, x_0, lin, demo, alpha, beta, zeta, lambdas[i], stateFlags[i]))
        
        stl_twos.append(wstl_synthesis_control(phi, W2, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                        control_ub, x_0, lin, demo, alpha, beta, zeta, lambdas[i], stateFlags[i]))
        
        stl_threes.append((wstl_synthesis_control(phi, W3, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                    control_ub, x_0, lin, demo, alpha, beta, zeta, lambdas[i], stateFlags[i])))
    

    # Visualize the results
    # state_var_name = ['px', 'py', 'v', 'theta']
    # px_1 = np.array([stl_milp_1.model.getVarByName('px_' + str(i)).x for i in range(horizon+1)])
    # py_1 = np.array([stl_milp_1.model.getVarByName('py_' + str(i)).x for i in range(horizon+1)])
    # v_1 = np.array([stl_milp_1.model.getVarByName('v_' + str(i)).x for i in range(horizon+1)])
    # theta_1 = np.array([stl_milp_1.model.getVarByName('theta_' + str(i)).x for i in range(horizon+1)])
    # state_var_1 = np.vstack((px_1, py_1, v_1, theta_1))

    # px_2 = np.array([stl_milp_2.model.getVarByName('px_' + str(i)).x for i in range(horizon+1)])
    # py_2 = np.array([stl_milp_2.model.getVarByName('py_' + str(i)).x for i in range(horizon+1)])
    # v_2 = np.array([stl_milp_2.model.getVarByName('v_' + str(i)).x for i in range(horizon+1)])
    # theta_2 = np.array([stl_milp_2.model.getVarByName('theta_' + str(i)).x for i in range(horizon+1)])
    # state_var_2 = np.vstack((px_2, py_2, v_2, theta_2))

    # px_3 = np.array([stl_milp_3.model.getVarByName('px_' + str(i)).x for i in range(horizon+1)])
    # py_3 = np.array([stl_milp_3.model.getVarByName('py_' + str(i)).x for i in range(horizon+1)])
    # v_3 = np.array([stl_milp_3.model.getVarByName('v_' + str(i)).x for i in range(horizon+1)])
    # theta_3 = np.array([stl_milp_3.model.getVarByName('theta_' + str(i)).x for i in range(horizon+1)])
    # state_var_3 = np.vstack((px_3, py_3, v_3, theta_3))
    # plot_multi_vars_mpc(state_var_name, state_var, T, state_var_demo)
    # plot_multi_vars(stl_milp, ['u_a', 'u_delta'], T)
    # region = [118, 124, 1, 4.5]
    region = [98, 140, 1, 4.5]
    # ani = visualize_demo_and_stl(x_demo, y_demo, stl_milp_1, T)
    
    ani = visualize_grid(x_demo, y_demo, stl_ones, stl_threes, stl_twos, region, T, lambdas)
    # ani = visualize_just_one(x_demo, y_demo, stl_ones, stl_threes, stl_twos, region, T, lambdas)
    # Create a csv file with the trajectory of the car for the whole horizon, were each row is a different time containing x and y
    # x_traj = np.zeros(horizon)
    # y_traj = np.zeros(horizon)
    # v_traj = np.zeros(horizon)
    # th_traj = np.zeros(horizon)
    # for i in range(horizon):
    #     x_traj[i] = stl_milp_2.model.getVarByName('px_' + str(i)).x
    #     y_traj[i] = stl_milp_2.model.getVarByName('py_' + str(i)).x
    #     v_traj[i] = stl_milp_2.model.getVarByName('v_' + str(i)).x
    #     th_traj[i] = stl_milp_2.model.getVarByName('theta_' + str(i)).x
    # np.savetxt('../carla_settings/preference_synthesis/carla_traj_3.csv', np.vstack((x_traj, y_traj, v_traj)).T, delimiter=',')

    # save last animation frame as a png
    # save_vid(ani, "anim/lambda_1.png")