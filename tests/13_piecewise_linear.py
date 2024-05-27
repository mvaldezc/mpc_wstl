from antlr4 import InputStream, CommonTokenStream
import time
from visualize import visualize_demo_and_stl, plot_multi_vars, save_vid
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
                fs: np.ndarray,
                As : np.ndarray,
                Bs : np.ndarray,
                v_vals : list, 
                th_vals : list, 
                u_delta_vals : list,
                T : float,
                vars_lb : dict,
                vars_ub : dict,
                control_lb : dict, 
                control_ub : dict,
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

    # Define linearization points
    lin_points = np.array(np.meshgrid(v_vals, th_vals, u_delta_vals)).T.reshape(-1, 3)

    # Define the variables for the state, control inputs and jerk
    px = dict()
    py = dict()
    v = dict()
    theta = dict()
    u_a = dict()
    u_delta = dict()
    jerk = dict()
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

    # Define variables for the distance to the linearization points
    delta_lin = [dict() for i in range(lin_points.shape[0])]
    abs_delta_lin = [dict() for i in range(lin_points.shape[0])]
    curr_point = dict()
    z_lin = [dict() for i in range(lin_points.shape[0])]
    inter = [dict() for i in range(lin_points.shape[0])]
    r_lin = dict() # min distance to linearization points

    px_possible =  [dict() for i in range(lin_points.shape[0])]
    py_possible =  [dict() for i in range(lin_points.shape[0])]
    v_possible =  [dict() for i in range(lin_points.shape[0])]
    th_possible =  [dict() for i in range(lin_points.shape[0])]


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
        
        # Possible state variables
        for i in range(lin_points.shape[0]):
            name = "px_possible_{}_{}".format(i, k)
            px_possible[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['px'],
                                        ub=vars_ub['px'], name=name)
            name = "py_possible_{}_{}".format(i, k)
            py_possible[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['py'],
                                        ub=vars_ub['py'], name=name)
            name = "v_possible_{}_{}".format(i, k)
            v_possible[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['v'],
                                        ub=vars_ub['v'], name=name)
            name = "th_possible_{}_{}".format(i, k)
            th_possible[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['theta'],
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
            
            name = "jerk_{}".format(k)
            jerk[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['jerk'], ub=control_ub['jerk'], name=name)
            name = "j_abs_{}".format(k)
            j_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            name = "sr_{}".format(k)
            sr[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['sr'], ub=control_ub['sr'], name=name)
            name = "sr_abs_{}".format(k)
            sr_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            
            # Linearization variables
            for i in range(lin_points.shape[0]):
                name = "delta_lin_{}_{}".format(i, k)
                delta_lin[i][k] = wstl_milp.model.addMVar(shape=3, vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, name=name)
                name = "z_lin_{}_{}".format(i, k)
                z_lin[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.BINARY, name=name)
                name = "abs_delta_lin_{}_{}".format(i, k)
                abs_delta_lin[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
                name = "inter_{}_{}".format(i,k)
                inter[i][k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, name=name)
            name = "curr_point_{}".format(k)
            curr_point[k] = wstl_milp.model.addMVar(shape=3, vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, name=name)
            name = "r_lin_{}".format(k)
            r_lin[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
        

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
    wstl_milp.variables['jerk'] = jerk
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
    wstl_milp.variables['delta_lin'] = delta_lin
    wstl_milp.variables['abs_delta_lin'] = abs_delta_lin
    wstl_milp.variables['curr_point'] = curr_point
    wstl_milp.variables['z_lin'] = z_lin
    wstl_milp.variables['r_lin'] = r_lin
    wstl_milp.variables['px_possible'] = px_possible
    wstl_milp.variables['py_possible'] = py_possible
    wstl_milp.variables['v_possible'] = v_possible
    wstl_milp.variables['th_possible'] = th_possible
    wstl_milp.variables['inter'] = inter

    # Linearization constraints
    for t in range(time_horizon-1):
        # Distance to linearization points
        wstl_milp.model.addConstr(curr_point[t][0] == v[t])
        wstl_milp.model.addConstr(curr_point[t][1] == theta[t])
        if t == 0:
            wstl_milp.model.addConstr(curr_point[t][2] == 0)
        else:
            wstl_milp.model.addConstr(curr_point[t][2] == 0)#u_delta[t-1]
        wstl_milp.model.addConstrs(delta_lin[i][t] == curr_point[t]-lin_points[i] for i in range(lin_points.shape[0]))
        wstl_milp.model.addConstrs(abs_delta_lin[i][t] == grb.norm(delta_lin[i][t],1) for i in range(lin_points.shape[0]))
        wstl_milp.model.addConstr(r_lin[t] == grb.min_(abs_delta_lin[i][t] for i in range(lin_points.shape[0])))
        # add a constraint for if delta_lin[i] == r_lin then z_lin[i] = 1 else z_lin[i] = 0
        wstl_milp.model.addConstrs(inter[i][t] == 1 - (abs_delta_lin[i][t] - r_lin[t])/0.001 for i in range(lin_points.shape[0]))
        wstl_milp.model.addConstrs((z_lin[i][t] == grb.max_(inter[i][t],constant=0)) for i in range(lin_points.shape[0]))
        wstl_milp.model.addConstr(grb.quicksum(z_lin[i][t] for i in range(lin_points.shape[0])) == 1)
        

    # Define A and B matrices to use according to the index of z_v, z_th and z_u_delta
    v_indices = np.linspace(0, len(v_vals)-1, len(v_vals), dtype=int)
    th_indices = np.linspace(0, len(th_vals)-1, len(th_vals), dtype=int)
    u_delta_indices = np.linspace(0, len(u_delta_vals)-1, len(u_delta_vals), dtype=int)
    indices = np.array(np.meshgrid(v_indices, th_indices, u_delta_indices)).T.reshape(-1, 3)

    # system constraints x[k+1] = A X[k]+ B U[k]
    for t in range(time_horizon-1):
        for l, (i,j,k) in enumerate(indices):            
            wstl_milp.model.addConstr(px_possible[l][t+1] == As[i][j][k][0][0] * px[t]   +  As[i][j][k][0][1] * py[t]      +
                                                 As[i][j][k][0][2] * (v[t]-v_vals[i])    +  As[i][j][k][0][3] * (theta[t]-th_vals[j])   +
                                                 Bs[i][j][k][0][0] * u_a[t]  +  Bs[i][j][k][0][1] * (u_delta[t]-u_delta_vals[k]) + fs[i][j][k][0])

            wstl_milp.model.addConstr(py_possible[l][t+1] == As[i][j][k][1][0] * px[t]   +  As[i][j][k][1][1] * py[t]      +
                                                 As[i][j][k][1][2] * (v[t]-v_vals[i])    +  As[i][j][k][1][3] * (theta[t]-th_vals[j])   +
                                                 Bs[i][j][k][1][0] * u_a[t]  +  Bs[i][j][k][1][1] * (u_delta[t]-u_delta_vals[k]) + fs[i][j][k][1])
            
            wstl_milp.model.addConstr(v_possible[l][t+1] ==  As[i][j][k][2][0] * px[t]   +  As[i][j][k][2][1] * py[t]      +
                                                 As[i][j][k][2][2] * (v[t]-v_vals[i])    +  As[i][j][k][2][3] * (theta[t]-th_vals[j])   +
                                                 Bs[i][j][k][2][0] * u_a[t]  +  Bs[i][j][k][2][1] * (u_delta[t]-u_delta_vals[k]) + fs[i][j][k][2])

            wstl_milp.model.addConstr(th_possible[l][t+1] == As[i][j][k][3][0] * px[t]  +  As[i][j][k][3][1] * py[t]      +
                                                    As[i][j][k][3][2] * (v[t]-v_vals[i])   +  As[i][j][k][3][3] * (theta[t]-th_vals[j])   +
                                                    Bs[i][j][k][3][0] * u_a[t] +  Bs[i][j][k][3][1] * (u_delta[t]-u_delta_vals[k]) + fs[i][j][k][3])
        
        # wstl_milp.model.addConstr(px[t+1] == grb.quicksum(z_lin[i][t]*px_possible[i][t+1] for i in range(lin_points.shape[0])))
        # wstl_milp.model.addConstr(py[t+1] == grb.quicksum(z_lin[i][t]*py_possible[i][t+1] for i in range(lin_points.shape[0])))
        # wstl_milp.model.addConstr(v[t+1] == grb.quicksum(z_lin[i][t]*v_possible[i][t+1] for i in range(lin_points.shape[0])))
        # wstl_milp.model.addConstr(theta[t+1] == grb.quicksum(z_lin[i][t]*th_possible[i][t+1] for i in range(lin_points.shape[0])))
        wstl_milp.model.addConstr(px[t+1] == px_possible[0][t+1])
        wstl_milp.model.addConstr(py[t+1] == py_possible[0][t+1])
        wstl_milp.model.addConstr(v[t+1] == v_possible[0][t+1])
        wstl_milp.model.addConstr(theta[t+1] == th_possible[0][t+1])
    

    # Jerk constraints
    wstl_milp.model.addConstr(jerk[0] == (u_a[0] - 0)/T)
    wstl_milp.model.addConstrs(jerk[k] == (u_a[k] - u_a[k-1])/T for k in range(1, time_horizon-1))
    wstl_milp.model.addConstr(sr[0] == (u_delta[0] - 0)/T)
    wstl_milp.model.addConstrs(sr[k] == (u_delta[k] - u_delta[k-1])/T for k in range(1, time_horizon-1))

    # Initial conditions as additional constraints
    wstl_milp.model.addConstr(px[0] == x_0['px'])
    wstl_milp.model.addConstr(py[0] == x_0['py'])
    wstl_milp.model.addConstr(v[0] == x_0['v'])
    wstl_milp.model.addConstr(theta[0] == x_0['theta'])
    wstl_milp.model.addConstrs(px_possible[i][0] == x_0['px'] for i in range(lin_points.shape[0]))
    wstl_milp.model.addConstrs(py_possible[i][0] == x_0['py'] for i in range(lin_points.shape[0]))
    wstl_milp.model.addConstrs(v_possible[i][0] == x_0['v'] for i in range(lin_points.shape[0]))
    wstl_milp.model.addConstrs(th_possible[i][0] == x_0['theta'] for i in range(lin_points.shape[0]))

    # Setpoint constraints
    wstl_milp.model.addConstr(px[time_horizon-1] == 120)#demo['px'][time_horizon-1])
    wstl_milp.model.addConstr(py[time_horizon-1] == 2.5)#demo['py'][time_horizon-1])
    # wstl_milp.model.addConstr(v[time_horizon-1] == 0)
    # wstl_milp.model.addConstr(theta[time_horizon-1] == 0)

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
    wstl_milp.model.addConstrs(j_abs[k] == grb.abs_(jerk[k]) for k in range(time_horizon-1))
    jerk_cost = sum(zeta[0]*j_abs[k] for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(sr_abs[k] == grb.abs_(sr[k]) for k in range(time_horizon-1))
    steering_cost = sum(zeta[1]*sr_abs[k] for k in range(time_horizon-1))

    wstl_milp.model.addConstr(rho_formula >= 0)
    
    wstl_milp.model.setObjective(lambd*rho_formula - state_cost , grb.GRB.MAXIMIZE)#- control_cost - jerk_cost - steering_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()

    return wstl_milp, rho_formula, z_formula


if __name__ == '__main__':  

    # Read demonstration
    x_demo, y_demo, v_demo, th_demo, t_demo = read_demonstration('../carla_settings/demonstrations/trajectory-a_5.csv')
    demo = {'px': x_demo.squeeze(), 'py': y_demo.squeeze(), 'v': v_demo.squeeze(), 'th': th_demo.squeeze()}

    horizon = x_demo.shape[0]-1

    T = 0.2 # sampling time

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
    phi_confort = f"G[0,{horizon-1}]^w2 &&^p2(u_a <= 10, jerk <= 30)"
    phi = f"&&^p1 ({phi_rule}, {phi_confort})"

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 1, 'v': 0, 'theta': -2}
    vars_ub = {'px': 125, 'py': 6.5, 'v': 20, 'theta': 2}
    control_lb = {'u_a': -5, 'u_delta': -1, 'jerk': -40, 'sr': -1}
    control_ub = {'u_a': 15, 'u_delta': 1, 'jerk': 40, 'sr': 1}

    # Define the initial and final conditions
    x_0 = {'px': 0, 'py': 2.5, 'v': 0, 'theta': 0}
    x_f = {'px': 120, 'py': 2.5, 'v': 0, 'theta': 0}

    # Linearization point
    x0 = torch.tensor([0, 0, 0.1, 0]).reshape(1,4) # x, y doesn't matter for linearization
    u0 = torch.tensor([0, 0]).reshape(1,2)

    # Compute a big matrix with A,B matrices for different linearization points as a grid 5x5x5
    model = BicycleModel(dt=T)

    # v_values = [-5, -2.5, -0.2, 0, 0.2, 2.5, 5]
    # th_values = [-0.5, -0.25, 0, 0.25, 0.5]
    # u_delta_values = [-0.5, -0.25, 0, 0.25, 0.5]
    #v_values = [-3.0,-0.2,0.2, 3.0]
    v_values = [0.1, 2.5]
    th_values = [0, 0.25]
    u_delta_values = [0]

    # v_values = [0.1]
    # th_values = [0]
    # u_delta_values = [0]

    v_i = len(v_values)
    th_i = len(th_values)
    ud_i = len(u_delta_values)

    A = np.zeros((v_i, th_i, ud_i, 4, 4))
    B = np.zeros((v_i, th_i, ud_i, 4, 2))
    f = np.zeros((v_i, th_i, ud_i, 4))

    for i, v in enumerate(v_values):
        for j, th in enumerate(th_values):
            for k, u_delta in enumerate(u_delta_values):
                x0 = torch.tensor([[0, 0, v, th]])
                u0 = torch.tensor([[0, u_delta]])
                Ad, Bd = model.discretize_dynamics(x0, u0)
                Ad, Bd = Ad.detach().numpy(), Bd.detach().numpy()
                f0 = model.integrate_dynamics(x0, u0).detach().numpy().reshape(4)
                A[i, j, k] = Ad
                B[i, j, k] = Bd
                f[i, j, k] = f0
    
    # Define the pedestrian model
    class pedestrian:
        def __init__(self):
            self.x_ped = 116.0 #+ 5*np.random.rand()
            self.y_ped = 17.0
        def __call__(self, t):
            if t <= 20:
                vel = 0.9
                self.x_ped = self.x_ped #+ 0.1*np.random.randn(1)
                self.y_ped = 17.0 - vel*t
            return [self.x_ped, self.y_ped] 
    ped = pedestrian()

    alpha = np.array([0.001, 0.005, 0.05, 0.001])#np.array([0.002, 0.005, 0.0001, 0.01])#np.array([0.5, 0.05, 0.5, 0.01])
    beta = np.array([0.0001, 0.005])
    zeta =  np.array([0.01, 0.0001])
    lambd = 1

    # Translate WSTL to MILP and retrieve integer variable for the formula
    stl_start = time.time()
    stl_milp, rho_formula, z = wstl_synthesis_control(phi, weights, ped, f, A, B, v_values, th_values, u_delta_values, T, 
                                                      vars_lb, vars_ub, control_lb, control_ub, x_0, demo, alpha, beta, zeta, lambd)
    stl_end = time.time()
    stl_time = stl_end - stl_start

    print(f"Robustness: {rho_formula.x}")
    print('Time needed:', stl_time)
    print("Time horizon:", horizon)
    print("z:", z.x)

    # for t in range(horizon):
    #     print(stl_milp.model.getVarByName("inter_0_" + str(t)).x, 
    #           stl_milp.model.getVarByName("inter_1_" + str(t)).x,
    #           stl_milp.model.getVarByName("inter_2_" + str(t)).x,
    #           stl_milp.model.getVarByName("inter_3_" + str(t)).x,
    #           stl_milp.model.getVarByName("inter_4_" + str(t)).x,
    #           stl_milp.model.getVarByName("inter_5_" + str(t)).x,
    #           stl_milp.model.getVarByName("inter_6_" + str(t)).x,
    #           stl_milp.model.getVarByName("inter_7_" + str(t)).x)
        # print()

    # Visualize the results
    plot_multi_vars(stl_milp, ['px', 'py', 'v', 'theta'], T)
    plot_multi_vars(stl_milp, ['u_a', 'u_delta'], T)
    ani = visualize_demo_and_stl(x_demo, y_demo, stl_milp, T)
    # save_vid(ani, "anim/carla.gif")

    # Create a csv file with the trajectory of the car for the whole horizon, were each row is a different time containing x and y
    x_traj = np.zeros(horizon)
    y_traj = np.zeros(horizon)
    v_traj = np.zeros(horizon)
    th_traj = np.zeros(horizon)
    for i in range(horizon):
        x_traj[i] = stl_milp.model.getVarByName('px_' + str(i)).x
        y_traj[i] = stl_milp.model.getVarByName('py_' + str(i)).x
        v_traj[i] = stl_milp.model.getVarByName('v_' + str(i)).x
        th_traj[i] = stl_milp.model.getVarByName('theta_' + str(i)).x
    #np.savetxt('../carla_settings/preference_synthesis/carla_traj.csv', np.vstack((x_traj, y_traj, v_traj)).T, delimiter=',')

    # save last animation frame as a png
    # save_vid(ani, "anim/lambda_1.png")