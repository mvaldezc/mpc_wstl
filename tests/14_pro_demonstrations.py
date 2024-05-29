from antlr4 import InputStream, CommonTokenStream
import time
from visualize import visualize_demo_and_stl, plot_multi_vars, save_vid, plot_multi_vars_mpc, visualize_multiple
from demonstrations import read_pro_demonstrations
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
    
    wstl_milp.model.setObjective(lambd*rho_formula, grb.GRB.MAXIMIZE)#- control_cost - jerk_cost - steering_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()
    return wstl_milp, rho_formula, z_formula


if __name__ == '__main__':  

    # Read demonstration
    x_demo, y_demo, v_demo, th_demo, t_demo, x_ped, y_ped = read_pro_demonstrations(2)
    demo = {'px': x_demo.squeeze(), 'py': y_demo.squeeze(), 'v': v_demo.squeeze(), 'th': th_demo.squeeze()}

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
    
    W2 = {
        "w1" : lambda k : [1.0883, 0.6928, 0.3473, 0.9792, 0.2450, 0.7883, 0.4861, 0.4759, 0.3067,
        0.1386, 0.1696, 0.1146, 0.7138, 0.4675, 0.8649, 0.9011, 0.6475, 0.7120,
        0.8644, 0.3407, 0.6326, 0.9444, 1.0191, 0.1805, 0.8252, 1.0541, 0.7816,
        0.7204, 0.8411, 0.7058, 0.2704, 0.9529, 0.5694, 0.1646, 0.4609, 0.9386,
        0.9535, 1.0082, 0.3296, 0.7905, 0.2065, 0.4261, 0.2265, 0.5642, 0.4121,
        0.9713, 0.5755, 0.3277, 0.4301, 1.0397, 0.4791, 0.9973, 0.3128, 1.0057,
        0.7096, 0.2603, 0.1022, 0.9569, 0.1551, 0.6936, 0.9723, 0.5476, 0.3062,
        1.0942, 1.0570, 0.5089, 0.8414, 1.0496, 0.5917, 1.0818, 0.6306, 0.6233,
        0.1635, 0.3553, 0.9171, 1.0638, 0.6672, 0.8289, 1.0600, 0.6644, 0.5416,
        0.1720, 0.5081, 0.7351, 0.5004, 0.3584, 0.2566, 0.1932, 0.9398, 0.8522,
        0.4381, 1.0241, 0.8505, 0.2480, 0.5669, 0.4464, 1.0818, 0.1879, 1.0340,
        0.5480, 0.2751, 0.3751, 0.5850, 0.4566, 0.8084, 0.4189, 0.3431, 1.0567,
        0.8534, 0.9223, 0.9831, 1.0515, 0.8729, 0.1703, 1.0503, 0.2764, 1.0639,
        0.3330, 0.8162, 0.9108, 0.9501, 0.6385, 1.0272, 0.7955, 0.2761, 0.7496,
        0.6207, 0.5702, 0.2005, 1.0290, 0.1012, 0.7991, 0.2998][k], 
        "w2" : lambda k : [0.7535, 1.0379, 0.4478, 0.8925, 0.5037, 0.8823, 0.7940, 0.5869, 0.3555,
        0.1258, 0.9732, 1.0111, 0.7356, 0.2991, 0.2374, 0.4935, 0.2337, 0.6030,
        0.5769, 0.2056, 0.7437, 0.4421, 0.9333, 0.5033, 0.5745, 1.0103, 0.3093,
        0.7560, 0.5248, 0.3176, 0.9789, 0.6314, 1.0206, 0.2468, 0.9889, 0.8684,
        0.8517, 0.1513, 1.0013, 0.6274, 0.3027, 0.3672, 0.9616, 0.9445, 0.7645,
        0.9730, 0.9404, 0.5446, 0.1896, 1.0168, 1.0579, 0.7442, 0.7322, 0.7693,
        0.5996, 0.5682, 0.7794, 0.3294, 0.5157, 0.8355, 1.0046, 0.2713, 0.5298,
        0.1052, 0.6088, 1.0153, 0.2407, 0.3797, 0.9424, 0.5763, 0.1024, 1.0174,
        1.0023, 1.0350, 0.1259, 0.9321, 0.7758, 0.9110, 0.3467, 0.6136, 0.4995,
        0.5727, 0.6617, 0.4390, 0.3339, 1.0633, 0.9601, 0.3203, 0.3765, 0.4677,
        0.4315, 0.7166, 0.4826, 0.8277, 0.8433, 0.1154, 0.2113, 0.8350, 0.6956,
        0.7343, 0.9682, 0.9423, 0.5270, 0.5430, 0.9451, 0.3956, 0.4055, 0.2756,
        0.6497, 0.4653, 0.8567, 0.1058, 0.7928, 0.9632, 0.5174, 0.4265, 0.7878,
        0.1595, 0.7756, 0.9444, 0.7889, 0.2198, 0.5515, 1.0493, 0.5089, 0.4971,
        0.3590, 0.5605, 0.4066, 1.0200, 0.1487, 0.6839, 0.9597][k], 
        "w3" : lambda k : [0.4862, 0.8633, 0.4024, 0.2744, 0.4045, 0.6676, 0.6576, 0.4214, 0.5857,
        0.1670, 0.6578, 0.7073, 1.0075, 0.2245, 0.2234, 0.6231, 0.4287, 0.5346,
        0.2889, 0.7947, 0.7587, 1.0022, 0.7983, 0.6095, 0.4621, 0.6241, 0.7512,
        0.9957, 0.6932, 1.0318, 0.6478, 0.6885, 0.1909, 0.1236, 0.9425, 0.1387,
        1.0380, 0.2607, 0.2327, 0.3607, 1.0264, 0.1516, 0.9151, 0.1749, 0.6286,
        0.1067, 0.1169, 0.9587, 0.4422, 1.0172, 0.5864, 0.4389, 0.3089, 0.2573,
        0.3072, 1.0795, 0.7900, 0.1180, 0.9572, 1.0174, 0.2064, 1.0771, 0.3477,
        0.1176, 0.4004, 1.0384, 0.4473, 0.1886, 0.3952, 0.7983, 0.7605, 0.5373,
        0.1056, 0.6688, 0.2872, 0.1161, 0.8754, 0.3494, 0.5220, 0.3272, 0.2913,
        0.2345, 0.8456, 0.7635, 0.5024, 0.1973, 1.0380, 0.3072, 0.1129, 0.6662,
        0.2718, 0.9438, 0.9407, 0.6682, 0.7649, 0.1145, 0.4891, 1.0946, 0.1730,
        0.4894, 0.2648, 0.3456, 1.0442, 0.1392, 0.7528, 0.7431, 0.4389, 0.2756,
        0.4064, 0.6926, 0.3040, 0.4865, 0.9475, 0.5577, 0.3108, 0.2129, 0.4567,
        0.4002, 1.0100, 0.6143, 0.9824, 0.1739, 0.6810, 0.7851, 0.7385, 0.4492,
        0.6986, 0.1620, 1.0816, 0.1061, 0.2669, 1.0352, 0.7471][k],
        "w4" : lambda k : [0.9893, 0.7597, 0.7519, 0.4525, 0.5759, 0.1758, 0.6048, 0.6425, 0.1070,
        0.3670, 0.1956, 0.9473, 1.0920, 0.3004, 0.3636, 0.5798, 0.5893, 0.2729,
        0.8341, 0.4146, 0.5490, 0.5802, 0.9843, 0.2047, 0.1481, 0.7515, 0.3562,
        0.2505, 0.3070, 1.0197, 0.6384, 0.8254, 0.5828, 0.1336, 0.2174, 0.8112,
        0.4072, 0.6852, 1.0505, 1.0810, 0.2125, 0.5505, 0.7541, 0.1870, 1.0105,
        0.5539, 0.8372, 0.6158, 0.9917, 0.1885, 0.7601, 0.9762, 0.8624, 0.3599,
        0.8864, 0.2489, 0.5639, 0.7312, 0.6078, 0.6455, 0.8934, 0.3495, 0.5344,
        0.8242, 1.0429, 1.0638, 1.0077, 0.3038, 0.9492, 0.2690, 0.5052, 1.0864,
        1.0322, 0.9921, 0.4413, 0.7232, 0.8442, 0.8801, 0.6517, 0.9458, 0.4446,
        0.6338, 0.5969, 0.2585, 0.8265, 0.9693, 0.6909, 0.9634, 0.4551, 0.2353,
        0.1861, 1.0713, 0.2623, 0.7971, 0.9684, 0.3647, 0.6551, 0.5309, 1.0013,
        0.3804, 0.8678, 0.2408, 0.1535, 0.9913, 0.5434, 0.8098, 0.3041, 0.8639,
        0.7659, 0.4782, 1.0067, 0.3657, 0.3850, 0.2160, 0.9319, 0.4371, 0.1894,
        0.3871, 0.2132, 0.5971, 0.2690, 0.3983, 0.8569, 0.6367, 0.4646, 0.1949,
        0.4001, 0.9691, 0.7438, 0.3081, 0.4417, 0.7154, 0.5092][k],
        "w5" : lambda k : [0.9439, 0.9929, 1.0306, 0.2044, 0.5630, 0.3309, 0.1191, 0.9786, 0.7874,
        0.1210, 0.7865][k],
        "p1" : lambda k : [0.2037, 0.8957][k],
        "p2" : lambda k : [0.9405, 1.0665][k],
        "p3" : lambda k : [0.4746, 0.7092][k],
        "p4" : lambda k : [82.4754, 43.7077][k]
        }
    
    W3 = {
        "w1" : lambda k : [0.6392, 0.5688, 1.0372, 0.8930, 0.5444, 0.4754, 0.9873, 0.1692, 0.1257,
        0.7593, 0.4679, 0.6567, 0.6349, 1.0924, 0.5132, 0.5424, 0.9524, 0.6668,
        0.7026, 0.8570, 0.7321, 0.1253, 0.6630, 1.0362, 0.1899, 0.2378, 0.8085,
        0.9786, 0.3455, 0.7455, 0.1339, 0.4850, 0.3845, 0.8401, 0.3398, 0.8895,
        0.4744, 0.2692, 0.6164, 0.1381, 0.4838, 1.0953, 1.0242, 0.5717, 0.3469,
        0.4906, 0.8915, 0.5333, 0.9807, 0.4199, 0.3698, 1.0123, 0.4013, 0.2302,
        1.0688, 0.1519, 1.0980, 0.3474, 0.3419, 0.3525, 0.7929, 1.0599, 0.7009,
        0.3718, 0.3039, 0.6381, 0.8038, 0.6303, 0.2297, 0.4198, 0.6001, 0.3750,
        0.4543, 0.6659, 0.8337, 0.1625, 0.9297, 0.2029, 1.0360, 1.0102, 0.9408,
        0.4798, 0.2957, 0.4237, 0.6685, 0.2114, 0.5107, 0.9137, 1.0268, 0.8693,
        0.7318, 0.4890, 0.1810, 0.7137, 0.8788, 0.1297, 0.2916, 0.3160, 1.0205,
        0.7211, 0.5054, 1.0245, 0.2496, 0.3405, 0.9851, 0.6282, 0.4241, 0.4757,
        0.1619, 1.0075, 0.2712, 0.1564, 0.4853, 0.7862, 0.5470, 0.5666, 0.6230,
        0.7608, 0.1156, 0.6078, 0.2074, 0.7920, 0.6309, 0.1791, 0.1051, 1.0684,
        0.2569, 0.8303, 0.4913, 0.5739, 0.8352, 0.5979, 0.2322][k], 
        "w2" : lambda k : [0.8599, 0.4465, 0.6622, 0.3500, 0.1123, 0.4332, 0.5939, 0.7725, 0.8565,
        0.4608, 0.5742, 0.8282, 0.9588, 0.9701, 0.1172, 0.5096, 0.7563, 0.9786,
        1.0921, 1.0100, 1.0272, 1.0737, 0.7480, 0.4723, 0.4239, 0.1737, 0.4325,
        0.4626, 0.4752, 0.2459, 0.8816, 0.6420, 1.0248, 1.0933, 0.5735, 0.8491,
        0.6946, 0.2664, 0.8956, 0.6541, 0.8435, 0.8988, 0.1681, 0.7675, 0.2202,
        0.1569, 0.5477, 0.4559, 0.1546, 1.0733, 0.2629, 0.4419, 0.4401, 0.8755,
        0.1623, 0.8983, 0.1325, 0.9485, 0.6998, 1.0919, 0.8531, 0.4189, 0.2920,
        0.3716, 0.9580, 0.3050, 0.8646, 0.9371, 0.3024, 0.7139, 1.0556, 0.1998,
        1.0591, 0.5665, 0.6653, 0.7011, 0.1032, 1.0563, 0.5514, 0.4781, 0.8295,
        0.5850, 0.3620, 0.6148, 0.3358, 0.7934, 0.7777, 0.4923, 0.7702, 0.7134,
        0.6496, 0.9670, 0.5707, 0.4638, 0.3807, 0.2422, 0.8004, 0.7968, 0.8192,
        0.4181, 0.7731, 0.2561, 0.7677, 0.1347, 0.4566, 0.7809, 0.8168, 0.3577,
        0.6519, 0.3307, 0.4741, 0.4584, 0.9882, 0.2886, 0.7012, 0.6487, 0.4285,
        0.6211, 0.8817, 0.1595, 0.2830, 0.1950, 0.5932, 1.0285, 0.3890, 0.3688,
        0.8114, 0.9806, 0.4442, 0.9554, 0.4793, 0.3956, 0.3524][k], 
        "w3" : lambda k : [0.7410, 0.2067, 1.0291, 0.5550, 0.9589, 0.2690, 0.5218, 0.6061, 1.0068,
        0.7209, 0.2257, 0.1464, 0.9919, 0.2748, 0.9603, 0.1571, 0.6089, 0.1060,
        0.6165, 1.0724, 0.8891, 0.7672, 0.8055, 0.7115, 0.2158, 0.5699, 0.8582,
        0.7956, 0.8561, 0.3035, 0.8370, 0.7352, 0.2519, 0.5719, 0.8428, 0.8539,
        0.3840, 0.9249, 0.3087, 0.2458, 0.1440, 0.5189, 1.0394, 0.1506, 0.1556,
        0.5276, 0.1455, 0.6976, 0.9192, 0.3606, 0.9291, 0.6690, 0.9147, 0.5046,
        0.4502, 0.4785, 0.8390, 0.5580, 0.4659, 0.1950, 1.0953, 0.7647, 0.5928,
        0.5413, 0.4302, 0.4242, 0.4192, 1.0127, 0.6249, 0.9911, 0.6812, 0.3342,
        0.9509, 0.7652, 0.9041, 0.4172, 0.8301, 0.2096, 0.6841, 0.8476, 0.8704,
        0.8611, 0.4827, 0.1077, 0.7517, 0.3402, 0.6850, 0.7872, 0.4446, 0.9463,
        0.1990, 0.6765, 0.3411, 0.7703, 0.7183, 0.1570, 1.0267, 0.5826, 0.3484,
        0.6740, 0.8404, 0.7184, 0.5443, 0.7238, 1.0209, 0.8236, 0.8206, 1.0530,
        1.0625, 0.5406, 0.2478, 0.4393, 0.9985, 0.9515, 0.4618, 0.4427, 0.7569,
        0.7626, 0.5887, 0.2113, 0.1695, 0.9124, 0.2866, 0.7384, 0.8452, 1.0760,
        0.3530, 1.0359, 0.7882, 0.3534, 0.9182, 0.1425, 0.8828][k],
        "w4" : lambda k : [0.5140, 0.9268, 0.5283, 0.4572, 0.8071, 0.3055, 0.9470, 0.4079, 0.7100,
        0.7152, 0.2173, 0.8358, 0.9103, 0.2097, 0.6671, 0.9558, 0.2107, 0.6352,
        0.2409, 0.6225, 1.0723, 0.1399, 1.0365, 0.9242, 0.3212, 0.1245, 0.1287,
        0.2072, 0.2665, 0.9087, 0.5715, 0.3004, 0.6839, 0.9503, 0.7039, 0.9893,
        0.2959, 0.3756, 0.4859, 0.6137, 0.5667, 0.2650, 0.3515, 0.1684, 0.5205,
        0.6854, 1.0092, 0.5856, 0.2869, 0.4965, 0.8300, 0.4609, 0.8317, 0.6471,
        0.1718, 0.4670, 1.0628, 0.4184, 0.9016, 1.0647, 0.2186, 0.5991, 0.4262,
        0.6199, 0.3306, 0.5620, 0.3745, 0.3206, 0.9340, 0.7689, 0.7271, 0.3627,
        0.9045, 0.8721, 0.4972, 0.4423, 0.1223, 0.1351, 0.6003, 0.7166, 1.0636,
        0.3734, 0.3087, 0.3657, 0.9295, 0.6884, 0.7057, 1.0685, 1.0401, 0.6028,
        0.2166, 1.0499, 0.5267, 0.3522, 0.5127, 0.1274, 0.9748, 0.9189, 0.9907,
        1.0084, 1.0132, 1.0620, 0.7714, 0.8493, 0.9785, 0.6271, 0.8339, 0.4468,
        0.3316, 0.9137, 0.8972, 0.6986, 0.1984, 0.1009, 0.4726, 0.8547, 1.0559,
        0.5351, 0.1204, 0.6277, 0.8709, 0.4791, 0.2334, 0.8250, 0.7372, 0.8857,
        0.7449, 1.0579, 0.3314, 0.9068, 0.6073, 0.2514, 0.2870][k],
        "w5" : lambda k : [0.4994, 0.6864, 0.1573, 1.0927, 0.8275, 0.6522, 0.8480, 0.1428, 1.0441,
        0.9208, 0.8220][k],
        "p1" : lambda k : [0.3016, 0.8629][k],
        "p2" : lambda k : [0.1269, 0.2461][k],
        "p3" : lambda k : [0.6757, 0.3612][k],
        "p4" : lambda k : [5833.6202, 2032.0262][k]
        }

    phi_rule = f"G[0,{horizon}]^w1 (distance >= 2)"
    phi_confort = f"&&^p2((G[0,{horizon-1}]^w2 (u_a <= 10)), (G[0,{horizon-1}]^w3 (j <= 30)))"
    phi_destination = f"F[{horizon-20},{horizon-10}]^w4 G[0,10]^w5 &&^p3(px >= 98, px <= 140)"#(px >= 118, px <= 124)"
    phi = f"&&^p4 (&&^p1 ({phi_rule}, {phi_confort}), {phi_destination}))"

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 1, 'v': 0, 'theta': -np.pi}
    vars_ub = {'px': 150, 'py': 4.5, 'v': 30, 'theta': np.pi}
    control_lb = {'u_a': -10, 'u_delta': -1, 'j': -80, 'sr': -1}
    control_ub = {'u_a': 15, 'u_delta': 1, 'j': 80, 'sr': 1}

    # Define the initial conditions
    x_0 = {'px': 0, 'py': 2.5, 'v': 0.1, 'theta': 0}

    # Linearization point
    x0 = torch.tensor([0, 0, 0.5, 0]).reshape(1,4) # x, y doesn't matter for linearization
    u0 = torch.tensor([0, 0]).reshape(1,2)
    lin = {'v': x0[0,2], 'th': x0[0,3]}

    # Define the matrices for linear system 
    model = BicycleModel(dt=T)
    Ad, Bd = model.discretize_dynamics(x0, u0)
    Ad, Bd = Ad.detach().numpy(), Bd.detach().numpy()
    f0 = model.integrate_dynamics(x0, u0).detach().numpy().reshape(4)

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
    stl_milp_1, rho_formula_1, z_1 = wstl_synthesis_control(phi, W1, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                    control_ub, x_0, lin, demo, alpha, beta, zeta, lambd)
    
    stl_milp_2, rho_formula_2, z_2 = wstl_synthesis_control(phi, W2, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                    control_ub, x_0, lin, demo, alpha, beta, zeta, lambd)
    
    stl_milp_3, rho_formula_3, z_3 = wstl_synthesis_control(phi, W3, ped, f0, Ad, Bd, T, vars_lb, vars_ub, control_lb, 
                                    control_ub, x_0, lin, demo, alpha, beta, zeta, lambd)
    
    print(f"Robustness 1:", rho_formula_1.x, "z:", z_1.x)
    print(f"Robustness 2:", rho_formula_2.x, "z:", z_2.x)
    print(f"Robustness 3:", rho_formula_3.x, "z:", z_3.x)

    # Visualize the results
    state_var_name = ['px', 'py', 'v', 'theta']
    px_1 = np.array([stl_milp_1.model.getVarByName('px_' + str(i)).x for i in range(horizon+1)])
    py_1 = np.array([stl_milp_1.model.getVarByName('py_' + str(i)).x for i in range(horizon+1)])
    v_1 = np.array([stl_milp_1.model.getVarByName('v_' + str(i)).x for i in range(horizon+1)])
    theta_1 = np.array([stl_milp_1.model.getVarByName('theta_' + str(i)).x for i in range(horizon+1)])
    state_var_1 = np.vstack((px_1, py_1, v_1, theta_1))

    px_2 = np.array([stl_milp_2.model.getVarByName('px_' + str(i)).x for i in range(horizon+1)])
    py_2 = np.array([stl_milp_2.model.getVarByName('py_' + str(i)).x for i in range(horizon+1)])
    v_2 = np.array([stl_milp_2.model.getVarByName('v_' + str(i)).x for i in range(horizon+1)])
    theta_2 = np.array([stl_milp_2.model.getVarByName('theta_' + str(i)).x for i in range(horizon+1)])
    state_var_2 = np.vstack((px_2, py_2, v_2, theta_2))

    px_3 = np.array([stl_milp_3.model.getVarByName('px_' + str(i)).x for i in range(horizon+1)])
    py_3 = np.array([stl_milp_3.model.getVarByName('py_' + str(i)).x for i in range(horizon+1)])
    v_3 = np.array([stl_milp_3.model.getVarByName('v_' + str(i)).x for i in range(horizon+1)])
    theta_3 = np.array([stl_milp_3.model.getVarByName('theta_' + str(i)).x for i in range(horizon+1)])
    state_var_3 = np.vstack((px_3, py_3, v_3, theta_3))
    # plot_multi_vars_mpc(state_var_name, state_var, T, state_var_demo)
    # plot_multi_vars(stl_milp, ['u_a', 'u_delta'], T)
    # region = [118, 124, 1, 4.5]
    region = [98, 140, 1, 4.5]
    # ani = visualize_demo_and_stl(x_demo, y_demo, stl_milp_1, T)
    ani = visualize_multiple(x_demo, y_demo, stl_milp_1, stl_milp_2, stl_milp_3, region, T)

    # Create a csv file with the trajectory of the car for the whole horizon, were each row is a different time containing x and y
    x_traj = np.zeros(horizon)
    y_traj = np.zeros(horizon)
    v_traj = np.zeros(horizon)
    th_traj = np.zeros(horizon)
    for i in range(horizon):
        x_traj[i] = stl_milp_2.model.getVarByName('px_' + str(i)).x
        y_traj[i] = stl_milp_2.model.getVarByName('py_' + str(i)).x
        v_traj[i] = stl_milp_2.model.getVarByName('v_' + str(i)).x
        th_traj[i] = stl_milp_2.model.getVarByName('theta_' + str(i)).x
    np.savetxt('../carla_settings/preference_synthesis/carla_traj_3.csv', np.vstack((x_traj, y_traj, v_traj)).T, delimiter=',')

    # save last animation frame as a png
    # save_vid(ani, "anim/lambda_1.png")