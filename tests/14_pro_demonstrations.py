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
    
    wstl_milp.model.setObjective(lambd*rho_formula - state_cost, grb.GRB.MAXIMIZE)#- control_cost - jerk_cost - steering_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()
    return wstl_milp, rho_formula, z_formula


if __name__ == '__main__':  

    # Read demonstration
    x_demo, y_demo, v_demo, th_demo, t_demo, x_ped, y_ped = read_pro_demonstrations(0)
    demo = {'px': x_demo.squeeze(), 'py': y_demo.squeeze(), 'v': v_demo.squeeze(), 'th': th_demo.squeeze()}

    horizon = x_demo.shape[0]-1

    T = 0.1 # sampling time

    # Define wSTL specification
    W1 = {"w1" : lambda k : 1.0, 
        "w2" : lambda k : 1.0, 
        "w3" : lambda k : 1.0,
        "w4" : lambda k : 1.0,
        "p1" : lambda k : [1.0, 1.0, 1.0][k],
        "p2" : lambda k : [1.0, 1.0][k],
        "p3" : lambda k : 0.4
        }
    
    W2 = {"w1" : lambda k : [0.6591, 0.2479, 0.7001, 0.3661, 0.3714, 0.5254, 0.8606, 0.8964, 0.7174,
        0.1646, 0.6573, 0.9197, 0.8310, 0.6905, 0.7640, 0.8672, 0.3870, 0.7378,
        0.9523, 0.7714, 0.9404, 0.5060, 0.7217, 0.3997, 0.7891, 0.2067, 1.0854,
        0.3575, 0.6924, 0.8328, 0.7094, 0.2280, 0.8337, 0.4518, 0.4550, 0.5685,
        1.0855, 0.7583, 0.5062, 0.6923, 0.2469, 0.1060, 0.8191, 1.0572, 0.8916,
        0.5959, 0.3530, 0.2141, 1.0349, 0.7534, 1.0362, 0.3864, 0.5278, 1.0830,
        0.7200, 0.1429, 0.9137, 0.3696, 0.7996, 0.3259, 0.4591, 0.9511, 1.0480,
        0.2279, 0.3048, 0.4480, 0.8305, 0.9135, 0.2044, 0.2135, 0.2284, 0.7177,
        1.0485, 0.7703, 0.4410, 0.4179, 0.7212, 0.2247, 1.0268, 0.7489, 0.5164,
        0.1633, 0.9217, 0.6974, 0.2526, 0.6498, 0.4977, 0.6662, 0.4460, 0.1238,
        0.7860, 0.6323, 0.9684, 0.3372, 0.4001, 1.0208, 0.3299, 0.5833, 0.2961,
        0.1472, 0.7732, 0.9798, 0.1397, 0.8021, 0.7228, 0.4708, 1.0019, 0.4650,
        0.7686, 0.7784, 0.6006, 0.9024, 0.8881, 0.5671, 0.7605, 0.7920, 0.1732,
        1.0006, 0.6923, 0.6946, 1.0280, 0.1704, 0.7731, 0.5674, 0.3564, 0.8753,
        0.4347, 0.9864, 0.3881, 0.2028, 0.1126, 0.3263, 1.0254, 0.8457, 0.8598,
        0.2539, 0.4787, 0.9216, 0.5608, 0.7101, 0.4316, 0.1077, 0.7914, 0.3239,
        0.7994, 1.0543, 0.4173, 0.9701, 0.5051, 1.0587, 0.4297, 1.0364][k], 
        "w2" : lambda k : [0.3963, 0.3712, 0.8803, 1.0302, 0.7638, 0.3440, 0.2628, 1.0234, 0.5086,
        1.0054, 0.6984, 0.5897, 0.7873, 0.8615, 0.3946, 0.9238, 0.5924, 0.7453,
        0.1626, 0.2346, 0.1442, 0.9294, 0.3495, 0.6443, 0.2803, 0.5208, 0.2795,
        0.3723, 0.5837, 0.1555, 0.3180, 0.6137, 1.0985, 1.0475, 0.9363, 0.1651,
        0.6328, 0.6047, 0.5556, 0.7999, 0.3395, 0.9379, 0.2729, 0.3036, 0.4716,
        0.5984, 0.8057, 0.1610, 0.6307, 0.8502, 0.2420, 0.7186, 0.3576, 0.6489,
        0.4777, 0.8648, 0.4455, 0.3923, 1.0113, 0.6567, 0.2606, 0.9818, 0.8770,
        0.6566, 0.8486, 0.4905, 0.5602, 0.2038, 0.8588, 0.1448, 0.5763, 0.7972,
        0.3920, 0.5504, 0.2051, 0.8299, 0.1935, 0.3442, 0.2893, 1.0495, 0.2503,
        0.3893, 0.8831, 0.1643, 0.3368, 0.3122, 0.2380, 0.1490, 0.1466, 0.3288,
        0.9479, 1.0105, 0.2964, 0.2508, 0.2690, 0.7026, 0.3838, 0.5749, 0.1724,
        1.0343, 0.4168, 0.1365, 0.9048, 0.1554, 0.4879, 0.3948, 0.4661, 0.9610,
        0.7210, 0.9759, 0.6936, 0.9323, 0.6650, 0.2795, 1.0420, 0.6506, 0.2162,
        0.6154, 0.3708, 0.8556, 0.5332, 0.6643, 0.6954, 0.7338, 0.2661, 0.9437,
        0.4900, 1.0696, 1.0024, 1.0788, 0.4637, 0.3807, 0.2062, 0.3870, 0.4054,
        0.9012, 1.0114, 0.2477, 0.8863, 0.4843, 0.9123, 0.6751, 0.6145, 0.3001,
        0.5029, 0.3837, 0.5436, 0.4763, 1.0404, 0.7888, 0.6217, 0.6293][k], 
        "w3" : lambda k : [0.2577, 0.4514, 0.7863, 0.5857, 0.1287, 1.0296, 0.1322, 0.6367, 0.1610,
        0.7671, 0.8217, 0.3034, 0.7517, 0.9544, 0.2769, 0.5833, 0.3270, 1.0140,
        0.3115, 0.7615, 0.7416, 0.6422, 0.3597, 0.6400, 0.4977, 0.4184, 0.1788,
        0.8717, 0.7218, 0.7427, 1.0797, 1.0329, 0.5838, 0.3853, 1.0537, 0.2987,
        0.2001, 0.8096, 1.0851, 0.6280, 0.3829, 0.6179, 0.8253, 0.6158, 0.6204,
        0.5880, 1.0013, 0.7779, 0.5593, 0.5475, 0.1713, 0.3290, 0.6464, 0.4008,
        0.8563, 1.0398, 0.1152, 0.8237, 0.1673, 1.0368, 0.3919, 0.3552, 0.6833,
        0.6666, 0.9801, 0.2399, 0.7514, 0.5146, 1.0860, 1.0880, 0.1699, 0.9902,
        0.3479, 0.5570, 0.4125, 1.0961, 0.1046, 0.5440, 0.6756, 0.8897, 0.4567,
        0.8543, 0.5227, 0.7018, 0.3001, 0.7984, 1.0054, 0.8649, 0.9937, 1.0822,
        0.7906, 0.8576, 1.0751, 0.2140, 0.7572, 0.9191, 1.0393, 0.2824, 0.4122,
        0.7922, 0.9179, 0.9562, 0.9410, 0.9364, 1.0441, 0.2638, 1.0490, 0.8577,
        0.2310, 0.7380, 0.2538, 0.7221, 0.5548, 0.8427, 0.2320, 1.0966, 0.3874,
        0.6604, 0.5209, 0.7682, 1.0182, 0.7657, 0.6137, 0.8801, 0.4778, 0.4095,
        0.3708, 0.7690, 0.5239, 0.2168, 0.4400, 0.8911, 0.2738, 0.5073, 0.8890,
        0.1113, 0.6135, 0.4458, 0.5641, 0.3671, 0.4413, 0.1309, 0.8907, 0.7061,
        0.9627, 0.1163, 0.7948, 0.8119, 1.0329, 0.3034, 0.7437, 0.3281][k],
        "w4" : lambda k : 1.0,
        "p1" : lambda k : [337.1116, 1102.9098, 300.0][k],
        "p2" : lambda k : [0.7504, 0.1153][k],
        "p3" : lambda k : 0.4
        }
    
    W3 = {"w1" : lambda k : [0.4074, 0.8898, 0.5452, 0.7681, 0.9910, 1.0426, 0.6157, 1.0767, 0.3227,
        0.9105, 0.8823, 0.9255, 0.7573, 0.5502, 0.2570, 0.6417, 0.7025, 0.8202,
        1.0947, 0.7050, 0.5340, 0.5811, 1.0973, 0.8901, 0.9520, 0.1976, 0.7315,
        1.0053, 0.5676, 0.4995, 0.8583, 0.8381, 1.0253, 0.8581, 0.7672, 0.2239,
        0.3857, 0.7170, 0.6068, 0.1745, 0.4693, 0.9261, 0.9976, 0.8745, 0.7303,
        0.9978, 0.1742, 0.7567, 0.9868, 0.5360, 0.7526, 0.9272, 0.2297, 0.4580,
        0.9806, 0.7142, 0.6396, 0.4802, 0.6270, 0.7701, 0.4072, 0.6792, 0.6313,
        0.1392, 0.9568, 0.1285, 0.1408, 1.0016, 0.9420, 0.7969, 0.4595, 0.1665,
        0.7312, 0.3461, 0.2961, 0.9338, 0.1960, 0.9687, 0.2890, 0.6721, 0.8071,
        0.3145, 0.6024, 0.3716, 0.4964, 0.8446, 0.5661, 0.5528, 0.7804, 0.1724,
        0.4233, 0.4233, 0.3671, 0.3103, 0.9483, 0.4098, 0.5871, 0.9842, 0.7223,
        0.1500, 0.4646, 0.4288, 0.3534, 0.6453, 0.4708, 0.9820, 1.0830, 0.6163,
        0.9099, 0.3614, 0.8553, 0.2213, 0.4226, 0.8734, 0.2312, 0.9051, 0.5087,
        0.6959, 1.0063, 0.1233, 0.6228, 0.9325, 0.8204, 0.6931, 0.4759, 0.3628,
        0.4209, 0.6940, 0.9281, 0.9355, 0.5953, 0.9958, 0.2700, 0.9342, 0.8775,
        0.7120, 0.4533, 0.1016, 0.7846, 0.4955, 0.7394, 0.9304, 0.5970, 0.7415,
        0.5169, 0.1823, 1.0498, 0.4143, 0.2056, 0.2357, 0.9082, 0.6245][k], 
        "w2" : lambda k : [0.6684, 0.2995, 0.8426, 0.3190, 1.0798, 0.6465, 0.6563, 0.9239, 0.2450,
        0.4510, 0.8170, 0.9226, 0.9777, 0.7764, 0.2619, 0.2768, 0.9816, 0.9807,
        0.2611, 0.8764, 0.1116, 0.7641, 0.8019, 0.5427, 1.0326, 1.0197, 0.8291,
        1.0335, 0.1031, 1.0666, 0.4686, 0.1219, 0.2516, 0.9970, 0.2400, 1.0105,
        1.0489, 0.8134, 0.4170, 0.8929, 0.9074, 0.9280, 0.5877, 0.5643, 0.4848,
        0.3751, 0.9538, 1.0991, 0.9285, 0.3416, 1.0127, 0.8256, 0.4395, 1.0921,
        0.1505, 0.6220, 0.6386, 0.9484, 0.2019, 0.5340, 0.9133, 0.3834, 0.4148,
        0.6998, 0.6376, 0.9585, 0.6509, 0.7625, 0.3764, 0.7018, 0.9334, 0.7512,
        0.3095, 0.3616, 0.3270, 0.8455, 0.9335, 0.8093, 0.4750, 0.4846, 0.6335,
        0.7165, 0.9139, 0.5954, 0.7076, 0.9008, 0.1978, 0.5625, 0.6825, 1.0596,
        0.5888, 0.5877, 0.4713, 0.2387, 0.7854, 0.4110, 0.9617, 1.0550, 0.3033,
        0.6410, 0.6020, 0.7795, 0.3399, 1.0763, 0.6782, 0.7133, 0.6079, 0.7346,
        0.3484, 0.3126, 0.2716, 0.3078, 0.3837, 0.2800, 0.3970, 0.3124, 0.5520,
        1.0736, 0.8413, 0.4818, 0.2627, 0.6548, 0.3936, 0.3906, 0.2534, 0.3182,
        0.2695, 0.7610, 0.7882, 0.4167, 0.8773, 0.4455, 0.7677, 0.8228, 0.9794,
        0.1183, 0.9494, 0.9326, 0.9463, 1.0442, 0.8734, 1.0183, 0.4835, 0.7916,
        0.6713, 0.6136, 0.7262, 0.9628, 0.7261, 0.9135, 0.9505][k], 
        "w3" : lambda k : [0.1695, 1.0405, 0.9651, 0.6679, 1.0814, 0.8631, 0.3158, 0.9134, 0.2561,
        0.4979, 0.9131, 0.5042, 1.0880, 0.8037, 1.0605, 0.9132, 0.1607, 0.1249,
        0.2612, 0.3513, 0.6389, 0.8843, 0.9219, 0.3455, 0.7670, 0.8783, 0.8885,
        0.7694, 1.0437, 0.4543, 0.9689, 1.0585, 0.8696, 0.7796, 0.2655, 0.7289,
        1.0855, 0.3105, 0.2814, 0.5074, 0.8595, 0.4159, 0.6908, 1.0596, 0.7199,
        0.2212, 0.2505, 0.1598, 0.8464, 0.9000, 0.6167, 0.5220, 0.4580, 0.8436,
        0.4145, 0.4440, 0.1648, 0.6662, 0.5421, 0.8961, 0.6710, 0.3740, 0.7296,
        0.6419, 0.1801, 0.3065, 0.5176, 0.2219, 0.3392, 0.6039, 0.8975, 0.3110,
        0.4563, 0.4839, 0.8874, 0.8979, 0.2477, 0.8021, 1.0760, 0.5908, 0.8217,
        0.2314, 0.1927, 0.3680, 0.3667, 0.8237, 0.4970, 0.1603, 0.6694, 0.2223,
        0.4259, 0.6089, 0.8777, 0.2472, 0.6766, 0.4771, 0.6211, 0.7430, 0.5962,
        0.6298, 0.6999, 0.7671, 0.8682, 0.6237, 0.6345, 0.3902, 0.9768, 0.2890,
        1.0105, 0.5093, 0.2386, 0.8792, 0.1714, 0.8260, 0.2536, 0.1471, 0.1872,
        0.1914, 1.0681, 0.8957, 0.1548, 0.8094, 0.3837, 0.4845, 1.0558, 1.0051,
        0.3616, 0.8315, 0.4581, 0.2528, 0.8384, 0.1804, 0.7733, 0.5331, 0.1370,
        0.1565, 0.9960, 0.8689, 0.7153, 0.3765, 0.8706, 0.2422, 0.4711, 0.1991,
        0.1063, 0.8958, 0.5821, 0.5472, 0.8979, 0.3467, 0.8917][k],
        "w4" : lambda k : 1.0,
        "p1" : lambda k : [384.1328, 250.9762, 300.0][k],
        "p2" : lambda k : [0.2969, 0.9437][k],
        "p3" : lambda k : 0.4
        }

    phi_rule = f"G[0,{horizon}]^w1 (distance >= 2)"
    # phi_confort = f"G[0,{horizon-1}]^w2 &&^p2(u_a <= 10, j <= 30)"
    phi_confort = f"&&^p2((G[0,{horizon-1}]^w2 (u_a <= 10)), (G[0,{horizon-1}]^w3 (j <= 30)))"
    phi_destination = f"F[{horizon-20},{horizon-10}]^w4 G[0,10]^w4 &&^p3(px >= 98, px <= 140)"#(px >= 118, px <= 124)"
    phi = f"&&^p1 ({phi_rule}, {phi_confort}, {phi_destination})"

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 1, 'v': 0, 'theta': -np.pi}
    vars_ub = {'px': 150, 'py': 4.5, 'v': 30, 'theta': np.pi}
    control_lb = {'u_a': -10, 'u_delta': -1, 'j': -80, 'sr': -1}
    control_ub = {'u_a': 15, 'u_delta': 1, 'j': 80, 'sr': 1}

    # Define the initial and final conditions
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

    print("_____F_________", f0)

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
    # alpha = np.array([0.002, 0.002, 0.00001, 0.001])
    # alpha = np.array([2.0, 4.0, 0.01, 10.0])
    beta = np.array([0.00, 0.00])
    zeta =  np.array([0.00, 0.00])
    lambd = 20.0


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