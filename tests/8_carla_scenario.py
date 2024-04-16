from antlr4 import InputStream, CommonTokenStream
import time
from visualize import visualize2d, visualize_animation, plot_var, save_vid
import gurobipy as grb
import sys
import numpy as np
from typing import Callable
sys.path.append('../../pytelo/')

from wstl.wstl import WSTLAbstractSyntaxTreeExtractor
from wstl.wstl2milp import wstl2milp
from wstl.wstlLexer import wstlLexer
from wstl.wstlParser import wstlParser

def wstl_synthesis_control(
                formula : str,
                weights : dict, 
                pedestrian : Callable[[float], list],
                A : list,
                B : list,
                T : float,
                vars_lb : dict,
                vars_ub : dict,
                control_lb : dict, 
                control_ub : dict,
                x_0 : dict,
                sp : dict,
                alpha: float, 
                betax :float,
                betay :float,
                zeta : float):
    """
    Synthesize a controller for a given wSTL formula and a linear system

    Inputs
    ----------
    formula :  wSTL formula
    weights :  weights for the formula
    pedestrian : pedestrian model, receivestime and returns a list with the pedestrian positions
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
    betax : weight for the x-component of the control cost
    betay : weight for the y-component of the control cost
    zeta : weight for the jerk cost

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
    vx = dict()
    vy = dict()
    u_ax = dict()
    u_ay = dict()
    jx = dict()
    jy = dict()

    # Define variables for pedestrian
    x_ped = dict()
    y_ped = dict()
    x_dist = dict()
    y_dist = dict()
    distance = dict()  # define 

    # Define variables for the cost
    delta_px = dict() # define state-setpoint
    delta_py = dict()
    delta_vx = dict()
    delta_vy = dict()
    s_abs = dict() # define abs(state-setpoint)
    u_abs = dict() # define magnitude of control
    j_abs = dict() # define magnitude of jerk
    u_ax_abs = dict()
    u_ay_abs = dict()

    # Couple predicate variables with constraint variables
    for k in range(time_horizon):  

        # State variables
        name = "px_{}".format(k) 
        px[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['px'], 
                                       ub=vars_ub['px'], name=name)                             
        name = "py_{}".format(k)
        py[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['py'], 
                                       ub=vars_ub['py'], name=name)
        name = "vx_{}".format(k) 
        vx[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['vx'], 
                                       ub=vars_ub['vx'], name=name)                             
        name = "vy_{}".format(k)
        vy[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb['vy'], 
                                       ub=vars_ub['vy'], name=name)
        
        # Control inputs
        if k < time_horizon - 1:    
            name = "u_ax_{}".format(k)
            u_ax[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['u_ax'], 
                                            ub=control_ub['u_ax'], name=name)
            name = "u_ay_{}".format(k)
            u_ay[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['u_ay'], 
                                            ub=control_ub['u_ay'], name=name)
            name = "jx_{}".format(k)
            jx[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['jx'],
                                            ub=control_ub['jx'], name=name)
            name = "jy_{}".format(k)
            jy[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb['jy'],
                                            ub=control_ub['jy'], name=name)
            name = "u_ax_abs_{}".format(k)
            u_ax_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            name = "u_ay_abs_{}".format(k)
            u_ay_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            name = "u_abs_{}".format(k)
            u_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
            name = "j_abs_{}".format(k)
            j_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)

        # Error variables
        name = "delta_px_{}".format(k)
        delta_px[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-130, ub=130, name=name)
        name = "delta_py_{}".format(k)
        delta_py[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-20, ub=20, name=name)
        name = "delta_vx_{}".format(k)
        delta_vx[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-130, ub=130, name=name)
        name = "delta_vy_{}".format(k)
        delta_vy[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-20, ub=20, name=name)
        name = "s_abs_{}".format(k)
        s_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, name=name)
        
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
    wstl_milp.variables['vx'] = vx
    wstl_milp.variables['vy'] = vy
    wstl_milp.variables['u_ax'] = u_ax
    wstl_milp.variables['u_ay'] = u_ay
    wstl_milp.variables['jx'] = jx
    wstl_milp.variables['jy'] = jy
    wstl_milp.variables['x_ped'] = x_ped
    wstl_milp.variables['y_ped'] = y_ped
    wstl_milp.variables['x_dist'] = x_dist
    wstl_milp.variables['y_dist'] = y_dist
    wstl_milp.variables['distance'] = distance
    wstl_milp.variables['delta_px'] = delta_px
    wstl_milp.variables['delta_py'] = delta_py
    wstl_milp.variables['delta_vx'] = delta_vx
    wstl_milp.variables['delta_vy'] = delta_vy
    wstl_milp.variables['s_abs'] = s_abs
    wstl_milp.variables['u_abs'] = u_abs
    wstl_milp.variables['j_abs'] = j_abs
    wstl_milp.variables['u_ax_abs'] = u_ax_abs
    wstl_milp.variables['u_ay_abs'] = u_ay_abs

    # system constraints x[k+1] = A X[k]+ B U[k]
    for k in range(time_horizon-1):
        wstl_milp.model.addConstr(px[k+1] == A[0][0] * px[k]   +  A[0][1] * py[k]  +
                                             A[0][2] * vx[k]   +  A[0][3] * vy[k]  +
                                             B[0][0] * u_ax[k] +  B[0][1] * u_ay[k])

        wstl_milp.model.addConstr(py[k+1] == A[1][0] * px[k]   +  A[1][1] * py[k]  +
                                             A[1][2] * vx[k]   +  A[1][3] * vy[k]  +
                                             B[1][0] * u_ax[k] +  B[1][1] * u_ay[k])
        
        wstl_milp.model.addConstr(vx[k+1] == A[2][0] * px[k]   +  A[2][1] * py[k]  +
                                             A[2][2] * vx[k]   +  A[2][3] * vy[k]  +
                                             B[2][0] * u_ax[k] +  B[2][1] * u_ay[k])

        wstl_milp.model.addConstr(vy[k+1] == A[3][0] * px[k]   +  A[3][1] * py[k]  +
                                             A[3][2] * vx[k]   +  A[3][3] * vy[k]  +
                                             B[3][0] * u_ax[k] +  B[3][1] * u_ay[k])
    
    # Jerk constraints
    wstl_milp.model.addConstr(jx[0] == (u_ax[0] - 0)/T)
    wstl_milp.model.addConstr(jy[0] == (u_ay[0] - 0)/T)
    wstl_milp.model.addConstrs(jx[k] == (u_ax[k] - u_ax[k-1])/T for k in range(1, time_horizon-1))
    wstl_milp.model.addConstrs(jy[k] == (u_ay[k] - u_ay[k-1])/T for k in range(1, time_horizon-1))

    # Initial conditions as additional constraints
    wstl_milp.model.addConstr(px[0] == x_0['px'])
    wstl_milp.model.addConstr(py[0] == x_0['py'])
    wstl_milp.model.addConstr(vx[0] == x_0['vx'])
    wstl_milp.model.addConstr(vy[0] == x_0['vy'])

    # Setpoint constraints
    wstl_milp.model.addConstr(px[time_horizon-1] == sp['px'])
    wstl_milp.model.addConstr(py[time_horizon-1] == sp['py'])
    wstl_milp.model.addConstr(vx[time_horizon-1] == sp['vx'])
    wstl_milp.model.addConstr(vy[time_horizon-1] == sp['vy'])

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
    wstl_milp.model.addConstrs(delta_vx[k] == sp['vx'] - vx[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(delta_vy[k] == sp['vy'] - vy[k] for k in range(time_horizon))
    wstl_milp.model.addConstrs(s_abs[k] == grb.norm(
                                                [delta_px[k],
                                                delta_py[k],
                                                delta_vx[k],
                                                delta_vy[k]],1) 
                                                for k in range(time_horizon))
    state_cost = sum(s_abs[k] for k in range(time_horizon))

    # Control magnitude cost
    wstl_milp.model.addConstrs(u_ax_abs[k] == grb.abs_(u_ax[k]) for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(u_ay_abs[k] == grb.abs_(u_ay[k]) for k in range(time_horizon-1))
    control_cost = sum(betax*u_ax_abs[k] + betay*u_ay_abs[k] for k in range(time_horizon-1))
    # wstl_milp.model.addConstrs(u_abs[k] == grb.norm([u_ax[k], u_ay[k]], 1) for k in range(time_horizon-1))
    # control_cost = sum(u_abs[k] for k in range(time_horizon-1))

    # Jerk magnitude cost
    wstl_milp.model.addConstrs(j_abs[k] == grb.norm([jx[k], jy[k]], 1) for k in range(time_horizon-1))
    jerk_cost = sum(j_abs[k] for k in range(time_horizon-1))

    wstl_milp.model.addConstr(rho_formula >= 0)
    
    wstl_milp.model.setObjective(rho_formula - alpha*state_cost - control_cost - zeta*jerk_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()
    return wstl_milp, rho_formula, z_formula



if __name__ == '__main__':  

    T = 0.2 # sampling time
    time_length = 18 #+ 2*np.random.rand()
    horizon = int(time_length/T)+1

    # w1 = np.random.rand(horizon+1)
    # w2 = np.random.rand(horizon+1)
    # p1 = np.random.rand(2)
    # p2 = np.random.rand(2)
    # weight_list = [w1, w2, p1, p2]
    # print(w1)
    # print(w2)
    # print(p1)
    # print(p2)
    # Define wSTL specification
    weights = {"w1" : lambda k : 1.0, 
               "w2" : lambda k : 1.0, 
               "p1" : lambda k : [1.0, 1.0][k],
               "p2" : lambda k : [1.0, 1.0][k],
               }
    # weights = {"w1" : lambda k : w1[k], 
    #            "w2" : lambda k : w2[k], 
    #            "p1" : lambda k : p1[k],
    #            "p2" : lambda k : p2[k],
    #            }

    phi_rule = f"G[0,{horizon}]^w1 (distance >= 2)"
    phi_confort = f"G[0,{horizon}]^w2 &&^p2(u_ax <= 10, jx <= 30)"
    phi = f"&&^p1 ({phi_rule}, {phi_confort})"

    # Define the matrices for linear system 
    A = [[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]]
    B = [[T**2/2, 0], [0, T**2/2], [T, 0], [0, T]]

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -5, 'py': 0, 'vx': -40, 'vy': -20}
    vars_ub = {'px': 125, 'py': 5, 'vx': 40, 'vy': 20}
    control_lb = {'u_ax': -2, 'u_ay': -5, 'jx': -6, 'jy': -15}
    control_ub = {'u_ax': 15, 'u_ay': 5, 'jx': 45, 'jy': 15}

    # Define the initial and final conditions
    x_0 = {'px': 0, 'py': 2.5, 'vx': 0, 'vy': 0}
    x_f = {'px': 120, 'py': 2.5, 'vx': 0, 'vy': 0}

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

    # Translate WSTL to MILP and retrieve integer variable for the formula
    stl_start = time.time()
    stl_milp, rho_formula, z = wstl_synthesis_control(phi, weights, ped, A, B, T, vars_lb, vars_ub, control_lb, 
                                    control_ub, x_0, x_f, alpha=0.00001, betax=0.0001, betay=1.0, zeta=0.0001)
                                    # control_ub, x_0, x_f, alpha=0, betax=0, betay=1.0, zeta=0)
    stl_end = time.time()
    stl_time = stl_end - stl_start

    print(f"Robustness: {rho_formula.x}")
    print('Time needed:', stl_time)
    print("Time horizon:", horizon)
    print("z:", z.x)

    ani = visualize_animation(stl_milp, T, carla=True)
    #save_vid(ani, "anim/fixed_d48_wrandom.gif")

