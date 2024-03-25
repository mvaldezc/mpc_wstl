from antlr4 import InputStream, CommonTokenStream
import time
import matplotlib.pyplot as plt
import gurobipy as grb
import sys
sys.path.append('../../pytelo/')

from wstl.wstl import WSTLAbstractSyntaxTreeExtractor
from wstl.wstl2milp import wstl2milp
from wstl.wstlLexer import wstlLexer
from wstl.wstlParser import wstlParser

def wstl_synthesis_control(
                formula : str,
                weights : dict, 
                A : list,
                B : list,
                T : float,
                vars_lb : dict,
                vars_ub : dict,
                control_lb : dict, 
                control_ub : dict,
                x_0 : dict,
                alpha: float, 
                beta :float,
                zeta : float):
    """
    Synthesize a controller for a given wSTL formula and a linear system

    Inputs
    ----------
    formula :  wSTL formula
    weights :  weights for the formula
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
    beta : weight for the control cost
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

    # Define variables for the cost
    px_abs = dict()
    py_abs =dict()
    vx_abs = dict()
    vy_abs =dict()
    u_ax_abs = dict()
    u_ay_abs = dict()
    jx_abs = dict()
    jy_abs = dict()
    
    # Couple predicate variables with constraint variables
    for k in range(time_horizon):
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
        
        # additional variables to compute cost
        name = "px_abs_{}".format(k) 
        px_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub['px'], name=name)
        name = "py_abs_{}".format(k)                                      
        py_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub['py'], name=name)
        name = "vx_abs_{}".format(k) 
        vx_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub['vx'], name=name)
        name = "vy_abs_{}".format(k)                                      
        vy_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub['vy'], name=name)
        
        if k < time_horizon - 1:

            name = "u_ax_abs_{}".format(k) 
            u_ax_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                        ub=control_ub['u_ax'], name=name)
            name = "u_ay_abs_{}".format(k)                                      
            u_ay_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                        ub=control_ub['u_ay'], name=name)
            
            name = "jx_abs_{}".format(k)
            jx_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0,
                                            ub=control_ub['jx'], name=name)
            name = "jy_abs_{}".format(k)
            jy_abs[k] = wstl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0,
                                            ub=control_ub['jy'], name=name)
            

    # Indicate which variables are related
    wstl_milp.variables['px'] = px
    wstl_milp.variables['py'] = py
    wstl_milp.variables['vx'] = vx
    wstl_milp.variables['vy'] = vy
    wstl_milp.variables['u_ax'] = u_ax
    wstl_milp.variables['u_ay'] = u_ay
    wstl_milp.variables['jx'] = jx
    wstl_milp.variables['jy'] = jy

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

    # add the specification (STL) constraints and objective function
    z_formula, rho_formula = wstl_milp.translate(satisfaction=True)
    wstl_milp.model.addConstrs(px_abs[k] == grb.abs_(px[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(py_abs[k] == grb.abs_(py[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(vx_abs[k] == grb.abs_(vx[k]) for k in range(time_horizon))
    wstl_milp.model.addConstrs(vy_abs[k] == grb.abs_(vy[k]) for k in range(time_horizon))
    state_cost = sum(px_abs[k] + py_abs[k] + vx_abs[k] + vy_abs[k] for k in range(time_horizon))

    wstl_milp.model.addConstrs(u_ax_abs[k] == grb.abs_(u_ax[k]) for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(u_ay_abs[k] == grb.abs_(u_ay[k]) for k in range(time_horizon-1))
    control_cost = sum(u_ax_abs[k] + u_ay_abs[k] for k in range(time_horizon-1))

    wstl_milp.model.addConstrs(jx_abs[k] == grb.abs_(jx[k]) for k in range(time_horizon-1))
    wstl_milp.model.addConstrs(jy_abs[k] == grb.abs_(jy[k]) for k in range(time_horizon-1))
    jerk_cost = sum(jx_abs[k] + jy_abs[k] for k in range(time_horizon-1))

    wstl_milp.model.setObjective(rho_formula - alpha*state_cost - beta*control_cost - zeta*jerk_cost, grb.GRB.MAXIMIZE)

    # Solve the problem with gurobi 
    wstl_milp.model.optimize()
    return wstl_milp


def visualize(stl_milp, sampling_time):
    t = [k * sampling_time for k in stl_milp.variables['px'].keys()]
    
    stl_px = [var.x for var in stl_milp.variables['px'].values()]
    stl_py = [var.x for var in stl_milp.variables['py'].values()]
    stl_vx = [var.x for var in stl_milp.variables['vx'].values()]
    stl_vy = [var.x for var in stl_milp.variables['vy'].values()]
    stl_u_ax = [var.x for var in stl_milp.variables['u_ax'].values()]
    stl_u_ay = [var.x for var in stl_milp.variables['u_ay'].values()]
    stl_jx = [var.x for var in stl_milp.variables['jx'].values()]
    stl_jy = [var.x for var in stl_milp.variables['jy'].values()]
    
    fig, axs = plt.subplots(4, 2, figsize=(10, 7))
    fig.suptitle('STL-Control Synthesis')

    axs[0][0].plot(t, stl_px, '-r', label=r'x', 
                   linewidth=3, marker='s', markersize=7)             
    axs[0][0].set_title('x vs t')
    axs[0][0].grid()
    axs[0][0].legend(prop={'size': 10})
    axs[0][0].xaxis.set_tick_params(labelsize=12)
    axs[0][0].tick_params(labelsize=10)

    axs[0][1].plot(t, stl_py, '-r', label=r'y', 
                   linewidth=3, marker='s', markersize=7)
    axs[0][1].set_title('y vs t')
    axs[0][1].grid()
    axs[0][1].legend(prop={'size': 10})
    axs[0][1].tick_params(labelsize=10)

    axs[1][0].plot(t, stl_vx, '-r', label=r'vx', 
                   linewidth=3, marker='s', markersize=7)
    axs[1][0].set_title('v_x vs t')
    axs[1][0].grid()
    axs[1][0].legend(prop={'size': 10})
    axs[1][0].tick_params(labelsize=10)

    axs[1][1].plot(t, stl_vy, '-r', label=r'vy', 
                   linewidth=3, marker='s', markersize=7)
    axs[1][1].set_title('v_y vs t')
    axs[1][1].grid()
    axs[1][1].legend(prop={'size': 10})
    axs[1][1].tick_params(labelsize=10)

    axs[2][0].plot(t[0:-1] , stl_u_ax, '-r', label=r'ax', 
                   linewidth=3, marker='s', markersize=7)
    axs[2][0].set_title('a_x vs t')
    axs[2][0].grid()
    axs[2][0].legend(prop={'size': 10})
    axs[2][0].tick_params(labelsize=10)

    axs[2][1].plot(t[0:-1], stl_u_ay, '-r', label=r'ay', 
                   linewidth=3, marker='s', markersize=7)
    axs[2][1].set_title('a_y vs t')
    axs[2][1].grid()
    axs[2][1].legend(prop={'size': 10})
    axs[2][1].tick_params(labelsize=10)

    axs[3][0].plot(t[0:-1], stl_jx, '-r', label=r'jx', 
                   linewidth=3, marker='s', markersize=7)
    axs[3][0].set_title('j_x vs t')
    axs[3][0].grid()
    axs[3][0].legend(prop={'size': 10})
    axs[3][0].tick_params(labelsize=10)

    axs[3][1].plot(t[0:-1], stl_jy, '-r', label=r'jy',
                     linewidth=3, marker='s', markersize=7)
    axs[3][1].set_title('j_y vs t')
    axs[3][1].grid()
    axs[3][1].legend(prop={'size': 10})
    axs[3][1].tick_params(labelsize=10)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':  
    # Define wSTL specification
    weights = {"w1" : lambda k : 1.0, 
               "p1" : lambda k : 1.0
               }
    # Check positions stay within a box
    # formula = 'G[5,20]^w1 &&^p1(&&^p1(px <= 5.0, px >= -5.0), &&^p1(py <= 5.0, py >= -5.0))'  

    # Check positions and velocities stay within a box 
    # formula = 'G[3,20]^w1 &&^p1(&&^p1(px <= 5.0, px >= -5.0), &&^p1(py <= 5.0, py >= -5.0), \
    # &&^p1(vx <= 3.0, vx >= -3.0), &&^p1(vy <= 3.0, vy >= -3.0))'    

    # Formula to achieve a setpoint
    formula = 'G[20,21]^w1 &&^p1(&&^p1(px <= 5.0, px >= 5.0), &&^p1(py <= 0.0, py >= 0.0)), \
    &&^p1(vx <= 0.0, vx >= 0.0), &&^p1(vy <= 0.0, vy >= 0.0))'  

    # Define the matrices for linear system 
    T = 0.1 # sampling time
    A = [[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]]
    B = [[T**2/2, 0], [0, T**2/2], [T, 0], [0, T]]

    # Define the bounds for the state and control inputs
    vars_lb = {'px': -20, 'py': -20, 'vx': -40, 'vy': -40}
    vars_ub = {'px': 20, 'py': 20, 'vx': 40, 'vy': 40}
    control_lb = {'u_ax': -50, 'u_ay': -50, 'jx': -200, 'jy': -200}
    control_ub = {'u_ax': 50, 'u_ay': 50, 'jx': 200, 'jy': 200}

    # Define the initial condition
    x_0 = {'px': 0, 'py': 0, 'vx': 0, 'vy': 0}

    # Translate WSTL to MILP and retrieve integer variable for the formula
    stl_start = time.time()
    stl_milp = wstl_synthesis_control(formula, weights, A, B, T, vars_lb, vars_ub, control_lb,
                                    control_ub, x_0, alpha=0, beta=0, zeta=0.001*T)                            
    stl_end = time.time()
    stl_time = stl_end - stl_start

    print(formula, 'Time needed:', stl_time)
    
    visualize(stl_milp, T)
