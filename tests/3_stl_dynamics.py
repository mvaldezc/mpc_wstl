from antlr4 import InputStream, CommonTokenStream
import time
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb
import sys
sys.path.append('../../pytelo/')

from wstl.stl import STLAbstractSyntaxTreeExtractor
from wstl.stl2milp import stl2milp
from wstl.stlLexer import stlLexer
from wstl.stlParser import stlParser

def stl_synthesis_control(formula, A, B, vars_ub, vars_lb, control_ub, 
                          control_lb, alpha=0.1, beta=0.1):
    
    lexer = stlLexer(InputStream(formula))
    tokens = CommonTokenStream(lexer)
    parser = stlParser(tokens)
    t = parser.stlProperty()
    ast = STLAbstractSyntaxTreeExtractor().visit(t)
  
    stl_milp = stl2milp(ast, robust=True)
   
    time_horizon = int(ast.bound()) + 1
    x = dict()
    y = dict()
    u = dict()
    v = dict()
    x_aux = dict()
    y_aux =dict()
    u_aux = dict()
    v_aux =dict() 
    
    # Couple predicate variables with constraint variables
    for k in range(time_horizon):
        name = "x_{}".format(k) 
        x[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb, 
                                     ub=vars_ub, name=name)                             
        name = "y_{}".format(k)
        y[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=vars_lb, 
                                     ub=vars_ub, name=name)
        name = "u_{}".format(k)
        u[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb, 
                                     ub=control_ub, name=name)
        name = "v_{}".format(k)
        v[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=control_lb, 
                                     ub=control_ub, name=name)
        # additional variables to compute cost
        name = "x_aux_{}".format(k) 
        x_aux[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub, name=name)
        name = "y_aux_{}".format(k)                                      
        y_aux[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub, name=name)
        name = "u_aux_{}".format(k) 
        u_aux[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub, name=name)
        name = "v_aux_{}".format(k)                                      
        v_aux[k] = stl_milp.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, 
                                     ub=vars_ub, name=name)

    # Indicate which variables are related
    stl_milp.variables['x'] = x
    stl_milp.variables['y'] = y
    stl_milp.variables['u'] = u
    stl_milp.variables['v'] = v

    # system constraints x[k+1] = A X[k]+ B U[k]
    for k in range(time_horizon-1):
        stl_milp.model.addConstr(x[k+1] == A[0][0] * x[k] + A[0][1] * y[k] +
                                           B[0][0] * u[k])

        stl_milp.model.addConstr(y[k+1] == A[1][0] * x[k] + A[1][1] * y[k] +
                                           B[0][1] * v[k])
    
    # Initial conditions as additional constraints
    stl_milp.model.addConstr(x[0] == 0)
    stl_milp.model.addConstr(y[0] == 0)
    stl_milp.model.addConstr(u[0] == 0)
    stl_milp.model.addConstr(v[0] == 0)

    # add the specification (STL) constraints
    stl_milp.translate(satisfaction=True)
    
    # add objective function
    stl_milp.model.addConstrs(x_aux[k] == grb.abs_(x[k]) for k in range(time_horizon))
    stl_milp.model.addConstrs(y_aux[k] == grb.abs_(y[k]) for k in range(time_horizon))
    state_cost = sum(x_aux[k] + y_aux[k] for k in range(time_horizon))

    stl_milp.model.addConstrs(u_aux[k] == grb.abs_(u[k]) for k in range(time_horizon))
    stl_milp.model.addConstrs(v_aux[k] == grb.abs_(v[k]) for k in range(time_horizon))
    control_cost = sum(u_aux[k] + v_aux[k] for k in range(time_horizon))

    stl_milp.model.setObjectiveN(state_cost, 2, weight=alpha, name='state_cost')
    stl_milp.model.setObjectiveN(control_cost, 3, weight=beta, name='control_cost')
    # Solve the problem with gurobi 
    stl_milp.model.optimize()
    return stl_milp


def visualize(stl_milp, stl_milp2):
    t = stl_milp.variables['x'].keys()
    t2 = stl_milp2.variables['x'].keys()

    stl_x = [var.x for var in stl_milp.variables['x'].values()]
    stl_y = [var.x for var in stl_milp.variables['y'].values()]
    stl_u = [var.x for var in stl_milp.variables['u'].values()]
    stl_v = [var.x for var in stl_milp.variables['v'].values()]
    stl_x2 = [var.x for var in stl_milp2.variables['x'].values()]
    stl_y2 = [var.x for var in stl_milp2.variables['y'].values()]
    stl_u2 = [var.x for var in stl_milp2.variables['u'].values()]
    stl_v2 = [var.x for var in stl_milp2.variables['v'].values()]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('STL-Control Synthesis')

    axs[0][0].plot(t, stl_x, '-r', label=r'$\lambda=1, \alpha=0, \beta=0$', 
                   linewidth=3, marker='s', markersize=7)
    axs[0][0].plot(t, stl_x2, '-b', label=r'$\lambda=1, \alpha=0.1, \beta=0.1$',
                   linewidth=3, marker='s', markersize=7)                
    axs[0][0].set_title('x vs t')
    axs[0][0].grid()
    axs[0][0].legend(prop={'size': 10})
    axs[0][0].xaxis.set_tick_params(labelsize=12)
    axs[0][0].tick_params(labelsize=10)

    axs[1][0].plot(t, stl_y, '-r', label=r'$\lambda=1, \alpha=0, \beta=0$', 
                   linewidth=3, marker='s', markersize=7)
    axs[1][0].plot(t, stl_y2, '-b', label=r'$\lambda=1, \alpha=0.1, \beta=0.1$',
                   linewidth=3, marker='s', markersize=7)
    axs[1][0].set_title('y vs t')
    axs[1][0].grid()
    axs[1][0].legend(prop={'size': 10})
    axs[1][0].tick_params(labelsize=10)

    axs[0][1].plot(t, stl_u, '-r', label=r'$\lambda=1, \alpha=0, \beta=0$', 
                   linewidth=3, marker='s', markersize=7)
    axs[0][1].plot(t, stl_u2, '-b', label=r'$\lambda=1, \alpha=0.1, \beta=0.1$',
                   linewidth=3, marker='s', markersize=7)
    axs[0][1].set_title('u vs t')
    axs[0][1].grid()
    axs[0][1].legend(prop={'size': 10})
    axs[0][1].tick_params(labelsize=10)


    axs[1][1].plot(t, stl_v, '-r', label=r'$\lambda=1, \alpha=0, \beta=0$', 
                   linewidth=3, marker='s', markersize=7)
    axs[1][1].plot(t, stl_v2, '-b', label=r'$\lambda=1, \alpha=0.1, \beta=0.1$',
                   linewidth=3, marker='s', markersize=7)
    axs[1][1].set_title('v vs t')
    axs[1][1].grid()
    axs[1][1].legend(prop={'size': 10})
    axs[1][1].tick_params(labelsize=10)
    fig.tight_layout()
    plt.show()
    


if __name__ == '__main__':  
    formula = '(G[3,5] (x >= 3)) && (G[9,10] (y >= 2))'    

    # Define the matrices for linear system 
    A = [[1, 1], [0, 1]] 
    B = [[1,  1]] 
    vars_ub = 9
    vars_lb = -9
    control_ub = 5
    control_lb = -5

    # Translate WSTL to MILP and retrieve integer variable for the formula
    stl_start = time.time()
    stl_milp = stl_synthesis_control(formula, A, B, vars_ub, vars_lb, control_ub,
                                    control_lb, alpha=0, beta=0)
    stl_milp2 = stl_synthesis_control(formula, A, B, vars_ub, vars_lb, control_ub,
                                    control_lb, alpha=0.1, beta=0.1)
                                   
    stl_end = time.time()
    stl_time = stl_end - stl_start

    print(formula, 'Time needed:', stl_time)
    
    visualize(stl_milp, stl_milp2)
 
    
    

