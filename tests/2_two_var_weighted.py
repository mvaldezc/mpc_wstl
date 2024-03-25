#!/usr/bin/env python3
from antlr4 import InputStream, CommonTokenStream

import sys
sys.path.append('../../pytelo/')

from wstl.wstl import WSTLFormula, WSTLAbstractSyntaxTreeExtractor
from wstl.wstlLexer import wstlLexer
from wstl.wstlParser import wstlParser

# 1 Choose the correct script
from wstl.wstl2milp import wstl2milp 

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as grb

# 2 Define the STL formula
# Always between 9 am to 10 am bring tea or preferably coffee to the office desk. 
# Always between 12 to 1 pm, be as far from the office room as possible and visit 
# the living room between times 2 pm to 3 pm, preferably towards the end of the period.”

# 9,10,11,12,1,2,3 its 
# 0, 1, 2, 3,4,5,6
# x = 0/1, z = {office : 0, other_areas : 1, living_room : 2}

weights = {}
weights["p1"] = lambda k : 1
weights["p2"] = lambda k : [1.0, 0.5][k]
weights["p3"] = lambda k : 1
weights["w1"] = lambda k : 1
weights["w2"] = lambda k : 1
weights["w3"] = lambda k : [1.0, 0.1][k-5]

subformula1 = 'G[0,1]^w1 ((x >= 1) ||^p2 (x <= 0))'
subformula2 = 'G[3,4]^w2 ((z > 0))'
subformula3 = 'F[5,6]^w3 ((z >= 2) &&^p3 (z <= 2))'
formula = '&&^p1 (' + subformula1 + ', ' + subformula2 + ', ' + subformula3 + ')'
# Robustness will be zero if the formula is satisfied, 
# but weights won't make a difference in this case

# 3 Create AST
lexer = wstlLexer(InputStream(formula))
tokens = CommonTokenStream(lexer)
parser = wstlParser(tokens)
t = parser.wstlProperty()
ast = WSTLAbstractSyntaxTreeExtractor(weights).visit(t)
print(ast)
print('AST:', str(ast))

# 4 Create Gurobi object, ast is input, robust is as true to maximize robustness
x_range = [0, 1]
z_range = [0, 2]
wstl_milp = wstl2milp(ast, ranges={'x': x_range, 'z': z_range})

# 5 Translate the AST to MILP
#wstl_milp.translate(satisfaction=True)
z_formula, rho_formula = wstl_milp.translate()
wstl_milp.model.update()
wstl_milp.model.setObjective(rho_formula, grb.GRB.MAXIMIZE)
wstl_milp.model.update()

# 6 Optimize the MILP
wstl_milp.model.optimize()

# Compute time horizon and robustness
horizon = int(WSTLFormula.bound(ast))
robustness = wstl_milp.model.getObjective().getValue()

print('Objective')
obj = wstl_milp.model.getObjective()
print(str(obj), ':', obj.getValue())

# Extract vars to plot
x_traj = np.zeros(horizon + 1)
z_traj = np.zeros(horizon + 1)
for i in range(horizon + 1):
    if wstl_milp.model.getVarByName('x_' + str(i)) is not None:
        x_traj[i] = wstl_milp.model.getVarByName('x_' + str(i)).x
    if wstl_milp.model.getVarByName('z_' + str(i)) is not None:
        z_traj[i] = wstl_milp.model.getVarByName('z_' + str(i)).x

print("\n=================================================")
print("Specification: ", formula)
print("Time horizon: ", horizon)
print('rho_opt: ', robustness)
print("x_opt: ", x_traj)
print("x_opt: ", z_traj)

# 7 Plot the trajectory, enable grid, plot in red, square markers, in different subplots
fig, axs = plt.subplots(2)
fig.suptitle('STL robustness optimization')
axs[0].plot(x_traj, 'r-s')
axs[0].grid(True)
axs[0].set_title('Drink')
axs[0].set_yticks([0, 1])
y_labels_0 = [item.get_text() for item in axs[0].get_yticklabels()]
y_labels_0[0] = 'Tea'
y_labels_0[1] = 'Coffee'
axs[0].set_xticks([0, 1, 2, 3, 4, 5, 6])
time_labels = [item.get_text() for item in axs[0].get_xticklabels()]
time_labels[0] = '9'
time_labels[1] = '10'
time_labels[2] = '11'
time_labels[3] = '12'
time_labels[4] = '1'
time_labels[5] = '2'
time_labels[6] = '3'
axs[0].set_yticklabels(y_labels_0)
axs[0].set_xticklabels(time_labels)
axs[0].axhline(y=x_range[0], color='b', linestyle='-')
axs[0].axhline(y=x_range[1], color='b', linestyle='-')

axs[1].plot(z_traj, 'r-s')
axs[1].grid(True)
axs[1].set_title('Location')
axs[1].set_yticks([0, 1, 2])
y_labels_1 = [item.get_text() for item in axs[1].get_yticklabels()]
y_labels_1[0] = 'Office'
y_labels_1[1] = 'Other areas'
y_labels_1[2] = 'Living room'
axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
axs[1].set_xticklabels(time_labels)
axs[1].set_yticklabels(y_labels_1)
axs[1].axhline(y=z_range[0], color='b', linestyle='-')
axs[1].axhline(y=z_range[1], color='b', linestyle='-')
fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()

