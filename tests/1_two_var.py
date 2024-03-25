#!/usr/bin/env python3
from antlr4 import InputStream, CommonTokenStream

import sys
sys.path.append('../../pytelo/')

from stl.stl import STLFormula, STLAbstractSyntaxTreeExtractor
from stl.stlLexer import stlLexer
from stl.stlParser import stlParser

# 1 Choose the correct script
from stl.stl2milp import stl2milp 

import matplotlib.pyplot as plt
import numpy as np

# 2 Define the STL formula
formula = 'G[0,10] x >= 3 && F[0,10] y < 3]'

# 3 Create AST
lexer = stlLexer(InputStream(formula))
tokens = CommonTokenStream(lexer)
parser = stlParser(tokens)
t = parser.stlProperty()
ast = STLAbstractSyntaxTreeExtractor().visit(t)
print('AST:', str(ast))

# 4 Create Gurobi object, ast is input, robust is as true to maximize robustness
x_range = [-4, 4]
y_range = [-4, 4]
stl_milp = stl2milp(ast, ranges={'x': x_range, 'y': y_range}, robust=True)

# 5 Translate the AST to MILP
stl_milp.translate(satisfaction=True)

# 6 Optimize the MILP
stl_milp.model.optimize()

# Compute time horizon and robustness
horizon = int(STLFormula.bound(ast))
robustness = -stl_milp.model.getObjective().getValue()

# Extract vars to plot
x_traj = np.zeros(horizon + 1)
y_traj = np.zeros(horizon + 1)
for i in range(horizon + 1):
    x_traj[i] = stl_milp.model.getVarByName('x_' + str(i)).x
    y_traj[i] = stl_milp.model.getVarByName('y_' + str(i)).x

print("\n=================================================")
print("Specification: ", formula)
print("Time horizon: ", horizon)
print('rho_opt: ', robustness)
print("x_opt: ", x_traj)
print("y_opt: ", y_traj)

# 7 Plot the trajectory, enable grid, plot in red, square markers, in different subplots
fig, axs = plt.subplots(2)
fig.suptitle('STL robustness optimization')
axs[0].plot(x_traj, 'r-s')
axs[0].grid(True)
axs[0].set_title('x')
axs[1].plot(y_traj, 'r-s')
axs[1].grid(True)
axs[1].set_title('y')
axs[0].axhline(y=x_range[0], color='b', linestyle='-')
axs[0].axhline(y=x_range[1], color='b', linestyle='-')
axs[1].axhline(y=y_range[0], color='b', linestyle='-')
axs[1].axhline(y=y_range[1], color='b', linestyle='-')
fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()

