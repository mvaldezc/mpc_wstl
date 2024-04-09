# mpc_wstl

### Installation

1. Install and activate [gurobi](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer).
2. Create a directory that will contain Pytelo and this repo.

```
project
│
├─── PyTeLo
│
└─── mpc_wstl
```

3. Download [PyTeLo](https://github.com/erl-lehigh/PyTeLo).
    - Checkout to wstl branch.
    - Follow instructions, perform antrl4 step with both stlg4 and wstl.g4 .
4. `$ cd mpc_wstl/tests` and run python files.
    - If  `No module named 'stl'` error appears might be due to a missreference of the PyTeLo library to itself, in the PyTeLo python files there are references like `from stl import` , they need to be imported with a '.' as `from .stl import`.