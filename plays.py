# %%
%load_ext autoreload
%autoreload 2

import numpy as np
from rflearn.env import GridWorld
from rflearn.algo import MCIteration, TabularPolicy, TabularQValue

# %%
grid = GridWorld(4, 4)
qvalue = TabularQValue(grid.S, grid.A)
policy = TabularPolicy(grid.S, grid.A, epsilon=0.05)

mc = MCIteration(grid, qvalue, policy)
mc.fit(gamma=1)
mc.__dict__