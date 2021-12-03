# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from rflearn.algo.episode import Episode
from rflearn.algo.base import State
from rflearn.env import GridWorld
from rflearn.algo import PolicyIteration, ValueIteration, TabularPolicy, TabularQValue

# %%
grid = GridWorld(4, 4)
qvalue = TabularQValue(grid.S, grid.A)
policy = TabularPolicy(grid.S, grid.A)
pi_model = PolicyIteration(grid, qvalue, policy)
pi_model.fit(gamma=1, theta=0.001)
pi_model.evaluate_policy()
print(qvalue.get_all_values(policy))

pi_model.improve_policy()
print(qvalue.get_all_values(policy))


# %%
qvalue.qvalue


# %%
policy.to_numpy()