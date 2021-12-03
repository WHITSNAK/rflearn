# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from rflearn.algo.episode import Episode
from rflearn.algo.base import State
from rflearn.algo.policy import TabularPolicy

    
s1 = State(np.array([1,2,3]*10000))
s2 = State(np.array([1,2,3]))
s3 = State(np.array([1,2,3]))
pol = TabularPolicy(
    [s1,s2,s3], ['a','b','c'],
    {s1:[0,0,1], s2:[0,1,0], s3:[0.5,0,0.5]}
)

print(pol.to_numpy())
# %%
import numpy as np
from rflearn.algo import PolicyIteration, ValueIteration
from rflearn.env import GridWorld
from rflearn.algo import TabularPolicy

grid = GridWorld(4, 4)
value = np.zeros(shape=len(grid.S))
policy = TabularPolicy(grid.S, grid.A)