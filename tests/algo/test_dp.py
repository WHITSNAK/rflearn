import pytest
import numpy as np
from rflearn.algo import PolicyIteration, ValueIteration
from rflearn.env import GridWorld
from rflearn.algo import TabularPolicy, TabularQValue


@pytest.fixture
def grid44():
    grid = GridWorld(4, 4)
    qvalue = TabularQValue(grid.S, grid.A)
    # value = np.zeros(shape=len(grid.S))
    policy = TabularPolicy(grid.S, grid.A)
    # policy = np.ones(shape=(len(grid.S), len(grid.A))) / len(grid.A)
    return grid, qvalue, policy


@pytest.fixture
def pi_on_44grid(grid44):
    grid, qvalue, policy = grid44

    pi_model = PolicyIteration(grid, qvalue, policy)
    pi_model.fit(gamma=1, theta=0.001)
    return pi_model

@pytest.fixture
def vi_on_44grid(grid44):
    grid, qvalue, policy = grid44

    pi_model = ValueIteration(grid, qvalue, policy)
    pi_model.fit(gamma=1, theta=0.001)
    return pi_model


def test_policy_iteration_grid_44_1step(pi_on_44grid):
    pi_model = pi_on_44grid

    pi_model.evaluate_policy()
    assert np.allclose(
        pi_model.get_all_values(),
        np.array([[  0., -14., -20., -22.],
                  [-14., -18., -20., -20.],
                  [-20., -20., -18., -14.],
                  [-22., -20., -14.,   0.]]).ravel(),
        rtol=1e-2, atol=1e-2
    )

    pi_model.improve_policy()
    assert np.allclose(
        pi_model.policy.to_numpy(),
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.  , 0.  , 1.  , 0.  ],
                  [0.  , 0.  , 1.  , 0.  ],
                  [0.  , 0.50, 0.50, 0.  ],
                  [1.  , 0.  , 0.  , 0.  ],
                  [0.50, 0.  , 0.50, 0.  ],
                  [0.  , 0.50, 0.50, 0.  ],
                  [0.  , 1.  , 0.  , 0.  ],
                  [1.  , 0.  , 0.  , 0.  ],
                  [0.50, 0.  , 0.  , 0.50],
                  [0.  , 0.50, 0.  , 0.50],
                  [0.  , 1.  , 0.  , 0.  ],
                  [0.50, 0.  , 0.  , 0.50],
                  [0.  , 0.  , 0.  , 1.  ],
                  [0.  , 0.  , 0.  , 1.  ],
                  [0.25, 0.25, 0.25, 0.25]]),
        rtol=1e-2, atol=1e-2
    )


def test_value_iteration_grid_44_allsteps(vi_on_44grid):
    vi_model = vi_on_44grid
    vi_model.transform()

    assert np.allclose(
        vi_model.get_all_values(),
        np.array([[ 0., -1., -2., -3.],
                  [-1., -2., -3., -2.],
                  [-2., -3., -2., -1.],
                  [-3., -2., -1.,  0.]]).ravel(),
        rtol=1e-2, atol=1e-2
    )

    assert np.allclose(
        vi_model.policy.to_numpy(),
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.  , 0.  , 1.  , 0.  ],
                  [0.  , 0.  , 1.  , 0.  ],
                  [0.  , 0.50, 0.50, 0.  ],
                  [1.  , 0.  , 0.  , 0.  ],
                  [0.50, 0.  , 0.50, 0.  ],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.  , 1.  , 0.  , 0.  ],
                  [1.  , 0.  , 0.  , 0.  ],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.  , 0.50, 0.  , 0.50],
                  [0.  , 1.  , 0.  , 0.  ],
                  [0.50, 0.  , 0.  , 0.50],
                  [0.  , 0.  , 0.  , 1.  ],
                  [0.  , 0.  , 0.  , 1.  ],
                  [0.25, 0.25, 0.25, 0.25]]),
        rtol=1e-2, atol=1e-2
    )


def test_policy_iteration_grid_44_allsteps(pi_on_44grid):
    pi_model = pi_on_44grid
    pi_model.transform()

    # this is not nesscary coverage to a same result with value iteration
    assert np.allclose(
        pi_model.get_all_values(),
        np.array([[ 0., -1., -2., -3.],
                  [-1., -2., -3., -2.],
                  [-2., -3., -2., -1.],
                  [-3., -2., -1.,  0.]]).ravel(),
        rtol=1e-2, atol=1e-2
    )

    assert np.allclose(
        pi_model.policy.to_numpy(),
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.  , 0.  , 1.  , 0.  ],
                  [0.  , 0.  , 1.  , 0.  ],
                  [0.  , 0.5 , 0.5 , 0.  ],
                  [1.  , 0.  , 0.  , 0.  ],
                  [0.5 , 0.  , 0.5 , 0.  ],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.  , 1.  , 0.  , 0.  ],
                  [1.  , 0.  , 0.  , 0.  ],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.  , 0.5 , 0.  , 0.5 ],
                  [0.  , 1.  , 0.  , 0.  ],
                  [0.5 , 0.  , 0.  , 0.5 ],
                  [0.  , 0.  , 0.  , 1.  ],
                  [0.  , 0.  , 0.  , 1.  ],
                  [0.25, 0.25, 0.25, 0.25]]),
        rtol=1e-2, atol=1e-2
    )
