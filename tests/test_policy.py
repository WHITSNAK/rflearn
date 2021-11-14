import pytest
import numpy as np
from rflearn.policy import PolicyIteration, ValueIteration
from rflearn.env import GridWorld


@pytest.fixture
def grid44_random():
    grid = GridWorld(4, 4)
    value = np.zeros(shape=len(grid.S))
    policy = np.ones(shape=(value.shape + (len(grid.ACTIONS), ))) / len(grid.ACTIONS)
    return grid, value, policy


@pytest.fixture
def pi_on_44grid(grid44_random):
    grid, value, policy = grid44_random

    pi_model = PolicyIteration(theta=0.001)
    pi_model.fit(grid, value, policy, gamma=1)
    return pi_model

@pytest.fixture
def vi_on_44grid(grid44_random):
    grid, value, policy = grid44_random

    pi_model = ValueIteration(theta=0.001)
    pi_model.fit(grid, value, policy, gamma=1)
    return pi_model

def test_policy_iteration_grid_44_1step(pi_on_44grid):
    pi_model = pi_on_44grid
    pi_model.evaluate_policy()
    pi_model.improve_policy()

    assert np.allclose(
        pi_model.value,
        np.array([[  0., -14., -20., -22.],
                  [-14., -18., -20., -20.],
                  [-20., -20., -18., -14.],
                  [-22., -20., -14.,   0.]]).ravel(),
        rtol=1e-2, atol=1e-2
    )

    assert np.allclose(
        pi_model.policy,
        np.array([[1., 1., 1., 1.],
                  [0., 0., 1., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 1., 0.],
                  [1., 0., 0., 0.],
                  [1., 0., 1., 0.],
                  [0., 1., 1., 0.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 0.],
                  [1., 0., 0., 1.],
                  [0., 1., 0., 1.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 1.],
                  [0., 0., 0., 1.],
                  [0., 0., 0., 1.],
                  [1., 1., 1., 1.]]),
        rtol=1e-2, atol=1e-2
    )


def test_value_iteration_grid_44_allsteps(vi_on_44grid):
    pi_model = vi_on_44grid
    pi_model.transform()

    assert np.allclose(
        pi_model.value,
        np.array([[ 0., -1., -2., -3.],
                  [-1., -2., -3., -2.],
                  [-2., -3., -2., -1.],
                  [-3., -2., -1.,  0.]]).ravel(),
        rtol=1e-2, atol=1e-2
    )

    assert np.allclose(
        pi_model.policy,
        np.array([[1., 1., 1., 1.],
                  [0., 0., 1., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 1., 0.],
                  [1., 0., 0., 0.],
                  [1., 0., 1., 0.],
                  [1., 1., 1., 1.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 0.],
                  [1., 1., 1., 1.],
                  [0., 1., 0., 1.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 1.],
                  [0., 0., 0., 1.],
                  [0., 0., 0., 1.],
                  [1., 1., 1., 1.]]),
        rtol=1e-2, atol=1e-2
    )


def test_policy_iteration_grid_44_allsteps(pi_on_44grid):
    pi_model = pi_on_44grid
    pi_model.transform()

    # this is not nesscary coverage to a same result with value iteration
    assert np.allclose(
        pi_model.value,
        np.array([[ 0., -1., -2., -6.],
                  [-1., -4., -6., -2.],
                  [-2., -6., -4., -1.],
                  [-6., -2., -1.,  0.]]).ravel(),
        rtol=1e-2, atol=1e-2
    )

    assert np.allclose(
        pi_model.policy,
        np.array([[1., 1., 1., 1.],
                  [0., 0., 1., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 1., 0.],
                  [1., 0., 0., 0.],
                  [1., 0., 1., 0.],
                  [1., 0., 0., 1.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 0.],
                  [0., 1., 1., 0.],
                  [0., 1., 0., 1.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 1.],
                  [0., 0., 0., 1.],
                  [0., 0., 0., 1.],
                  [1., 1., 1., 1.]]),
        rtol=1e-2, atol=1e-2
    )
