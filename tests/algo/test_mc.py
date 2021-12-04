import pytest
import numpy as np
from unittest.mock import patch
from rflearn.algo import MCIteration, TabularPolicy, TabularQValue
from rflearn.algo import Episode
from rflearn.env import GridWorld


@pytest.fixture
def mc1():
    grid = GridWorld(4, 4)
    qvalue = TabularQValue(grid.S, grid.A)
    policy = TabularPolicy(grid.S, grid.A, epsilon=0.05)

    mc_model = MCIteration(grid, qvalue, policy)
    return mc_model

@pytest.fixture
def mc2():
    grid = GridWorld(2, 2)
    qvalue = TabularQValue(grid.S, grid.A)
    policy = TabularPolicy(grid.S, grid.A, epsilon=0.05)

    mc_model = MCIteration(grid, qvalue, policy)
    return mc_model


def test_init(mc1):
    mc = mc1
    mc.fit(gamma=1, alpha=0.1)
    assert 'sa_counts' not in mc.__dict__
    assert mc.gamma == 1
    assert mc.alpha == 0.1
    assert len(mc.hist) == 0

    mc.fit(gamma=1, alpha=None)
    assert 'sa_counts' in mc.__dict__


def test_get_episodes(mc1):
    mc = mc1
    eps = mc.get_episodes(n=5)
    assert len(eps) == 5
    assert type(eps[0]) is Episode


def test_transform(mc1):
    mc = mc1
    mc.fit(gamma=1, alpha=None)
    with patch.object(MCIteration, 'get_episodes', return_value=0), \
         patch.object(MCIteration, 'evaluate_policy', return_value=1), \
         patch.object(MCIteration, 'improve_policy'):
        mc.transform(iter=10, kbatch=30, pbar_leave=False)
        mc.evaluate_policy.assert_called_with(0)
        mc.improve_policy.assert_called_with(1)
        assert 0 in mc.hist


def test_iteration(mc2):
    mc = mc2
    np.random.seed(45345)
    mc.fit(gamma=1, alpha=None)
    mc.transform(500, kbatch=30, pbar_leave=False)

    values = np.round(mc.qvalue.get_all_values(mc.policy).reshape(mc.env.shape), 0)
    assert np.allclose(
        values,
        np.array([
            [ 0, -1],
            [-1,  0],
        ])
    )


def test_iteration_fixed_alpha(mc2):
    mc = mc2
    np.random.seed(45345)
    mc.fit(gamma=1, alpha=0.1)
    mc.transform(500, kbatch=30, pbar_leave=False)

    values = np.round(mc.qvalue.get_all_values(mc.policy).reshape(mc.env.shape), 0)
    assert np.allclose(
        values,
        np.array([
            [ 0, -1],
            [-1,  0],
        ])
    )