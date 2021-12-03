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


def test_init(mc1):
    mc = mc1
    mc.fit(gamma=1, alpha=0.1)
    assert 'sa_counts' not in mc.__dict__
    assert mc.gamma == 1
    assert mc.alpha == 0.1
    assert len(mc.hist) == 0
    assert len(mc.last_updated_s) == 0

    mc.fit(gamma=1, alpha=None)
    assert 'sa_counts' in mc.__dict__


def test_get_episodes(mc1):
    mc = mc1
    eps = mc.get_episodes(n=5)
    assert len(eps) == 5
    assert type(eps[0]) is Episode

    eps = mc.get_episodes(n=10, max_steps=1)
    for ep in eps:
        assert len(ep) <= 1


def test_transform(mc1):
    mc = mc1
    with patch.object(MCIteration, 'evaluate_policy'), \
         patch.object(MCIteration, 'improve_policy'):
        mc.transform(iter=10, kbatch=30, max_steps=None, pbar_leave=False)
        mc.evaluate_policy.assert_called_with(30, None)
        mc.improve_policy.assert_called()


def test_iteration(mc1):
    mc = mc1
    np.random.seed(45345)
    mc.fit(gamma=1, alpha=None)
    mc.transform(500, kbatch=30, max_steps=None, pbar_leave=False)

    values = np.round(mc.qvalue.get_all_values(mc.policy).reshape(mc.env.shape), 0)
    np.allclose(
        values,
        np.array([
            [ 0, -1, -2, -3],
            [-1, -2, -3, -2],
            [-2, -3, -2, -1],
            [-3, -2, -1,  0],
        ])
    )

    np.allclose(
        mc.policy.to_numpy(),
        np.array([
            [0.25  , 0.25  , 0.25  , 0.25  ],
            [0.0125, 0.0125, 0.9625, 0.0125],
            [0.0125, 0.0125, 0.9625, 0.0125],
            [0.0125, 0.9625, 0.0125, 0.0125],
            [0.9625, 0.0125, 0.0125, 0.0125],
            [0.0125, 0.0125, 0.9625, 0.0125],
            [0.0125, 0.0125, 0.0125, 0.9625],
            [0.0125, 0.9625, 0.0125, 0.0125],
            [0.9625, 0.0125, 0.0125, 0.0125],
            [0.0125, 0.0125, 0.0125, 0.9625],
            [0.0125, 0.0125, 0.0125, 0.9625],
            [0.0125, 0.9625, 0.0125, 0.0125],
            [0.0125, 0.0125, 0.0125, 0.9625],
            [0.0125, 0.0125, 0.0125, 0.9625],
            [0.0125, 0.0125, 0.0125, 0.9625],
            [0.25  , 0.25  , 0.25  , 0.25  ],
        ])
    )
