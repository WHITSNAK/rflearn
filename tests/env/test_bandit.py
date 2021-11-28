import pytest
import numpy as np
from unittest.mock import patch
from rflearn.env import bandit


@pytest.fixture
def bandit3_f():
    return bandit.MultiArmBandit(3, moving_target=False)


@pytest.fixture
def bandit3_s():
    return bandit.MultiArmBandit(3, moving_target=True)


def test_init(bandit3_f):
    env = bandit3_f
    assert env.S == [0]
    assert env.A == [0, 1, 2]
    assert env.s0 == 0  # never change this s0
    assert env.start() == 0
    assert not env.is_terminal()  # continuous
    assert env._truth.shape[0] == len(env.A)

    # fixed transitions, deterministic
    assert env.transitions(0, 1).shape == (1, 2)
    assert env.transitions(0, 1)[0, 1] == 1


def test_step_f(bandit3_f):
    env = bandit3_f

    with patch.object(bandit.MultiArmBandit, '_get_reward', return_value=1):
        s0 = env.start()
        s1, r = env.step(0)
        assert s1 == 0
        assert r == 1
        env._get_reward.assert_called_once_with(0)


def test_step_s(bandit3_s):
    env = bandit3_s
    
    old_truth = env._truth.copy()
    with patch('rflearn.env.bandit.normal', return_value=np.array([1,1,1])):
        s0 = env.start()
        a0 = 0
        s1, r = env.step(a0)

        # test moving target shifts
        bandit.normal.assert_called_with(env._mu, env._sigma/10, size=env.nbandits)
        assert np.array_equal(old_truth + bandit.normal.return_value, env._truth)
