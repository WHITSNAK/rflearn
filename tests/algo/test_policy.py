import pytest
import numpy as np
from rflearn.algo import TabularPolicy
from rflearn.algo import State


@pytest.fixture
def policy1():
    s1 = State(np.array([1,2,3]))
    s2 = State(np.array([1,2,3]))
    s3 = State(np.array([1,2,3]))
    pol = TabularPolicy(
        [s1,s2,s3], ['a','b','c'],
        policy={s1:[0,0,1], s2:[0,1,0], s3:[0.5,0,0.5]}
    )
    return pol


@pytest.fixture
def policy_eps():
    s1 = State(np.array([1,2,3]))
    s2 = State(np.array([1,2,3]))
    s3 = State(np.array([1,2,3]))
    pol = TabularPolicy(
        [s1,s2,s3], ['a','b','c'],
        epsilon=0.1,
        policy={s1:[0,0,1], s2:[0,1,0], s3:[0.5,0,0.5]}
    )
    return pol


def test_init(policy1):
    pol = policy1
    s1, s2, s3 = pol.states

    assert pol.shape == (3, 3)
    assert pol.get_aidx('a') == 0
    assert pol.get_aidx('c') == 2
    assert pol[s2] == [0, 1, 0]

    pol[s2] = [1, 0, 0]
    assert pol[s2] == [1, 0, 0]


def test_invariant(policy1):
    pol = policy1
    s1, s2, s3 = pol.states

    with pytest.raises(AssertionError):
        pol[s1] = [1, 0, 1]


def test_to_numpy(policy1):
    pol = policy1

    assert np.array_equal(
        pol.to_numpy(),
        np.array([[0. , 0. ,1. ],
                  [0. , 1. ,0. ],
                  [0.5, 0. ,0.5]]),
    )


@pytest.mark.parametrize(
    'fnm, state_idx, qs, expected',
    [
        ('policy1', 0, [100, 0, 50], [1, 0, 0]),
        ('policy1', 0, [1.222, 0, 1.222], [0.5, 0, 0.5]),
        ('policy1', 0, [1.222, 0, 1.223], [0, 0, 1]),
        ('policy1', 0, [0, 0, 0], [1/3, 1/3, 1/3]),
    ]
)
def test_greedify(fnm, state_idx, qs, expected, request):
    pol = request.getfixturevalue(fnm)
    state = pol.states[state_idx]
    new_π = pol.greedify(state, qs)
    assert np.allclose(new_π, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    'fnm, state_idx, qs, expected',
    [
        ('policy_eps', 0, [100, 0, 50], [0.933, 0.033, 0.033]),
        ('policy_eps', 0, [1.222, 0, 1.222], [0.483, 0.033, 0.483]),
        ('policy_eps', 0, [1.222, 0, 1.223], [0.033, 0.033, 0.933]),
        ('policy_eps', 0, [0, 0, 0], [1/3, 1/3, 1/3]),
    ]
)
def test_greedify_eps(fnm, state_idx, qs, expected, request):
    pol = request.getfixturevalue(fnm)
    state = pol.states[state_idx]
    new_π = pol.greedify(state, qs)
    assert np.allclose(new_π, expected, rtol=1e-2, atol=1e-2)