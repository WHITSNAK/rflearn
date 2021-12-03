import pytest
import numpy as np
from rflearn.algo import TabularQValue, TabularPolicy
from rflearn.env import GridWorld


@pytest.fixture
def grid22():
    return GridWorld(2, 2)

@pytest.fixture
def qvalue1(grid22):
    env = grid22
    qvalue = TabularQValue(
        env.S, env.A,
        {
            0: [0, 0.5, 0, 0.5],
            1: [0, 0.5, 0.5, 0],
            2: [1, 0, 0, 0],
            3: [0.5, 0, 0.5, 0],
        }
    )
    return qvalue

@pytest.fixture
def policy1(grid22):
    env = grid22
    policy = TabularPolicy(env.S, env.A, epsilon=0)
    return policy


def test_init(qvalue1):
    qv = qvalue1

    assert len(qv.states) == 4
    assert len(qv.actions) == 4
    assert qv.shape == (4, 4)
    assert qv[0] == [0, 0.5, 0, 0.5]
    assert qv[3] == [0.5, 0, 0.5, 0]
    
    qv[3] = [1, 0, 0, 0]
    assert qv[3] == [1, 0, 0, 0]

    assert qv.get_aidx('up') == 0
    assert qv.get_aidx('down') == 1

def test_invariant(qvalue1):
    qv = qvalue1

    with pytest.raises(AssertionError):
        qv[0] = [0, 0]


def test_to_numpy(qvalue1):
    qv = qvalue1
    assert np.array_equal(
        qv.to_numpy(),
        np.array([[0,0.5,0,0.5], [0,0.5,0.5,0], [1,0,0,0], [0.5,0,0.5,0]])
    )


@pytest.mark.parametrize(
    'fnm, state, pi, expected',
    [
        ('qvalue1', 0, [0,0,0,1], 0.5),
        ('qvalue1', 3, [1,0,1,0], 1),
    ]
)
def test_get_value(fnm, state, pi, expected, request):
    qv = request.getfixturevalue(fnm)
    assert qv.get_value(state, pi) == expected


@pytest.mark.parametrize(
    'fnm, state, expected',
    [
        ('qvalue1', 0, 0.5),
        ('qvalue1', 2, 1),
    ]
)
def test_get_value(fnm, state, expected, request):
    qv = request.getfixturevalue(fnm)
    assert qv.get_maxq(state) == expected


def test_get_all_values(qvalue1, policy1):
    qv = qvalue1
    pol = policy1
    assert np.allclose(
        qv.get_all_values(pol),
        np.array([0.25, 0.25, 0.25, 0.25])
    )

def test_q_getter_setter(qvalue1):
    qv = qvalue1
    assert qv.get_q(0, 'down') == 0.5
    assert qv.get_q(2, 'up') == 1

    assert qv.get_q(0, 'up') == 0
    qv.set_q(0, 'up', 1)
    assert qv.get_q(0, 'up') == 1
