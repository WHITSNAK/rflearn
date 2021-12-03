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
        {s1:[0,0,1], s2:[0,1,0], s3:[0.5,0,0.5]}
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
    