import pytest
import numpy as np
from rflearn.env import GridWorld


@pytest.fixture
def grid2x2():
    return GridWorld(2, 2, terminals=[0])

@pytest.fixture
def grid3x4():
    return GridWorld(3, 4)

@pytest.fixture
def grid3x3_b():
    _grid = [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    return GridWorld(grid=_grid, terminals=[0])


@pytest.mark.parametrize(
    'fnm, terminals',
    [
        ('grid2x2', [0]),
        ('grid3x4', [0, 11]),
        ('grid3x3_b', [0]),
    ]
)
def test_random_start(fnm, terminals, request):
    grid = request.getfixturevalue(fnm)
    s0_lst = []
    for _ in range(1000):
        s0_lst.append(grid.start())
    
    for t in terminals:
        assert t not in s0_lst


@pytest.mark.parametrize(
    'fnm, s0, a, s1, r, is_t',
    [
        ('grid2x2', 1, 'left', 0, -1, True),
        ('grid2x2', 2, 'down', 2, -1, False),
        ('grid2x2', 2, 'right', 3, -1, False),
        ('grid3x3_b', 4, 'up', 1, -1, False),
        ('grid3x3_b', 1, 'left', 0, -10, True),
    ]
)
def test_step(fnm, s0, a, s1, r, is_t, request):
    grid = request.getfixturevalue(fnm)
    grid.s0 = s0
    s1, r = grid.step(a)
    assert s1 == s1
    assert r == r
    assert grid.is_terminal() == is_t


def test_transitions(grid2x2):
    grid = grid2x2
    s0, a = 1, 'left'
    trans = grid.transitions(s0, a)
    assert trans[0, 1] == 1
    assert trans[0, 0] == -1
    assert np.isclose(trans[1:, :], np.zeros(shape=(3, 2))).all()

    s0, a = 0, 'up'
    trans = grid.transitions(s0, a)
    assert np.isclose(trans, np.zeros(shape=(4, 2))).all()
