import pytest
import numpy as np
from rflearn.env import GridWorld, WindGridWorld, CliffGridWorld

#################################################
# Regular GridWorld Test Suites
#################################################

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


#################################################
# Windy GridWorld Test Suites
#################################################

@pytest.fixture
def wgrid():
    return WindGridWorld()


def test_wgrid_init(wgrid):
    env = wgrid
    assert env.s0 is None
    s0 = env.start()
    assert s0 == 30  # fixed starting point
    assert env.terminals == [37]  # one terminal


@pytest.mark.parametrize(
    'env_nm, s0, a0, s1_expected',
    [
        ('wgrid', 63, 'left', 52),
        ('wgrid', 63, 'right', 54),
        ('wgrid', 63, 'up', 43),
        ('wgrid', 16, 'up', 6),
        ('wgrid', 26, 'down', 16),
        ('wgrid', 36, 'left', 15),
        ('wgrid', 36, 'right', 17),
    ]
)
def test_wgrid_step(env_nm, s0, a0, s1_expected, request):
    env = request.getfixturevalue(env_nm)
    env.s0 = s0
    s1, r = env.step(a0)
    assert s1 == s1_expected


#################################################
# Cliff GridWorld Test Suites
#################################################

@pytest.fixture
def cgrid_s():
    return CliffGridWorld(nrow=3, ncol=4)


def test_cgrid_init(cgrid_s):
    env = cgrid_s
    assert env.s0 is None
    s0 = env.start()
    assert s0 == 8
    assert env.terminals == [11]


@pytest.mark.parametrize(
    'env_nm, s, flag',
    [
        ('cgrid_s', 8, False),
        ('cgrid_s', 0, False),
        ('cgrid_s', 9, True),
        ('cgrid_s', 10, True),
        ('cgrid_s', 11, False),
    ]
)
def test_cgrid_is_cliff(env_nm, s, flag, request):
    env = request.getfixturevalue(env_nm)
    assert env.is_cliff(s) == flag


@pytest.mark.parametrize(
    'env_nm, s0, a0, s1_exp, r_exp',
    [
        ('cgrid_s', 9, 'right', 8, -100),
        ('cgrid_s', 10, 'right', 8, -100),
        ('cgrid_s', 8, 'right', 9, -1),
        ('cgrid_s', 9, 'up', 8, -100),
        ('cgrid_s', 4, 'left', 4, -1),
        ('cgrid_s', 5, 'up', 1, -1),
    ]
)
def test_cgrid_step(env_nm, s0, a0, s1_exp, r_exp, request):
    env = request.getfixturevalue(env_nm)
    env.s0 = s0
    s1, r = env.step(a0)
    assert s1 == s1_exp
    assert r == r_exp
