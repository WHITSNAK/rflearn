import pytest
from rflearn.algo.episode import Episode


@pytest.fixture
def episode_ups():
    ep = Episode()

    for _ in range(20):
        ep.add_step([0,1], 'up', 0, [1,1], False)
    ep.add_step([1,1], 'right', 10, [1,1], True)

    return ep


def test_episode_init():
    ep = Episode()

    assert ep.nsteps == 0
    assert ep.steps == []

    ep.add_step(100, 'a', 10, 200, False)
    assert ep.nsteps == 1
    assert len(ep.steps) == ep.nsteps
    assert ep.steps[-1].a0 == 'a'


def test_episode(episode_ups):
    ep = episode_ups

    assert len(ep) == 21 == ep.nsteps
    assert ep[-1].is_terminal is True
    assert ep.get_total_rewards() == 10
    assert ep.get_avg_rewards() == 10/21
