import pytest
import numpy as np
from copy import deepcopy
from rflearn.agent import EpsilonGreedyAgent, UCBAgent


def setup_agent(agent_obj, data):
    """Sets up an agent with all the parameters and preset values"""
    agent = agent_obj(**data['parameters'])
    for k, v in data['direct'].items():
        setattr(agent, k, v)
    return agent


@pytest.fixture
def epsagent():
    data = {
        'seed': 0,
        'parameters': {
            'num_actions': 5,
            'epsilon': 0,
            'init_q': 0,
        },
        'direct': {
            'q_values': [0, 0, 1, 0, 0],
            'action_count': [0, 2, 0, 0, 0],
            'last_action': 1
        },
    }
    return data


@pytest.fixture
def ucbagent():
    return {
        'seed': 0,
        'parameters': {
            'num_actions': 5,
            'init_q': 0,
            'step_size': 1,
            'bound_c': 2,
        },
        'direct': {
            'q_values': [0, 0, 1, 0, 0],
            'action_count': [0, 2, 0, 0, 0],
            'last_action': 1
        },
    }


def test_agent_basics(epsagent):
    agent = setup_agent(EpsilonGreedyAgent, epsagent)

    action = agent.get_action()
    # make sure all bookkeepings update to date
    assert action == 2
    assert agent.last_action == action
    assert agent.action_count[action] == 1
    assert agent.get_action_cnt(action) == 1


def test_greedy_agent_1steps(epsagent):
    # build a fake agent for testing and set some initial conditions
    np.random.seed(0)
    agent = setup_agent(EpsilonGreedyAgent, epsagent)
    action = agent.agent_step(reward=1)

    # make sure the q_values were updated correctly
    # and make sure the agent is using the argmax that breaks ties randomly
    assert agent.q_values == [0, 0.5, 1, 0, 0]
    assert action == 2
    assert agent.num_steps == 1


def test_greedy_agent_2steps(epsagent):
    # build a fake agent for testing and set some initial conditions
    np.random.seed(1)
    agent = setup_agent(EpsilonGreedyAgent, epsagent)

    # take a fake agent step
    action = agent.agent_step(reward=1)
    assert agent.action_count == [0, 2, 1, 0, 0]
    assert action == 2
    assert agent.q_values == [0, 0.5, 1.0, 0, 0]

    # take another step
    action = agent.agent_step(reward=2)
    assert agent.action_count == [0, 2, 2, 0, 0]
    assert action == 2
    assert agent.q_values == [0, 0.5, 2.0, 0, 0]
    assert agent.num_steps == 2


def test_epsilon_agent(epsagent):
    np.random.seed(0)
    ega = setup_agent(EpsilonGreedyAgent, epsagent)
    ega.epsilon = 0.5

    action = ega.agent_step(reward=1, observation=0)
    assert ega.q_values == [0, 0.5, 1.0, 0, 0]

    # manipulate the random seed so the agent takes a random action
    np.random.seed(1)
    action = ega.agent_step(reward=0, observation=0)
    assert action == 4

    # check to make sure we update value for action 4
    action = ega.agent_step(reward=1, observation=0)
    assert ega.q_values == [0, 0.5, 0.0, 0, 1.0]


def test_epsilon_agent_constant_step(epsagent):
    np.random.seed(0)
    # Check Epsilon Greedy with Different Constant Stepsizes
    data = epsagent
    for step_size in [0.01, 0.1, 0.5, 1.0]:
        _data = deepcopy(data)  # avoids direct mutation
        _data['parameters']['step_size'] = step_size
        ega = setup_agent(EpsilonGreedyAgent, _data)
        
        action = ega.agent_step(reward=1, observation=0)
        assert ega.q_values == [0, step_size, 1.0, 0, 0]    


def test_ucb1(ucbagent):
    agent = setup_agent(UCBAgent, ucbagent)
    
    assert agent.num_steps == 0
    action = agent.get_action()
    assert agent.get_action_cnt(action) == 1
    assert action == 0

    for _ in range(3):
        agent.get_action()
    assert (np.array(agent.action_count) == np.array([1,2,1,1,1])).all()


def test_ucb2(ucbagent):
    agent = setup_agent(UCBAgent, ucbagent)

    for i in range(4):
        action = agent.agent_step(reward=(10-i)/10)
    
    # action 1 has q=1 but cnts=2
    # last action has q=1, thus choose last action
    assert action == agent.agent_step(reward=1)
