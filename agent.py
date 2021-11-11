"""
Agent module that contains you know all agents
"""
import numpy as np
from collections import namedtuple
from abc import abstractmethod
from rflearn.utils import argmax
from numpy.random import uniform, randint


AgentState = namedtuple('State', ['num_steps', 'q_values', 'action_count', 'last_action'])

class FiniteActionAgent:
    """
    Toy RL Agent that is based a finite set of action space

    parameters
    ----------
    num_actions: int, numer of actions avaliable for the agent
    init_q: float, the initial value or prior knowledge of the action-values
    """
    def __init__(self, num_actions, init_q=0):
        self.num_actions = num_actions
        self.init_q = init_q
        self.num_steps = 0
        self.reset()


    def reset(self):
        """reset all agent's knowledge to default settings"""
        self.action_count = [0 for _ in range(self.num_actions)]
        self.q_values = [self.init_q for _ in range(self.num_actions)]
        self.last_action = None
        self.num_steps = 0
    
    def get_state(self):
        """Get the current state of the agent"""
        return AgentState(
            num_steps=self.num_steps,
            q_values=self.q_values,
            action_count=self.action_count,
            last_action=self.last_action,
        )

    def get_action_cnt(self, action):
        """get the counts of a specific action that had been chosen"""
        return self.action_count[action]
    
    def inc_action_cnt(self, action):
        """increment the count of a specific action"""
        self.action_count[action] += 1
    
    def get_q(self, action):
        """get the estimated action-value for a specific action"""
        return self.q_values[action]
    
    def set_q(self, action, q):
        """set the new action-value for a specific action"""
        self.q_values[action] = q

    @abstractmethod
    def agent_step(self, reward, observation=None):
        """main interaction function that is used to update the beliefs"""
        raise NotImplementedError()


class EpsilonGreedyAgent(FiniteActionAgent):
    """
    Toy RL Agent that is based a finite set of action space

    parameters
    ----------
    num_actions: int, numer of actions avaliable for the agent
    epsilon: float ~ [0, 1], the Pr(exploration)
        when epsilon = 0  ->  full greedy agent
    init_q: float, the initial value or prior knowledge of the action-values
    step_size: float ~ [0, 1], the learning rate or λ
        default None  ->  use sample mean method with the rate of 1/N(a)
    """
    def __init__(self, num_actions, epsilon=0.1, init_q=0, step_size=None):
        super().__init__(num_actions, init_q)

        self.epsilon = epsilon
        self.step_size = step_size

    def __repr__(self):
        return f'EGAgt<ϵ: {self.epsilon}, λ: {self.step_size}>'

    def agent_step(self, reward, observation=None):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        parameters
        ----------
        reward: float, the reward the agent recieved from the environment after taking the last action.
        observation: float, the observed state the agent is in.
            Do not worry about this as you will not use it until future lessons

        return
        -------
        current_action: int, the action chosen by the agent at the current time step.
        """
        self.num_steps += 1

        # get updated q-value
        a0 = self.last_action
        q = self.get_updated_q(a0, reward)

        # update belief, and get new action
        self.set_q(a0, q)
        a1 = self.get_action()
        return a1 


    def get_action(self):
        """make the decision on which action to take for the current time step"""
        if uniform() < self.epsilon:
            action = randint(0, high=self.num_actions)
        else:
            action = argmax(self.q_values)

        self.last_action = action
        self.inc_action_cnt(action)
        return action

    
    def get_step_size(self, action):
        """step size utility function that switch between sample mean method or constant rate"""
        if self.step_size is None:
            num_a0 = self.action_count[action]
            return 1 / num_a0
        else:
            return self.step_size
    

    def get_updated_q(self, action, reward):
        """calcualtes the newly updated q-value after take in new data"""
        q = self.get_q(action)
        step_size = self.get_step_size(action)
        q += step_size * (reward - q)  # sample mean update
        return q


class UCBAgent(EpsilonGreedyAgent):
    """Upper Confidence Bound Eps Greedy Agent"""
    def __init__(self, num_actions, bound_c=2, init_q=0, step_size=None):
        super().__init__(num_actions, None, init_q, step_size)
        self.bound_c = bound_c

    def __repr__(self):
        return f'UCBAgt<c: {self.bound_c}, λ: {self.step_size}>'

    def get_action(self):
        """
        Make the decision on which action to take for the current time step
            using upper confidence bound.
        Essentially, it is a simple way to estimate the variance of rewards online.
        """
        if self.num_steps <= 1:  # log(0) is undefined, log(1) is 0 bad for division
            action = np.argmin(self.action_count)
        else:
            with np.errstate(divide='ignore'):  # intended inf with c/0 from numpy
                var = np.log(self.num_steps) / np.array(self.action_count)
            var = np.where(np.isnan(var), np.inf, var)
            bound = self.bound_c * np.sqrt(var)
            action = argmax(self.q_values + bound)

        self.last_action = action
        self.inc_action_cnt(action)
        return action
