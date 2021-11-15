"""
Module that hosts base/abstract algorithm
"""
from abc import abstractmethod


class GPI:
    """
    General Policy Interation Framework
    - it does not impose resitriction on how to obtain q(s,a) or v(s)
    - it can be either on-policy or off-policy
    - with model or without model

    fit parameter
    -------------
    env: the enviornment
    value: [1 x S], expected value for all states in a row vector
    policy: [S x A], stochastic policy for all states map to action
    """
    def __init__(self, env, value, policy):
        self.env = env
        self.value = value.copy()
        self.policy = policy.copy()

    @abstractmethod
    def fit(self):
        """Fit the model with the env other policy"""
        return NotImplemented
    
    @abstractmethod
    def transform(self):
        """Finds the optimal policy"""
        self.evaluate_policy()
        self.improve_policy()
        return NotImplemented

    @abstractmethod
    def evaluate_policy(self):
        """Ways of updating value function"""
        return NotImplemented
    
    @abstractmethod
    def improve_policy(self):
        """Improve the current policy"""
        return NotImplemented

    @abstractmethod
    def get_qs(self, state):
        """Used for value updating from the update diagram"""
        return NotImplemented
      