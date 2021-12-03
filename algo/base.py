"""
Module that hosts base/abstract algorithm
"""
import uuid
from abc import abstractmethod


class State:
    """Simple state encoder that is invariant to the atual values"""
    def __init__(self, val):
        # random id unique to the Env
        self.id = uuid.uuid4()
        self.val = val
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'State<{self.val}>'
    
    def get_val(self):
        return self.val


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
      