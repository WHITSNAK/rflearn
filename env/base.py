"""
Base Enviornment Module
"""
from abc import abstractmethod


class FiniteMDPENV:
    """
    Abstract Finite MDP Enviornment that ensures template/API
    - it has finite discrete action space
    - and it also has finite discrete state space
    """
    @abstractmethod
    def A(self):
        raise NotImplementedError()
    
    @abstractmethod
    def S(self):
        raise NotImplementedError()
    
    @abstractmethod
    def start(self):
        raise NotImplementedError()
    
    @abstractmethod
    def step(self):
        raise NotImplementedError()
    
    @abstractmethod
    def transitions(self):
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self, state):
        raise NotImplementedError()
