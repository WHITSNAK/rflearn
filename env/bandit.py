"""
MultiArm Bandit Toy Enviornment
"""
import numpy as np
from numpy.random import normal
from .base import FiniteMDPENV


class MultiArmBandit(FiniteMDPENV):
    """
    Muti Arm Bandit Toy Enviornment
    - it generates bandit machines with ground truth μ[i] ~ N(0, 1)
    - every reward r[i] is also generated out of ~ N(μ[i], 1)
    - moving target is set up ~ N(0, 0.1) normal noise

    Avaliable actions: {0, 1, 2, ...., nbandits-1}
    Avaliable states: {0}  # no shifting states
    System transition: visible, there no states

    parameter
    ---------
    nbandits: int, number of bandit arms/actions
    moving_target: bool, whether to move the ground truth on every step
    """
    def __init__(self, nbandits=5, moving_target=False):
        self.nbandits = nbandits
        self.moving_target = moving_target

        self._actions = list(range(nbandits))
        self._mu = 0
        self._sigma = 1
        self._truth = normal(
            loc=self._mu, scale=self._sigma, size=self.nbandits
        )
        self.s0 = 0  # fixed state, no variation
    
    @property
    def A(self):
        return self._actions
    
    @property
    def S(self):
        return [self.s0]
    
    def start(self):
        """start the env"""
        return self.s0
    
    def step(self, action):
        """take one step, action"""
        reward = self._get_reward(action)
        if self.moving_target:
            self._truth += normal(self._mu, self._sigma/10, size=self.nbandits)
        return self.s0, reward

    def transitions(self, state, action):
        """the full system transition p(r, s'|s, a)"""
        trans = np.zeros(shape=(len(self.S), 2))
        reward = self._get_reward(action)
        trans[self.s0] = (reward, 1)
        return trans

    def is_terminal(self):
        """never terminates, it is continuous"""
        return False

    def _get_reward(self, action):
        return normal(loc=self._truth[action], scale=self._sigma)
    