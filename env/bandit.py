"""
MultiArm Bandit Toy Enviornment
"""
import numpy as np
import numpy.random as random
from .base import FiniteMDPENV


class MultiArmBandit(FiniteMDPENV):
    def __init__(self, nbandits=5):
        self.nbandits = nbandits
        self._actions = np.arange(nbandits)
        self._mu = 0
        self._sigma = 1
        self._truth = random.normal(
            loc=self._mu, scale=self._sigma, size=self.nbandits
        )
        self.s0 = None
    
    @property
    def A(self):
        return self._actions
    
    @property
    def S(self):
        return [0]
    
    def start(self):
        # get new means
        self.s0 = 0
        return self.s0
    
    def step(self, action):
        reward = random.normal(loc=self._truth[action], scale=self._sigma)
        # self._truth += random.normal(self._mu, self._sigma/10, size=self.nbandits)
        return self.s0, reward

    def transitions(self, state, action):
        trans = np.zeros(shape=(len(self.S), 2))
        reward = 0
        trans[self.s0] = (reward, 1)
        return trans

    def is_terminal(self):
        # continuous
        return False
