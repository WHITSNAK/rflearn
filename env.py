"""
Module that hosts all Agent Environments
"""
from abc import abstractmethod
import numpy as np
from itertools import product


class FiniteMDPENV:
    """Abstract Finite MDP Enviornment that ensures template/API"""
    @abstractmethod
    def A(self):
        return NotImplementedError()
    
    @abstractmethod
    def S(self):
        return NotImplementedError()
    
    @abstractmethod
    def step(self):
        return NotImplementedError()
    
    @abstractmethod
    def transitions(self):
        return NotImplementedError()


class GridWorld(FiniteMDPENV):
    ACTIONS = ['up', 'down', 'left', 'right']
    ACTION_MAP = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }

    def __init__(self, nrow, ncol, grid=None):
        self.nrow = nrow
        self.ncol = ncol
        self.shape = (nrow, ncol)
        self.grid = [[0]*ncol for _ in range(nrow)] if grid is None else grid
    
    def __iter__(self):
        for cell in product(range(self.nrow), range(self.ncol)):
            yield cell
    
    @property
    def A(self):
        return self.ACTIONS
    
    @property
    def S(self):
        return [state[0]*self.ncol + state[1] for state in self]

    def _to_square(self, state):
        return state//self.nrow, state%self.ncol
    
    def _to_state(self, row, col):
        return row * self.ncol + col

    def is_terminal(self, state):
        return state in [0, self.nrow*self.ncol-1]

    def get_move(self, action):
        return self.ACTION_MAP[action.lower()]

    def get_reward(self, state):
        if self.is_terminal(state):
            return 0
        else:
            return -1
    
    def get_new_state(self, state, move):
        row, col = self._to_square(state)
        _row = min(max(row + move[0], 0), self.nrow-1)
        _col = min(max(col + move[1], 0), self.ncol-1)
        new_state = self._to_state(_row, _col)
        return new_state
    
    def get_four_neighbors(self, state):
        return [
            self.get_new_state(state, self.get_move(move))
            for move in self.ACTIONS
        ]

    def step(self, state, action):
        move = self.get_move(action)
        new_state = self.get_new_state(state, move)
        reward = self.get_reward(state)
        return new_state, reward
    
    def transitions(self, state, action):
        new_state, reward = self.step(state, action)
        trans = []
        for s1 in self.S:
            if self.is_terminal(state):
                trans.append((0, 0))
            elif s1 == new_state:
                trans.append((reward, 1))
            else:
                trans.append((0 ,0))
        return np.array(trans)