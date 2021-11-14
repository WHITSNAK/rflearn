"""
Module that hosts all Agent Environments
"""
from abc import abstractmethod
import numpy as np
from itertools import product


class FiniteMDPENV:
    """
    Abstract Finite MDP Enviornment that ensures template/API
    - it has finite discrete action space
    - and it also has finite discrete state space
    """
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
    """
    2D Grid World Enviornment with blockage support
    - if the grid value is 0, it has -1 reward
    - if the grid value is 1, it has -10 reward, aka the blockage

    Avaliable actions: {up, down, left, right}
    Avaliable states: {0, 1, ...., nrow * ncol -1}
    System transition: visible

    parameter
    ---------
    nrow, ncol: int, the two dimensions of the grid
    grid: list of list, the actual 2D grid with 0/1 values
        it will replace nrow, ncol if this is given
    terminals: list of states, states that count as the terminal state with 0 reward
    """
    ACTIONS = ['up', 'down', 'left', 'right']
    ACTION_MAP = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }

    def __init__(self, nrow=None, ncol=None, grid=None, terminals=None):
        if grid is not None:
            self.grid = grid
            self.nrow = len(grid)
            self.ncol = len(grid[0])
        else:
            self.nrow = nrow
            self.ncol = ncol
            self.grid = [[0]*ncol for _ in range(nrow)]

        self.shape = (self.nrow, self.ncol)
        self.terminals = \
            [0, self.nrow * self.ncol - 1] \
            if terminals is None else terminals
    
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
        return state//self.ncol, state%self.ncol
    
    def _to_state(self, row, col):
        return row * self.ncol + col

    def is_terminal(self, state):
        return state in self.terminals

    def get_move(self, action):
        return self.ACTION_MAP[action.lower()]

    def get_reward(self, state):
        if self.is_terminal(state):
            return 0
        else:
            row, col = self._to_square(state)
            if self.grid[row][col] == 1:  # blocks
                return -10
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
        # essentially, rewards are the number of steps to terminal state
        reward = self.get_reward(state)
        move = self.get_move(action)
        new_state = self.get_new_state(state, move)
        return new_state, reward
    
    def transitions(self, state, action):
        new_state, reward = self.step(state, action)
        trans = np.zeros(shape=(len(self.S), 2))
        
        if self.is_terminal(state):
            return trans
        
        trans[new_state] = (reward, 1)
        return trans
