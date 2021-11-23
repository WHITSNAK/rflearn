"""
Module that hosts all Agent Environments
"""
import numpy as np
from itertools import product
from .base import FiniteMDPENV


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
        # default top-left, and bottom-right
        self.terminals = \
            [0, self.nrow * self.ncol - 1] \
            if terminals is None else terminals
        self.s0 = None
    
    def __iter__(self):
        for cell in product(range(self.nrow), range(self.ncol)):
            yield cell
    
    @property
    def A(self):
        """Actions set"""
        return self.ACTIONS
    
    @property
    def S(self):
        """States set, in numerical index in left to right order"""
        return [state[0]*self.ncol + state[1] for state in self]

    def start(self):
        """
        Start at a random position that is not terminal state
        
        return
        ------
        state s0
        """
        valid_states = list(filter(lambda x: not self._is_terminal(x), self.S))
        self.s0 = np.random.choice(valid_states)
        return self.s0

    def step(self, action):
        """
        Take action aka take a step forward from the agent

        parameter
        ---------
        action: a ∈ A, valid action to take

        return
        ------
        s1, r1: new state and observed reward
        """
        new_state, reward = self._step(self.s0, action)
        self.s0 = new_state
        return new_state, reward

    def is_terminal(self):
        """boolean, is terimnal state"""
        return self._is_terminal(self.s0)

    def transitions(self, state, action):
        """
        return the full enviornmnet dynamic transition p(s1, r1|s0, a0)

        return
        ------
        Matrix: [S x 2(reward, p)]
        """
        new_state, reward = self._step(state, action)
        trans = np.zeros(shape=(len(self.S), 2))
        
        if self._is_terminal(state):
            return trans
        
        trans[new_state] = (reward, 1)
        return trans
    
    def _step(self, state, action):
        """internal step forward function. Given the s0 a0, returns s1 r1"""
        # essentially, rewards are the number of steps to terminal state
        reward = self._get_reward(state)
        move = self._get_move(action)
        new_state = self._get_new_state(state, move)
        return new_state, reward

    def _is_terminal(self, state):
        """Whether the given state is the terminal state"""
        return state in self.terminals

    def _to_square(self, state):
        """numerical state to 2D state"""
        return state//self.ncol, state%self.ncol
    
    def _to_state(self, row, col):
        """2D state to numerical state"""
        return row * self.ncol + col

    def _get_move(self, action):
        """Get the 2D Grid actual move | the action taken"""
        return self.ACTION_MAP[action.lower()]

    def _get_reward(self, state):
        """Get the reward | state"""
        if self._is_terminal(state):
            return 0
        else:
            row, col = self._to_square(state)
            if self.grid[row][col] == 1:  # blocks
                return -10
            else:
                return -1
    
    def _get_new_state(self, state, move):
        """Move from the given state"""
        row, col = self._to_square(state)
        _row = min(max(row + move[0], 0), self.nrow-1)
        _col = min(max(col + move[1], 0), self.ncol-1)
        new_state = self._to_state(_row, _col)
        return new_state
    
    def get_four_neighbors(self, state):
        """return a list of all four neighbors from all 4 direction"""
        return [
            self._get_new_state(state, self._get_move(move))
            for move in self.A
        ]
