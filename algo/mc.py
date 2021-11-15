"""
Module that hosts algorithm to do policy iteration
using monte carlo based method, model-free
"""
import numpy as np
from tqdm import tqdm
from itertools import product
from .base import GPI


class MCIteration(GPI):
    """
    Monte Carlo Policy Interation Algorithm
    - with ϵ-soft policy support
    """
    def __init__(self, env, value, policy, qvalue=None):
        super().__init__(env, value, policy)
        # the action-value function/table
        self.qvalue = \
            {k:0 for k in product(env.S, env.A)} \
            if qvalue is None else qvalue

    def fit(self, gamma, epsilon=0.01):
        """Setting the algorithm"""
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_updated_s = []
        # state-action pair seen counts for step size inferring
        self.sa_counts = {k:0 for k in product(self.env.S, self.env.A)}
        
    def transform(self, iter=1000):
        """Obtain the optimal policy"""
        for _ in tqdm(range(iter)):
            self.evaluate_policy()
            self.improve_policy()
            self.last_updated_s = []  # reset updated stated

    def evaluate_policy(self):
        """Update q(a|s) with newly generated MC episode"""
        ret = 0
        episode = self.get_episode()
        for step in range(episode['steps']-1, -1, -1):
            state = episode['state'][step]
            action = episode['action'][step]
            sa = state, action  # tuple pair
            ret = self.gamma * ret + episode['reward'][step]

            # update q(a|s) with given new information
            step_size = self.sa_counts[sa]
            step_size += 1

            old_ret = self.qvalue[sa]
            self.qvalue[sa] += 1/step_size * (ret - old_ret)
            self.sa_counts[sa] = step_size

            # update seem states
            self.last_updated_s.append(state)
    
    def improve_policy(self):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates argmax[a] q(a|s)
            with ϵ-soft policy to ensure exploration in Monte Carlo
        """
        for state in self.last_updated_s:
            # here ignores the floating error belows theta threshold
            qs = self.get_qs(state)
            max_q = np.max(qs)

            # epsilon soft policy
            nA = len(self.env.A)
            ϵ = self.epsilon
            new_π = []
            for q in qs:
                if q == max_q:
                    new_π.append(1 - ϵ + ϵ/nA)
                else:
                    new_π.append(ϵ/nA)

            # normalize to ∑π = 1
            new_π = np.divide(new_π, np.sum(new_π))
            self.policy[state] = new_π

    def get_episode(self):
        """Generate a episode of data through Monte Carlo"""
        trace = {'steps': 0, 'state': [], 'action': [], 'reward': []}
        s0 = self.env.start()
        while not self.env.is_terminal():
            a = np.random.choice(self.env.A, p=self.policy[s0])
            s1, r = self.env.step(a)
            trace['state'].append(s0)
            trace['action'].append(a)
            trace['reward'].append(r)
            trace['steps'] += 1
            s0 = s1
        return trace


    def get_qs(self, state):
        """
        MC is model-free method
        does not need the complete transition env probability
        """
        selected = []
        for action in self.env.A:
            selected.append(self.qvalue[(state, action)])
        return selected
