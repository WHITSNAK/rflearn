import numpy as np
from tqdm import tqdm
from .base import GPI
from .episode import EpisodeStep
from itertools import product


class Sarsa(GPI):
    def fit(self, gamma=1, alpha=None):
        """Setting the algorithm"""
        self.gamma = gamma
        self.alpha = alpha
        self.hist = []

        # state-action pair seen counts for step size inferring
        if self.alpha is None:
            self.sa_counts = {k:0 for k in product(self.env.S, self.env.A)}

    def transform(self, iter=1000, pbar_leave=True):
        with tqdm(total=iter, leave=pbar_leave) as pbar:
            for step in self.get_steps(iter):
                state = self.evaluate_policy(step)
                self.improve_policy(state)
                pbar.update(1)
    
    def evaluate_policy(self, step):
        s0, a, r, s1, is_t = step

        # step size α
        if not self.alpha:
            self.sa_counts[(s0, a)] += 1
            α = 1 / self.sa_counts[(s0, a)]
        else:
            α = self.alpha

        q = self.qvalue.get_q(s0, a)
        target = self.get_target(r, s1)
        error = target - q
        new_q = q + α * error
        self.qvalue.set_q(s0, a, new_q)
        self.hist.append(error)
        return s0
    
    def get_target(self, r, s1):
        a1 = self.policy.get_action(s1)
        return r + self.gamma * self.qvalue.get_q(s1, a1)

    def improve_policy(self, state):
        self.policy.greedify(state, self.qvalue[state])
    
    def get_steps(self, steps=100):
        """Generate step data from ENV"""
        cnt = 0
        s0 = self.env.start()
        while cnt < steps:
            a = self.policy.get_action(s0)
            s1, r = self.env.step(a)
            is_t = self.env.is_terminal()
            step = EpisodeStep(s0, a, r, s1, is_t)
            yield step

            s0 = s1
            cnt += 1
            if is_t:
                s0 = self.env.start()



class ExpectedSarsa(Sarsa):
    def get_target(self, r, s1):
        pi = self.policy[s1]
        val = self.qvalue.get_value(s1, pi)
        return r + self.gamma * val



class QLearning(Sarsa):
    def get_target(self, r, s1):
        max_q = self.qvalue.get_maxq(s1)
        return r + self.gamma * max_q
