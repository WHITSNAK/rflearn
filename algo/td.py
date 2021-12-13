import numpy as np
from tqdm import tqdm
from .base import GPI
from .episode import EpisodeStep
from itertools import product


class TDIteration(GPI):
    """
    Sarsa Policy Interation Algorithm in TD Family
    - Works on both continuous and episodic tasks

    int parameter
    -------------
    env: the enviornment
    qvalue: [S x A], init expected value for all state-action pair in a matrix
    policy: [S x A], stochastic policy for all states map to action

    fit parameter
    -------------
    gamma: float, the reward discount rate
    alpha: float (default None), the learning rate
        - None, would use same average method instead 
    kind: str, {'simple', 'expected', 'maxq'}
        - this determents on how to caculate the TD-Target
        - simple -> Sarsa
        - expected -> ExpectedSarsa
        - maxq -> Q Learning
    """
    def fit(self, gamma=1, alpha=None, kind='expected'):
        """Setting the algorithm"""
        self.gamma = gamma
        self.alpha = alpha
        self.kind = kind
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
        self.hist.append({
            'td-error': error,
            'step': step,
        })
        return s0
    
    def get_target(self, r, s1):
        """Calcuates the TD-Target"""
        if self.kind == 'simple':
            a1 = self.policy.get_action(s1)
            target = r + self.gamma * self.qvalue.get_q(s1, a1)

        elif self.kind == 'expected':
            pi = self.policy[s1]
            val = self.qvalue.get_value(s1, pi)
            target = r + self.gamma * val

        elif self.kind == 'maxq':
            max_q = self.qvalue.get_maxq(s1)
            target = r + self.gamma * max_q
        
        return target


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
