"""
Module that hosts algorithm to do policy iteration
using monte carlo based method, model-free
"""
import numpy as np
from tqdm import tqdm
from itertools import product
from .base import GPI
from .episode import Episode, EpisodeStep



class MCEpsilonSoft(GPI):
    """
    Monte Carlo Policy Interation Algorithm
    - with ϵ-soft policy support
    - Monte Carlo only works on episodic tasks
    - uses first visit policy improvement

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
    """
    def fit(self, gamma=1, alpha=None):
        """Setting the algorithm"""
        self.gamma = gamma
        self.alpha = alpha

        # list used for targeted state-action policy improvment
        self.last_updated_s = set()

        # state-action pair seen counts for step size inferring
        if self.alpha is None:
            self.sa_counts = {k:0 for k in product(self.env.S, self.env.A)}
            
        self.hist = []

    def transform(self, iter=1000, kbatch=30, max_steps=None, pbar_leave=True):
        """
        Obtain the optimal policy

        parameter
        ---------
        kbatch: int, the number of episode in a batch to process in each policy evaluation
        max_steps: int (default None), max steps on taking a episode for early stopping if too long
        pbar_leave: bool, wheter to leave the progress bar on display
        """
        for _ in tqdm(range(iter), leave=pbar_leave):
            self.evaluate_policy(kbatch, max_steps)
            self.improve_policy()

    def evaluate_policy(self, kbatch=30, max_steps=None):
        """Update q(a|s) with newly generated MC episode"""
        # construct returns for each (s,a) pair
        epso_lst = self.get_episodes(n=kbatch, max_steps=max_steps)
        qs = {}
        for epso in epso_lst:
            ret = 0
            for step in epso[::-1]:  # start from the terminal step
                sa = step.s0, step.a0  # tuple pair
                ret = self.gamma * ret + step.r1
                if sa not in qs:
                    qs[sa] = [ret]
                else:
                    qs[sa].append(ret)
        

        # update q(a|s) with given new information
        # each (s,a) only updates once within a batch
        for sa, rets in qs.items():
            # step size α
            if not self.alpha:
                step_size = self.sa_counts[sa]
                step_size += 1
                self.sa_counts[sa] = step_size
                α = 1 / step_size
            else:
                α = self.alpha
            
            # calcualte new action-value
            s0, a0 = sa
            ret = np.mean(rets)
            q = self.qvalue.get_q(s0, a0)  # specific action value
            new_q = q + α * (ret - q)
            self.qvalue.set_q(s0, a0, new_q)

            # update seem states
            self.last_updated_s.add(s0)
        
        self.hist.append(epso_lst)
    
    def improve_policy(self):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates argmax[a] q(a|s)
            with ϵ-soft policy to ensure exploration in Monte Carlo
        """
        for state in self.last_updated_s:
            self.policy.greedify(state, self.qvalue[state])
        
        self.last_updated_s.clear()

    def get_episodes(self, n=1, max_steps=None):
        """Generate n episodes of data through Monte Carlo"""
        epso_lst = []
        for _ in range(n):
            epso = Episode()
            s0 = self.env.start()
            i = 0
            while True:
                a = np.random.choice(self.env.A, p=self.policy[s0])
                s1, r = self.env.step(a)
                is_t = self.env.is_terminal()
                epso.add_step(s0, a, r, s1, is_t)
                s0 = s1

                # fully stop or early stop
                i += 1
                if is_t or (max_steps is not None and i >= max_steps):
                    break

            epso_lst.append(epso)
        return epso_lst
