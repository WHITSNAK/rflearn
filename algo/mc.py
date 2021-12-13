"""
Module that hosts algorithm to do policy iteration
using monte carlo based method, model-free
"""
import numpy as np
from tqdm import tqdm
from itertools import product
from .base import GPI
from .episode import Episode



class MCIteration(GPI):
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

    Algorithm NOTE
    -------------
    Monte Carlo mimics brutal force approach in estimating the value function
    it is really slow and error prone when getting bad samples.
    And it will happen due to the inheritant random nature.
    
    For each policy π, it searchs through a combination of [T ^ (|S| x |A|)] possiabilities.
    It is really large, and samples are sparse.
    Therefore, always to use leinent batch size while using MC.
    """
    def fit(self, gamma=1, alpha=None):
        """Setting the algorithm"""
        self.gamma = gamma
        self.alpha = alpha
        self.hist = []

        # state-action pair seen counts for step size inferring
        if self.alpha is None:
            self.sa_counts = {k:0 for k in product(self.env.S, self.env.A)}

    def transform(self, iter=1000, kbatch=30, pbar_leave=True):
        """
        Obtain the optimal policy

        parameter
        ---------
        kbatch: int, the number of episode in a batch to process in each policy evaluation
        pbar_leave: bool, wheter to leave the progress bar on display
        """
        for _ in tqdm(range(iter), leave=pbar_leave):
            eps = self.get_episodes(kbatch)
            updated = self.evaluate_policy(eps)
            self.improve_policy(updated)
            self.hist.append(eps)


    def evaluate_policy(self, eps):
        """Update q(a|s) with newly generated MC episode"""
        # construct returns for each (s,a) pair
        qs = {}
        for ep in eps:
            ret = 0
            for step in ep[::-1]:  # start from the terminal step
                sa = step.s0, step.a0  # tuple pair
                ret = self.gamma * ret + step.r1
                if sa not in qs:
                    qs[sa] = [ret]
                else:
                    qs[sa].append(ret)
        

        # update q(a|s) with given new information
        # each (s,a) only updates once within a batch
        updated = set()
        for sa, rets in qs.items():
            # step size α
            if not self.alpha:
                self.sa_counts[sa] += 1
                α = 1 / self.sa_counts[sa]
            else:
                α = self.alpha
            
            # calcualte new action-value
            s0, a0 = sa
            ret = np.mean(rets)
            q = self.qvalue.get_q(s0, a0)  # specific action value
            new_q = q + α * (ret - q)
            self.qvalue.set_q(s0, a0, new_q)

            # update seem states
            updated.add(s0)
        return updated
    
    def improve_policy(self, updated):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates argmax[a] q(a|s)
            with ϵ-soft policy to ensure exploration in Monte Carlo
        """
        for state in updated:
            self.policy.greedify(state, self.qvalue[state])
        
    def get_episodes(self, n=1):
        """Generate n episodes of data through Monte Carlo"""
        epso_lst = []
        for _ in range(n):
            epso = Episode()
            s0 = self.env.start()
            while True:
                a = self.policy.get_action(s0)
                s1, r = self.env.step(a)
                is_t = self.env.is_terminal()
                epso.add_step(s0, a, r, s1, is_t)
                s0 = s1

                if is_t: break
            epso_lst.append(epso)
        return epso_lst
