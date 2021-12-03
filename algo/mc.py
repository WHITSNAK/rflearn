"""
Module that hosts algorithm to do policy iteration
using monte carlo based method, model-free
"""
import numpy as np
from tqdm import tqdm
from itertools import product
from .base import GPI
from .episode import Episode



class MCEpsilonSoft(GPI):
    """
    Monte Carlo Policy Interation Algorithm
    - with ϵ-soft policy support
    - Monte Carlo only works on episodic tasks
    - uses first visit policy improvement

    int parameter
    -------------
    env: the enviornment
    value: [1 x S], init expected value for all states in a row vector
    policy: [S x A], stochastic policy for all states map to action
    qvalue: [S x A], init expected value for all state-action pair in a matrix

    fit parameter
    -------------
    gamma: float, the reward discount rate
    epsilon: float, the epsilon greedy soft rate
    lam: float (default None), the learning rate
        - None, would use same average method instead 
    kbath: int, the number of episode in a batch to process in each policy evaluation
    max_steps: int (default None), max steps on taking a episode for early stopping if too long
    """
    def __init__(self, env, value, policy, qvalue=None):
        super().__init__(env, value, policy)
        # the action-value function/table
        self.qvalue = \
            {k:0 for k in product(env.S, env.A)} \
            if qvalue is None else qvalue
        
    def fit(self, gamma=1, epsilon=0.01, lam=None, kbatch=30, max_steps=None):
        """Setting the algorithm"""
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.kbatch = kbatch
        self.max_steps = max_steps

        # list used for targeted state-action policy improvment
        self.last_updated_s = set()
        # state-action pair seen counts for step size inferring
        self.sa_counts = {k:0 for k in product(self.env.S, self.env.A)}
        self.hist = {'avg_r': []}

    def transform(self, iter=1000, pbar_leave=True):
        """Obtain the optimal policy"""
        for _ in tqdm(range(iter), leave=pbar_leave):
            avg_r = self.evaluate_policy()
            self.improve_policy()

    def evaluate_policy(self):
        """Update q(a|s) with newly generated MC episode"""
        # construct returns for each (s,a) pair
        epso_lst = self.get_episodes(n=self.kbatch)
        avg_r = 0  # mean total rewards
        qs = {}
        for epso in epso_lst:
            avg_r += epso.get_total_rewards()
            ret = 0
            for step in epso[::-1]:  # start from the terminal step
                sa = step.s0, step.a0  # tuple pair
                ret = self.gamma * ret + step.r1
                if sa not in qs:
                    qs[sa] = [ret]
                else:
                    qs[sa].append(ret)
        
        avg_r /= self.kbatch

        # each (s,a) only updates once within a batch
        for sa, rets in qs.items():
            # update q(a|s) with given new information
            if not self.lam:
                step_size = self.sa_counts[sa] + 1
                step_size += 1
                self.sa_counts[sa] = step_size
            else:
                step_size = 1/self.lam

            old_ret = self.qvalue[sa]
            ret = np.mean(rets)
            self.qvalue[sa] += 1/step_size * (ret - old_ret)

            # update seem states
            self.last_updated_s.add(sa[0])
        
        self.hist['avg_r'].append(avg_r)
        return avg_r
    
    def improve_policy(self):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates argmax[a] q(a|s)
            with ϵ-soft policy to ensure exploration in Monte Carlo
        """
        # epsilon soft policy
        nA = len(self.env.A)
        ϵ = self.epsilon
        for state in self.last_updated_s:
            qs = np.array(self.get_qs(state))
            max_q = np.max(qs)
            max_flag = (qs==max_q)

            # in respect to bellman optimal equation q*
            new_π = max_flag * (1 - ϵ + ϵ/nA) + ~max_flag * ϵ/nA
            new_π = new_π / np.sum(new_π)  # normalize ∑π(a|s) = 1
            self.policy[state] = new_π
        
        self.last_updated_s.clear()

    def get_episodes(self, n=1):
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
                if is_t or (self.max_steps is not None and i >= self.max_steps):
                    break

            epso_lst.append(epso)
        return epso_lst

    def get_qs(self, state):
        """
        MC is model-free method
        - does not need the complete transition env probability
        - it ensures ordering by a ∈ A in the list
        """
        selected = []
        for action in self.env.A:
            selected.append(self.qvalue[(state, action)])
        return selected
