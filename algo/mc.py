"""
Module that hosts algorithm to do policy iteration
using monte carlo based method, model-free
"""
import numpy as np
from tqdm import tqdm
from itertools import product
from collections import namedtuple
from .base import GPI


EpisodeStep = namedtuple('EpisodeStep', ['s0','a0','r1','s1','is_terminal'])

class Episode:
    """Simple episode tracking class"""
    def __init__(self):
        self.nsteps = 0
        self.steps = []
    
    def __iter__(self):
        for step in self.steps:
            yield step
    
    def __getitem__(self, idx):
        return self.steps[idx]
        
    def __repr__(self):
        main_str = f'Episode<n:{self.nsteps}>'
        for step in self:
            main_str += '\n'+' '*4 + str(step)
        return main_str

    def add_step(self, s0, a0, r1, s1, is_terminal):
        self.nsteps += 1
        step = EpisodeStep(s0, a0, r1, s1, is_terminal)
        self.steps.append(step)


class MCEpsilonSoft(GPI):
    """
    Monte Carlo Policy Interation Algorithm
    - with ϵ-soft policy support
    - Monte Carlo only works on episodic tasks

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
    lam: float, the learning rate
    kbath: int, the number of episode in a batch to process in each policy evaluation
    """
    def __init__(self, env, value, policy, qvalue=None):
        super().__init__(env, value, policy)
        # the action-value function/table
        self.qvalue = \
            {k:0 for k in product(env.S, env.A)} \
            if qvalue is None else qvalue
        
    def fit(self, gamma, epsilon=0.01, lam=0.2, kbatch=30, max_steps=100):
        """Setting the algorithm"""
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.kbatch = kbatch
        self.max_steps = max_steps

        # list used for targeted state-action policy improvment
        self.last_updated_s = []
        # state-action pair seen counts for step size inferring
        self.sa_counts = {k:0 for k in product(self.env.S, self.env.A)}
        self.hist = {'avg_r': []}


    def transform(self, iter=1000):
        """Obtain the optimal policy"""
        for _ in tqdm(range(iter)):
            self.last_updated_s = []  # reset updated stated
            self.evaluate_policy()
            self.improve_policy()

    def evaluate_policy(self):
        """Update q(a|s) with newly generated MC episode"""
        # construct returns for each (s,a) pair
        epso_lst = self.get_episodes(n=self.kbatch)
        avg_r = 0
        qs = {}
        for epso in epso_lst:
            avg_r += sum(step.r1 for step in epso)
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
            self.last_updated_s.append(sa[0])
        
        self.hist['avg_r'].append(avg_r)
    
    def improve_policy(self):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates argmax[a] q(a|s)
            with ϵ-soft policy to ensure exploration in Monte Carlo
        """
        for state in self.last_updated_s:
            qs = np.array(self.get_qs(state))
            max_q = np.max(qs)

            # epsilon soft policy
            nA = len(self.env.A)
            ϵ = self.epsilon
            max_flag = (qs==max_q)

            # in respect to bellman optimal equation q*
            new_π = max_flag * (1 - ϵ + ϵ/nA) + ~max_flag * ϵ/nA
            new_π = new_π / np.sum(new_π).sum()  # normalize ∑π(a|s) = 1
            self.policy[state] = new_π

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
                if is_t or i >= self.max_steps:
                    break

            epso_lst.append(epso)
        return epso_lst

    def get_qs(self, state):
        """
        MC is model-free method
        does not need the complete transition env probability
        """
        selected = []
        for action in self.env.A:
            selected.append(self.qvalue[(state, action)])
        return selected
