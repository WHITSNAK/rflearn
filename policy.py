"""
Module that hosts algorithm to do policy iteration
and value iteration
"""

import numpy as np
from copy import deepcopy


class PolicyIteration:
    """
    Policy Iteration Algorithm

    init parameter
    --------------
    theta: float, the convergence threshold

    fit parameter
    -------------
    env: the enviornment
    value: [1 x S], expected value for all states in a row vector
    policy: [S x A], stochastic policy for all states map to action
    gamma: float, the reward discount rate
    """
    def __init__(self, theta=0.001):
        self.theta = theta
        self.sig_digits = str(round(1/theta)).count('0') - 1
    

    def fit(self, env, value, policy, gamma):
        """Fit the model with the env other policy"""
        self.env = env
        self.value = deepcopy(value)
        self.policy = deepcopy(policy)
        self.gamma = gamma
    

    def evaluate_policy(self):
        """
        Policy evaluation using bellman equation updates
            to get the expected value for the current policy
        """
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            for state in self.env.S:
                val = self.value[state]
                pi_s = self.policy[state]
                q_sa = self.get_all_qvalues(state)
                new_v = pi_s @ q_sa
                self.value[state] = new_v
                delta = max(delta, abs(val - new_v))
    

    def improve_policy(self):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates v_k(s)
        If π(a|s) does not change means we converaged to the optimal policy π*
        """
        policy_stable = True
        for state in self.env.S:
            old = self.policy[state].copy()
            self.q_greedify_policy(state)
            
            if not np.array_equal(self.policy[state], old):
                policy_stable = False
                
        return policy_stable


    def policy_iteration(self):
        """
        Policy iteration algorithm that dances between value and policy
        - this finds and returns the first optimal policy π*,
          but does not gaurentee a optimal value function v(s)
        - it does iterative indefinite sweeps of all states until convergence
          while in evaluation before improve the existing policy π
        
        """
        policy_stable = False
        
        while not policy_stable:
            self.evaluate_policy()
            policy_stable = self.improve_policy()
    

    def value_iteration(self):
        """
        Value iteration algorithm that truncates the full policy evaluation sweeps
        - each step it update the v_k(s) in respect to argmax of a ∈ A
        - it coverages to the optimal value function v*(s)
          and use this to generate the optimal policy π*
        """
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            for state in self.env.S:
                val = self.value[state]
                q_sa = self.get_all_qvalues(state)
                new_val = np.max(q_sa)
                self.value[state] = new_val
                delta = max(delta, abs(val - new_val))

        _ = self.improve_policy()


    def get_qvalue(self, p, r):
        """Calculates the q(s, a) value"""
        # it sums over all immediate rewards and discounted v(s')
        # then weighted by the jointed transition p(s',r|s,a) probability
        return (np.c_[r, self.gamma*self.value].sum(1) * p).sum()


    def get_all_qvalues(self, state):
        """Get all sucessor q(s',a') for the give state from the updates diagram"""
        q_sa = []
        for action in self.env.A:
            # trans matrix (states x 2) where col = [reward, p(s',r|s,a)]
            trans = self.env.transitions(state, action)
            action_value = self.get_qvalue(p=trans[:,1], r=trans[:,0])
            q_sa.append(action_value)
        return q_sa
            

    def q_greedify_policy(self, state):
        """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
        # here ignores the floating error belows theta threshold
        q_sa = np.round(self.get_all_qvalues(state), self.sig_digits)
        max_q = np.max(q_sa)
        new_pi_s = [1 if q==max_q else 0 for q in q_sa]
        self.policy[state] = new_pi_s
