"""
Module that hosts algorithm to do policy iteration
and value iteration
"""

from abc import abstractmethod
import numpy as np
from copy import deepcopy


class GPI:
    """
    General Policy Interation Framework
    - it does not impose resitriction on how to obtain q(s,a) or v(s)
    - it can be either on-policy or off-policy
    - with model or without model

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
    
    @abstractmethod
    def transform(self):
        """Transform the old policy to the optimal policy"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_policy(self):
        """Ways of updating value function"""
        raise NotImplementedError()

    @abstractmethod
    def get_qvalues(self, state):
        """Used for value updating from the update diagram"""
        raise NotImplementedError()

    def update_value_v(self, state):
        """
        Policy evaluation using bellman state-value equation updates
            to get the expected value v(s) for the current policy

        parameter
        ---------
        state: the target state to do the update

        return
        ------
        new value of the state v_k+1(s)
        """
        pi_s = self.policy[state]
        q_sa = self.get_qvalues(state)
        new_v = pi_s @ q_sa
        self.value[state] = new_v
        return new_v
    

    def update_value_q(self, state):
        """
        Policy evaluation using bellman action-value equation updates
            to get the expected value v(s) for the current policy

        parameter
        ---------
        state: the target state to do the update

        return
        ------
        new value of the state v_k+1(s)
        """
        q_sa = self.get_qvalues(state)
        new_val = np.max(q_sa)
        self.value[state] = new_val
        return new_val
      

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


    def q_greedify_policy(self, state):
        """
        Mutate the policy to be greedy with respect to the q-values
            induced by the existing value function
        """
        # here ignores the floating error belows theta threshold
        q_sa = np.round(self.get_qvalues(state), self.sig_digits)
        max_q = np.max(q_sa)
        new_pi_s = [1 if q==max_q else 0 for q in q_sa]
        self.policy[state] = new_pi_s



class PolicyIteration(GPI):
    """
    Policy iteration algorithm that dances between value and policy
    - this finds and returns the first optimal policy π*,
        but does not gaurentee a optimal value function v(s)
    - it does iterative indefinite sweeps of all states until convergence
        while in evaluation before improve the existing policy π

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
    def get_qvalues(self, state):
        """
        Get all sucessor q(s',a') for the give state from the updates diagram

        return
        ------
        [q(s, a_0), q(s, a_1), ..., q(s, a_n)]
        """
        q_sa = []
        for action in self.env.A:
            # trans matrix (states x 2) where col = [reward, p(s',r|s,a)]
            trans = self.env.transitions(state, action)

            # it sums over all immediate rewards and discounted v(s')
            # then weighted by the jointed transition p(s',r|s,a) probability
            action_value = (
                np.c_[trans[:,0], self.gamma*self.value].sum(1)
                * trans[:,1]
            ).sum()

            q_sa.append(action_value)
        return q_sa


    def evaluate_policy(self):
        """Evaluate the existing value until converge with indefinte number of sweeps"""
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            for state in self.env.S:
                val = self.value[state]
                new_v = self.update_value_v(state)
                delta = max(delta, abs(val - new_v))
    

    def transform(self):
        """Find the optimal policy"""
        policy_stable = False
        
        while not policy_stable:
            self.evaluate_policy()
            policy_stable = self.improve_policy()
    

class ValueIteration(PolicyIteration):
    """
    Value iteration algorithm that truncates the full policy evaluation sweeps
    - each step it update (one complete sweep) of the v_k(s) in respect to argmax of a ∈ A
    - it coverages to the optimal value function v*(s)
        and use this to generate the optimal policy π*

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
    def evaluate_policy(self):
        """Evaluate the existing policy with only one sweep"""
        for state in self.env.S:
            _ = self.update_value_q(state)
