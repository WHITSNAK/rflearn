"""
Module that hosts algorithm to do policy iteration
and value iteration using dynamic programming
- it requires full knowledge of the system
"""
import numpy as np
from .base import GPI


class PolicyIteration(GPI):
    """
    Policy iteration algorithm that dances between value and policy
    - this finds and returns the first optimal policy π*,
        but does not gaurentee a optimal value function v(s)
    - it does iterative indefinite sweeps of all states until convergence
        while in evaluation before improve the existing policy π

    int parameter
    -------------
    env: the enviornment
    value: [1 x S], expected value for all states in a row vector
    policy: [S x A], stochastic policy for all states map to action

    fit parameter
    -------------
    gamma: float, the reward discount rate
    theta: float, the convergence threshold
    """
    def fit(self, gamma, theta=0.001):
        """Fit the model with the env other policy"""
        self.gamma = gamma
        self.theta = theta
        # ending rouding error handling
        self.sig_digits = str(round(1/theta)).count('0') - 1
    
    def transform(self):
        """Find the optimal policy"""
        policy_stable = False
        
        while not policy_stable:
            self.evaluate_policy()
            policy_stable = self.improve_policy()

    def update_value(self, state):
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
        q_sa = self.get_qs(state)
        new_v = pi_s @ q_sa
        self.value[state] = new_v
        return new_v

    def get_qs(self, state):
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
                new_v = self.update_value(state)
                delta = max(delta, abs(val - new_v))
    
    def improve_policy(self):
        """
        Improve the existing policy by greedifying in respect to the
            current value estimates v_k(s)
        If π(a|s) does not change means we converaged to the optimal policy π*
        """
        policy_stable = True
        for state in self.env.S:
            old_π = self.policy[state]
            qs = np.round(self.get_qs(state), self.sig_digits)
            new_π = self.policy.greedify(state, qs)
            
            if not np.array_equal(new_π, old_π):
                policy_stable = False
                
        return policy_stable



class ValueIteration(PolicyIteration):
    """
    Value iteration algorithm that truncates the full policy evaluation sweeps
    - each step it update (one complete sweep) of the v_k(s) in respect to argmax of a ∈ A
    - it coverages to the optimal value function v*(s)
        and use this to generate the optimal policy π*

    int parameter
    -------------
    env: the enviornment
    value: [1 x S], expected value for all states in a row vector
    policy: [S x A], stochastic policy for all states map to action

    fit parameter
    -------------
    gamma: float, the reward discount rate
    theta: float, the convergence threshold
    """
    def update_value(self, state):
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
        q_sa = self.get_qs(state)
        new_val = np.max(q_sa)
        self.value[state] = new_val
        return new_val


    def evaluate_policy(self):
        """Evaluate the existing policy with only one sweep"""
        for state in self.env.S:
            _ = self.update_value(state)
