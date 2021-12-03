import numpy as np


class TabularPolicy:
    """
    Finite State & action policy

    parameter
    ---------
    states: list, [s0, s1, ...] all avaliable states s ∈ 𝑆
    actions: list, [a0, a1, ...] all avaliable actions a ∈ 𝐴
    policy: dict of list (default None) {|𝑆|: |𝐴|}
        if not provided, will use uniform random policy
    """
    def __init__(self, states, actions, policy=None):
        self.states = states
        self.actions = actions
        self.shape = len(self.states), len(self.actions)
        self._a_mapper = {a:i for i, a in enumerate(self.actions)}

        # setting the policy lookup table π(a|s)
        if policy is None:
            # uniform random across actions | state
            self.policy = {
                s: np.ones(len(self.actions)) / len(self.actions)
                for s in self.states
            }
        else:
            self.policy = policy
        
        # ensures invariants
        assert set(policy) == set(self.states)  # one to one mapping
        for _, pi in self.policy.items():
            self.__policy_invariant(pi)
    
    def __repr__(self):
        shape = self.shape
        return f"Policy [{shape[0]} x {shape[1]}]"
    
    def __policy_invariant(self, pi):
        assert len(pi) == len(self.actions)  # maps all actions
        assert sum(pi) == 1  # proper distribution
    
    def __getitem__(self, state):
        """getter interface"""
        return self.policy[state]

    def __setitem__(self, state, new_pi):
        """setter interface"""
        self.__policy_invariant(new_pi)
        self.policy[state] = new_pi

    def get_aidx(self, action):
        """get action index"""
        return self._a_mapper[action]
    
