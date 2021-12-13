from numpy import ones, round, array, max, sum
from numpy.random import choice


class TabularPolicy:
    """
    Finite State & action policy

    parameter
    ---------
    states: list, [s0, s1, ...] all avaliable states s ∈ 𝑆
    actions: list, [a0, a1, ...] all avaliable actions a ∈ 𝐴
    epsilon: float (default 0), ϵ-soft parameter that takes random action
    policy: dict of {list, array} (default None) {|𝑆|: |𝐴|}
        if not provided, will use uniform random policy
    """
    def __init__(self, states, actions, epsilon=0, policy=None):
        self.states = states
        self.actions = actions
        self.epsilon = epsilon
        self.shape = len(self.states), len(self.actions)
        self._a_mapper = {a:i for i, a in enumerate(self.actions)}

        # setting the policy lookup table π(a|s)
        if policy is None:
            # uniform random across actions | state
            self.policy = {
                s: ones(len(self.actions)) / len(self.actions)
                for s in self.states
            }
        else:
            self.policy = policy
        
        # ensures invariants
        assert set(self.policy) == set(self.states)  # one to one mapping
        for _, pi in self.policy.items():
            self.__policy_invariant(pi)
    
    def __repr__(self):
        return f"Policy [{self.shape[0]} x {self.shape[1]} | ϵ={self.epsilon}]"
    
    def __policy_invariant(self, pi):
        assert len(pi) == len(self.actions)  # maps all actions
        assert round(sum(pi)) == 1  # proper distribution & floating precision
    
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
    
    def to_numpy(self):
        """Convert the policy table to numpy array as |𝑆| x |𝐴|"""
        lst = []
        for state in self.states:
            lst.append(self[state])
        return array(lst)
    
    def greedify(self, state, qvalues):
        """
        Greedify the policy π(a|s), in place, in respect to action-values/qvalues
        
        parameter
        ---------
        state: s ∈ 𝑆, state is the state to look up the policy ....
        qvalues: array q(s, a) ∀ a ∈ 𝐴, action-values array that is the same size of 𝐴

        return
        -------
        new π(a|s)
        """
        qs = array(qvalues)
        ϵ = self.epsilon
        nA = len(self.actions)

        # caluclates ϵ-soft policy if eps > 0
        max_q = max(qs)
        max_flag = (qs == max_q)
        nmax = max_flag.sum()  # adjustments for multiple best actions
        new_π = max_flag * (1 - ϵ + (nmax*ϵ)/nA) + ~max_flag * (nmax*ϵ)/nA
        new_π = new_π / sum(new_π)  # normalize ∑π(a|s) = 1, ensures float type

        # update
        self[state] = new_π
        return new_π

    def get_action(self, state):
        """Pick the action based on the policy and given observation/state"""
        return choice(self.actions, p=self[state])
    
    def is_trace(self, exp, behavior):
        """
        Caculates the trace of importance sampling ratios of a given experience
        
        parameter
        ---------
        exp: Episode, a seqeunce of experiences
        behavior: Policy, the behavior policy that generated the experience

        return
        ------
        list of tuple, [(target π(a|s), behavior b(a|s), isratio ρ), ...]
            that is the same size and order of the sequence of exp
        """
        data = []
        for step in exp:
            s0, a0, _, _, _ = step
            tp = self[s0][self.get_aidx(a0)]
            bp = behavior[s0][behavior.get_aidx(a0)]
            rho = tp / bp
            data.append((tp, bp, rho))
        return data
