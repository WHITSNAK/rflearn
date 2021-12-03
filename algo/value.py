from numpy import zeros, array, dot, max


class TabularQValue:
    """
    Finite State & action Action Value Q Function/Table

    parameter
    ---------
    states: list, [s0, s1, ...] all avaliable states s ∈ 𝑆
    actions: list, [a0, a1, ...] all avaliable actions a ∈ 𝐴
    qvalue: dict of {list, array} (default None) {|𝑆|: |𝐴|}
        if not provided, will use zero action value table
    """
    def __init__(self, states, actions, qvalue=None):
        self.states = states
        self.actions = actions
        self.shape = len(self.states), len(self.actions)
        self._a_mapper = {a:i for i, a in enumerate(self.actions)}

        # setting the action value lookup table q(.|s, a)
        if qvalue is None:
            # uniform random across actions | state
            self.qvalue = {
                s: zeros(len(self.actions))
                for s in self.states
            }
        else:
            self.qvalue = qvalue
        
        # ensures invariants
        assert set(self.qvalue) == set(self.states)  # one to one mapping
        for _, qs in self.qvalue.items():
            self.__qvalue_invariant(qs)
    
    def __repr__(self):
        return f"QValue [{self.shape[0]} x {self.shape[1]}]"
    
    def __qvalue_invariant(self, qs):
        assert len(qs) == len(self.actions)  # maps all actions
    
    def __getitem__(self, state):
        """getter interface"""
        return self.qvalue[state]

    def __setitem__(self, state, new_qs):
        """setter interface"""
        self.__qvalue_invariant(new_qs)
        self.qvalue[state] = new_qs

    def get_aidx(self, action):
        """get action index"""
        return self._a_mapper[action]
    
    def to_numpy(self):
        """Convert the policy table to numpy array as |𝑆| x |𝐴|"""
        lst = []
        for state in self.states:
            lst.append(self[state])
        return array(lst)
    
    def get_q(self, state, action):
        """Specific q value getter"""
        return self[state][self.get_aidx(action)]
    
    def set_q(self, state, action, new_q):
        """Specific q value setter"""
        self[state][self.get_aidx(action)] = new_q
    
    def get_value(self, state, pi):
        """
        Get expected value of a state, aggrates over all action

        parameter
        ---------
        state: the state to lookup
        pi: policy distribution of given state
        """
        qs = self[state]
        return dot(qs, pi)
    
    def get_maxq(self, state):
        """Get maximun action value of a state"""
        return max(self[state])
    
    def get_all_values(self, policy):
        """Get the value table/array out of policy and action values [1 x |𝑆|]"""
        lst = []
        for state in self.states:
            lst.append(self.get_value(state, policy[state]))
        return array(lst)
    