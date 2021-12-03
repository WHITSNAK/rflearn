"""
Episode/Experience Abstraction
"""
from collections import namedtuple


EpisodeStep = namedtuple('EpisodeStep', ['s0','a0','r1','s1','is_terminal'])


class Episode:
    """
    Simple episode tracking class
    that tracks all steps left to right chronically
    """
    def __init__(self):
        self.steps = []
    
    def __len__(self):
        return len(self.steps)
    
    def __iter__(self):
        for step in self.steps:
            yield step
    
    def __getitem__(self, idx):
        return self.steps[idx]
        
    def __repr__(self):
        main_str = f'Episode<n:{self.nsteps}>'
        for idx, step in enumerate(self):
            main_str += '\n'+ ' '*4 + f'{idx}: ' + str(step)
        return main_str
    
    @property
    def nsteps(self):
        return len(self)

    def add_step(self, s0, a0, r1, s1, is_terminal):
        """record a step forward"""
        step = EpisodeStep(s0, a0, r1, s1, is_terminal)
        self.steps.append(step)

    def get_total_rewards(self):
        """calculates total rewards of the episode"""
        sums = 0
        for step in self:
            sums += step.r1
        return sums
    
    def get_avg_rewards(self):
        """calculate the average rewards of the episode"""
        return self.get_total_rewards() / self.nsteps
