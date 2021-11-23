"""
Module for Utility functions
"""
import numpy as np


def argmax(values):
    """
    Takes in a list of values and returns the index of the item 
    with the highest value. Breaks ties randomly.

    return
    ------
    int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []
    
    for i in range(len(values)):
        q = values[i]
        if q > top_value:
            top_value = q
            ties = [i]  # reset
        elif q == top_value:
            ties.append(i)  # multiple ties
        
    return np.random.choice(ties)


def zero_values(env):
    """Returns value function [1xS] with zero values"""
    return np.zeros(shape=len(env.S))


def random_policy(env):
    """return policy [SxA] with uniform random probability π(a|S=s) of each action"""
    return np.ones(shape=(len(env.S), len(env.A))) / len(env.A)
