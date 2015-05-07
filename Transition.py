#!/usr/bin/env python
import random

class Transition:
    transition_probs_ = None
    #Action, State, => State 
    def __init__(self, transition_probs):
        self.transition_probs_ = transition_probs
    
    """
    ACCEPTS index of the action to take

    ASSUMES p's sum to one

    RETURNS in the index of the new state
    """
    def choose_transition(self, action, state):
        p_sum = 0
        for i in range(len(self.transition_probs_[state][action])):
            p_sum += self.transition_probs_[state][action][i]
            if random.random() < p_sum:
                return i #index state we chose
    
    def __str__(self):
        result = ""
        nActions = range(len(self.transition_probs_[0]))
        nStates = range(len(self.transition_probs_))
        for i in nActions:
            for j in nStates:
                for k in nStates:
                    result += str(self.transition_probs_[j][i][k]) + " "
                result += "\n"
            result += "\n"
        return result.strip()
    
    def get_p_transition(self, state, action, r_state):
        return self.transition_probs_[action][state][r_state]
