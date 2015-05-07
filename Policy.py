#!/usr/bin/env python
import random
STATE = 0
VALUE = 1
class Policy:
    def __init__(self, MDP, name):
        self.MDP_ = MDP
        self.name_ = name

    def choose_action(self, horizon):
        raise Exception("Not an implemented policy")

    def get_name(self):
        return self.name_

    def bellman_backup(self, cur_state, epsilon, discount = 1.0):
        backup_table = []
        action_table = []
        error = 10000
        t = 0
        #arbitrary upper bound
        while error > epsilon and t < 1000: 
            backup_table.append([])
            for state in self.MDP_.get_states():
                if t == 0: #i technically dont need to do this here but whatever
                    backup_table[t].append(self.MDP_.get_state_reward(state))
                    action_table.append(None)
                else:
                    max_action = None
                    max_id = -1
                    for action in self.MDP_.get_legal_actions():
                        sum_action = 0
                        for n_state in self.MDP_.get_states():
                            p_transition = self.MDP_.get_p_transition(state, action, n_state)
                            backup_value = discount**t * backup_table[t-1][n_state]
                            v_transition = p_transition * backup_value
                            sum_action += v_transition
                        sum_action += self.MDP_.get_state_reward(state) 
                        if sum_action > max_action:
                            max_action = sum_action
                            max_id = action

                    backup_table[t].append(max_action)
                    action_table[state] = max_id
                    error = abs(sum(backup_table[t]) - sum(backup_table[t-1]))
            t += 1
        return backup_table, action_table


class RandomPolicy(Policy):
    def __init__(self, MDP, name):
        Policy.__init__(self, MDP, name)

    def choose_action(self, horizon):
        to_take = random.randint(0, self.MDP_.get_num_actions() - 1)
        return to_take

"""
No idea what i'm doing with this for resusability
"""
class ValueIterationPolicy(Policy):
    def __init__(self, MDP, name, epsilon, discount):
        Policy.__init__(self, MDP, name)    
        self.epsilon = epsilon
        self.backups, self.actions = self.bellman_backup(self.MDP_.get_state(), epsilon, discount)
    """
    Now given our start state and horizon we have to construct our policy
    """
    def choose_action(self, steps_to_go):
        return self.actions[self.MDP_.get_time()][self.MDP_.get_state()]
    

    """
    New output should be two size-n vectors, since it is an infinite policy of which state youre in and what action you should take.

    value
    value
    value

    action
    action
    action
        
    """
    def display_value_f(self): 
        transpose = zip(*self.backups[::-1])
        for action in range(len(transpose)):
            print transpose[action][0]
        return

    def display_policy(self):
        for action in self.actions:
            print action
