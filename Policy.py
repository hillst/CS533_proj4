#!/usr/bin/env python
import random
from math import exp
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
Pure model-free reinforcement learning agent that uses q-learning.
1. Start with initial Q-function (e.g. all zeros)
2. Take action from explore/exploit policy giving new state s’ (this is given by the BasePolicy)
(should converge to greedy policy, i.e. GLIE)
q(s,a) = q(s,a) + alpha * (R(s) + beta max (a') (Q(s', a') - Q(s, a))

    -NOTE betamax (a') Q(s', a') is a very rough measure based on our current model of the world.
    -beta is our discount term
    -alpha is our explore/exploit term
3. Perform TD update
Q(s,a) is current estimate of optimal Q-function.
4. Goto 2
"""
class QLearningPolicy(Policy):
    def __init__(self, MDP, t):
        Policy.__init__(self, MDP, "QLearningPolicy")
        self.t = float(t)
        self.q_values = [[0, 0] for i in range(len(MDP.get_states()))]

    def boltzmann_action(self):
        p_action_0 = exp(self.q_values[self.MDP_.get_state()][0] / self.t) / \
                        (exp(self.q_values[self.MDP_.get_state()][0] / self.t) +
                        exp(self.q_values[self.MDP_.get_state()][1] / self.t))

        if random.random() < p_action_0:
            return 0
        else:
            return 1

    def choose_action(self, horizon):
        if horizon < training:
            #chooze boltzmann, update reward?
        print "butts"


    def q_value(self):
        pass


class GLIEPolicy(Policy):
    def __init__(self, MDP, t):
        Policy.__init__(self, MDP, "GLIEPolicy")
        self.t = float(t)
        self.q_values = [[0, 0] for i in range(len(MDP.get_states()))]

    def choose_action(self, horizon):
        #boltzmann equation to choose action
        p_action_0 = exp(self.q_values[self.MDP_.get_state()][0] / self.t) / \
                        (exp(self.q_values[self.MDP_.get_state()][0] / self.t) +
                        exp(self.q_values[self.MDP_.get_state()][1] / self.t))

        if random.random() < p_action_0:
            return 0
        else:
            return 1
    def q_value(self):
        pass


"""
hand written agent that improves upon the base greedy in the first MDP
"""
class NoHandicapPolicy(Policy):

    def __init__(self, MDP, p):
        Policy.__init__(self, MDP, "NoHandicapPolicy")
        self.p = p

    def choose_action(self, horizon):
        PARK = 1
        DRIVE = 0
        if self.MDP_.get_available() and random.random() < self.p and not self.MDP_.get_handicapped():
            return PARK
        else:
            return DRIVE
"""
Hand written policy that performs best in low-availability worlds.
"""
class ImpatientPolicy(Policy):
    def __init__(self, MDP):
        Policy.__init__(self, MDP, "ImpatientPolicy")

    def choose_action(self, horizon):
        PARK = 1
        DRIVE = 0
        if self.MDP_.get_available() and not self.MDP_.get_handicapped():
            return PARK
        else:
            return DRIVE


"""
Drive when O is True, and when O is FALSE park with probaility p,.
"""
class GreedyPolicy(Policy):

    def __init__(self, MDP, p):
        Policy.__init__(self, MDP, "GreedyPolicy")
        self.p = p
    
    def choose_action(self, horizon):
        PARK = 1
        DRIVE = 0
        if self.MDP_.get_available() and random.random() < self.p:
            return PARK
        else:
            return DRIVE

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
