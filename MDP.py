#!/usr/bin/env python
"""
MDP is defined by a set of states, a set of actions, a transition function that takes you from one state to another, and a reward function
"""
class MDP:
    states_ = None
    actions_ = None
    transition_ = None
    reward_ = None
    rewards_ = None
    cur_state_ = None
    is_alive = True
    time_ = 0
    def __init__(self, states, actions, transition, rewards, initial_state):
        self.states_ = states #idk what these are represented as or what tehy should be represented as, i dont think i care?
        self.actions_ = actions
        self.transition_ = transition
        self.reward_ = 0
        self.rewards_ = rewards
        self.cur_state_ = initial_state
        self.time_ = 0

    """
    Use choose_transition from Transition object as your function
    """
    def take_action(self, action):
        new_state_idx = self.transition_.choose_transition(action, self.cur_state_)
        self.cur_state_ = self.states_[new_state_idx]
        self.time_ += 1
        self.reward_ += self.rewards_[self.cur_state_]

    def get_p_transition(self, state, action, r_state):
        return self.transition_.get_p_transition(action, state, r_state)
        
    def get_reward(self):
        return self.reward_

    def get_num_actions(self):
        return len(self.actions_)

    def get_legal_actions(self):
        return self.actions_
    
    def get_state_reward(self, state):
        return self.rewards_[state]

    def get_time(self):
        return self.time_

    def get_state(self):
        return self.cur_state_

    def get_states(self):
        return self.states_

    def get_parked(self):
        return self.cur_state_ % 4 > 1

    def get_available(self):
        return self.cur_state_ % 4 in (1, 3)

    def get_handicapped(self):
        front = len(self.states_) / 4 / 2
        return self.get_spot()== front or self.get_spot() == front - 1
            
        
    def get_spot(self):
        return self.cur_state_ / 4
           

    def __str__(self):
        result = ""
        result += "Current State: " + str(self.cur_state_) + "\n"
        result += "Current Reward: "+ str(self.reward_)+ "\n"
        result += "Current time: "+ str(self.time_)+ "\n"
        result += "Actions: "+ str(self.actions_)+ "\n"
        result += "Rewards: "+ str(self.rewards_)+ "\n"
        result += "States: "+ str(self.states_)  + "\n"
        result += "Transitions: \n" + str(self.transition_)
        return result


