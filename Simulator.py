#!/usr/bin/env python
import sys
from mdp_reader import ReadMDP
from Policy import RandomPolicy
from Policy import ValueIterationPolicy
from MDP import MDP
from Transition import Transition
def main():
    if len(sys.argv) < 4:
        print >> sys.stderr, "Usage: Simulator.py\t<MDP.txt>\t<RandomPolicy|OptimalPolicy>\t<Epsilon>\t<discount>"
        sys.exit(-1)

    filename = sys.argv[1]
    discount = float(sys.argv[4])
    transition_p, rewards, nStates, nActions = ReadMDP(filename)
    epsilon = float(sys.argv[3]) 
    states = range(nStates)
    actions = range(nActions)
    initial_state = 0
    user_policy = sys.argv[2]
    transition_function = Transition(transition_p)
    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = ValueIterationPolicy(MyMDP, user_policy , epsilon, discount)
    policy.display_policy()
    print ""
    policy.display_value_f()
    #print policy.bellman_backup(initial_state, 10)
    #policy = RandomPolicy(RandomMDP, "RandomPolicy")
    #run_simulation(RandomMDP, policy, epsilon)

def run_simulation(MDP, policy, epsilon):
    print "Starting simulation for", MDP

    while MDP.get_time() < epsilon:
        action = policy.choose_action(MDP.get_time())
        print "[TIME", MDP.get_time() ,"]:", policy.get_name(), "chose action", action
        MDP.take_action(action)
        print "[TIME", MDP.get_time() ,"]: Moved to state", MDP.get_state(), "Current reward %.3f." % MDP.get_reward()
        

if __name__ == "__main__":
    main()
