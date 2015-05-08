#!/usr/bin/env python
import sys
from mdp_reader import ReadMDP
from Policy import RandomPolicy
from Policy import GreedyPolicy
from Policy import ValueIterationPolicy
from Policy import NoHandicapPolicy
from Policy import ImpatientPolicy
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
    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = RandomPolicy(MyMDP, "RandomPolicy")
    run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = GreedyPolicy(MyMDP, .3)
    run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = ImpatientPolicy(MyMDP)
    run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = NoHandicapPolicy(MyMDP, .3)
    run_simulation(MyMDP, policy)

def run_simulation(MDP, policy):
    print "Starting simulation for given MDP"

    while MDP.get_parked() == False:
        action = policy.choose_action(MDP.get_time())
        print "[TIME", MDP.get_time() ,"]:", policy.get_name(), "chose action", action
        MDP.take_action(action)
        print "[TIME", MDP.get_time() ,"]: Moved to state", MDP.get_state(), "Current reward %.3f." % MDP.get_reward()
    print "Exited in (spot, handicapped, available):", MDP.get_spot(), MDP.get_handicapped(), MDP.get_available()

if __name__ == "__main__":
    main()
