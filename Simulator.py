#!/usr/bin/env python
import sys
from mdp_reader import ReadMDP
from Policy import RandomPolicy
from Policy import GreedyPolicy
from Policy import ValueIterationPolicy
from Policy import NoHandicapPolicy
from Policy import ImpatientPolicy
from Policy import QLearningPolicy

from MDP import MDP
from Transition import Transition
def main():
    if len(sys.argv) < 5:
        print >> sys.stderr, "Usage: Simulator.py\t<MDP.txt>\t<RandomPolicy|OptimalPolicy>\t<Epsilon>\t<discount>\t<alpha>\t<training>"
        #sys.exit(-1)
    else:
        filename = sys.argv[1]
        training = float(sys.argv[6])
        alpha = float(sys.argv[5])
        discount = float(sys.argv[4])
        epsilon = float(sys.argv[3])
        user_policy = sys.argv[2]


    filename, epsilon, discount, alpha, training, user_policy = "example_1.mdp", .5, .9, .3, 200, "OptimalPolicy"

    transition_p, rewards, nStates, nActions = ReadMDP(filename)
    states = range(nStates)
    actions = range(nActions)
    initial_state = 0
    transition_function = Transition(transition_p)
    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = ValueIterationPolicy(MyMDP, user_policy , epsilon, discount)
    policy.display_policy()
    print ""
    policy.display_value_f()
    #print policy.bellman_backup(initial_state, 10)
    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = RandomPolicy(MyMDP, "RandomPolicy")
    evaluate_policies(policy, MyMDP)
    #run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = GreedyPolicy(MyMDP, .3)
    #evaluate_policies(policy, MyMDP)

    #run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = ImpatientPolicy(MyMDP)
#    evaluate_policies(policy, MyMDP)

    #run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = NoHandicapPolicy(MyMDP, .3)
   # evaluate_policies(policy, MyMDP)

    #run_simulation(MyMDP, policy)

    MyMDP = MDP(states, actions, transition_function, rewards, initial_state)
    policy = QLearningPolicy(MyMDP, alpha, discount)
    run_training(MyMDP, policy, training)
    MyMDP.reset()
    evaluate_policies(policy, MyMDP)

    #print policy.q_values
    #run_simulation(MyMDP, policy)



def evaluate_policies(policy, MDP):
    total_reward, handicapped, crashed = 0,0,0
    num_sims = 10000
    for i in range(num_sims):
        run_simulation(MDP, policy)
        #maybe do something fancier
        total_reward += MDP.get_reward()
        if MDP.get_handicapped():
            handicapped += 1
        if not MDP.get_available():
            crashed += 1
        MDP.reset()
    print policy.get_name(), total_reward / num_sims, handicapped, crashed


def run_simulation(MDP, policy):
    #print "Starting simulation for given MDP"
    while not MDP.get_parked():
        action = policy.choose_action(MDP.get_time())
        #print "[TIME", MDP.get_time() ,"]:", policy.get_name(), "chose action", action
        policy.take_action(action)
        #print "[TIME", MDP.get_time() ,"]: Moved to state", MDP.get_state(), "Current reward %.3f." % MDP.get_reward()
    #print "Exited in (spot, handicapped, available):", MDP.get_spot(), MDP.get_handicapped(), MDP.get_available()

def run_training(MDP, policy, horizon):
    t = 0
    trajectory = []
    while t < horizon:
        if MDP.get_parked():
            # we need to make it do one more update.
            action = policy.choose_training_action()
            state = MDP.get_state()
            trajectory.append((state, action, MDP.get_state()))
            #reset our simulator
            MDP.reset()
            policy.q_updates(trajectory)
            trajectory = []
        else:
            #record trajectory
            action = policy.choose_training_action()
            state = MDP.get_state()
            policy.take_action(action)
            trajectory.append((state, action, MDP.get_state()))
        t += 1

if __name__ == "__main__":
    main()
