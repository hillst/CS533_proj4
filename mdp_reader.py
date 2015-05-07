#!/usr/bin/env python
"""
Reads the spec MDP format and returns an MDP object with this format

3 2

0.2 0.8 0.0
0.0 0.2 0.8
1.0 0.0 0.0

0.9 0.05 0.05
0.05 0.9 0.05
0.05 0.05 0.9

-1.0 -1.0 0.0

nStates nActions
\n
Transition probabilities for action i
\n
Transition probabilities for action i+1
\n
Rewards for each state

"""
def ReadMDP(filename):
    #states = (State, State, State)
    #actions = (Action, Action, Action)
    #Transition = [state2][action][state2] (returns probability of transitioning to the from state1 to state2 using action
    #Reward = state => reward
    first = True
    with open(filename, 'r') as fd:
        line = fd.readline()
        nStates, nActions = [int(thing) for thing in line.split()]
        transition_data = []
        for i in range(nStates):
            transition_data.append([])
            for j in range(nActions):
                transition_data[i].append([])
                for k in range(nStates):
                    transition_data[i][j].append(0.0)
                    
        fd.readline() #consume newline
        for i in range(nActions):
            for j in range(nStates):
                line = fd.readline()
                transition_data[j][i] = [ float(thing) for thing in line.split() ]
            fd.readline() #consume
        
        rewards = [float(thing) for thing in fd.readline().split()]
        return (transition_data, rewards, nStates, nActions)

