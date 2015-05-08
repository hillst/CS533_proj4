# Parking MDP reinforcement Learning

## MDP generator
    The MDP generator takes several parameters pertaining to the MDP generated, these include different rewards and penalties. A key fact to keep in mind is that the 
distance reward should be negative and signifcantly smaller than the parking reward. A good way to keep all of these rewards positive is to set the parking reward == n and the distance reward == -1

## Example 1
    Probability of available is .8, parking reward is 10, crashing reward is -30, handicap reward is -4, driving reward is -1, n_states = 10, and -1 is the cost for distance.

## Example 2
    Example 2 is similar to the first example, however the probability of finding a spot is .1. This punishes agents that spend too much time driving around.

# Policies

## RandomPolicy
    The agent just sort of randomly chooses an action. He usually crashes.

## GreedyPolicy
    Given a probability p the agent parks in an available spot, otherwise he drives. He does not care if it's a handicapped spot.

## NoHandicapPolicy
    This is the same as the GreedyPolicy, however the agent will not park in a handicapped spot. In the first MDP example, this agent performs better than the GreedyPolicy due to the high availableility of spots.

## ImpatientPolicy
    The ImaptientAgent is in a rush and does not care about walking, so they park as soon as something is available. This agent performs best in environments when the parking lot is very full.

