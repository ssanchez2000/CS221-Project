import numpy as np
import gym

env = gym.make('Enduro-v0')

class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")


class ValueIteration(MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print("ValueIteration: %d iterations" % numIters)
        self.pi = pi
        self.V = V

class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        #self.states = set()
        self.states=list()
        queue = []
        #self.states.add(self.startState())
        self.states.append(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if(not any((newState == x).all() for x in self.states)):
                    #if newState not in self.states:
                        #self.states.add(newState)
                        self.states.append(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)
# print self.states

class player(MDP):
    #HErwe obs is a numpy array of image observation
    #where env is atari env
    def __init__(self,env):
        #self.obs = obs
        self.env = env
    def startState(self):
        return env.reset()

    def actions(self, state):
        return [0,1,2,3,4,5,6,7,8]

    #our states are obs, returns a list with tuple
    def succAndProbReward(self, state, action):
        result = []
        obs,reward,done,info = self.env.step(action)
        #end state check
        if done:
            return []

        prob = float(1)/float(len(self.actions(obs)))
        result.append((obs,prob,reward))

        return result

    def discount(self):
        return 1

for i_episode in range(1):
    observation = env.reset()
    mdp = player(env)
    mdp.computeStates()
    algorithm = ValueIteration()
    algorithm.solve(mdp, .001)
