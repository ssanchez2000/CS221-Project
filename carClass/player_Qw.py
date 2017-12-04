import numpy as np
import gym
import hashlib
from collections import defaultdict
from PIL import Image
import pickle
import os.path
import random
import math
import matplotlib.pyplot as plt
env = gym.make('Enduro-v0')

def hashState(state):
    if isinstance(state,tuple):
        return (hashlib.sha256(str(state[0])).hexdigest(),hashlib.sha256(str(state[1])).hexdigest())
    return hashlib.sha256(state).hexdigest()

class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")



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
    # m - no of games to run
    def computeStates(self,m):
        #self.states = set()
        self.states=list()
        queue = []
        #self.states.add(self.startState())
        self.states.append(self.startState())
        for i in range(m):
            queue.append(self.startState())
            while len(queue) > 0:
                state = queue.pop()
                action=random.choice(self.actions(state))
                #for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if(not any((newState == x).all() for x in self.states)):
                        self.states.append(newState)
                        queue.append(newState)
        print(len(self.states))
        # print "%d states" % len(self.states)
# print self.states

class player(MDP):
    #HEre we obs is a numpy array of image observation
    #where env is atari env
    def __init__(self,env):
        #self.obs = obs
        self.env = env
	self.state =0
        self.values={}
        for i in range(10):
            A=Image.open("../10digits/"+str(i)+".png")
            A= np.array(A)
            self.values[(A[2,2,0],A[2,5,0],A[4,1,0],A[5,1,0])]=str(i)

    def startState(self):
        return env.reset()

    def actions(self, state):
        return [1,7,8]
        #return [0,1,2,3,4,5,6,7,8]

    def reward_funct(self,state):
        img_array=state[179:188,80:104]
        num1=img_array[:,0:8]
        num2=img_array[:,8:16]
        num3=img_array[:,16:]
        reward=self.values[(num1[2,2,0],num1[2,5,0],num1[4,1,0],num1[5,1,0])]
        reward1=self.values[(num2[2,2,0],num2[2,5,0],num2[4,1,0],num2[5,1,0])]
        reward2=self.values[(num3[2,2,0],num3[2,5,0],num3[4,1,0],num3[5,1,0])]
        return reward+reward1+reward2
    #our states are obs, returns a list with tuple
    def succAndProbReward(self, state, action):
        result = []
        #obs,reward,done,info = self.env.step(action)
        self.state,reward,done,info = self.env.step(action)
        #reward=200-int(self.reward_funct(obs))
        reward=200-int(self.reward_funct(self.state))
        if done:
            return []

        #prob = float(1)/float(len(self.actions(obs)))
        #prob = float(1)/float(len(self.actions(self.state)))
        prob = 1
        #result.append((obs,prob,reward))
        result.append((self.state,prob,reward))
        return result

    def discount(self):
        return 1



# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        sequence = [hashState(state)]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(hashState(newState))

            rl.incorporateFeedback(state, action, reward, hashState(newState))
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRLAlgorithm(RLAlgorithm):
    def __init__(self, pi): self.pi = pi

    # Just return the action given by the policy.
    def getAction(self, state): return self.pi[state]

    # Don't do anything: just stare off into space.
def incorporateFeedback(self, state, action, reward, newState): pass


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
	    f_hash = hashState(f)
            #score += self.weights[f] * v
	    score += self.weights[f_hash] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        #check if s is an endcase
        def GETAction(s):
            QsMax=0
            for i, a in enumerate(self.actions(s)):
                Qs = self.getQ(s,a)
                if QsMax <= Qs:
                    QsMax = Qs

            return QsMax

        Qval = self.getQ(state,action)
        if newState == None:
            for f,v in self.featureExtractor(state,action):
                self.weights[hashState(f)] = self.weights[hashState(f)] - self.getStepSize()*(Qval-reward)*v
        else:
            vopt = GETAction(newState)
            for f,v in self.featureExtractor(state,action):
                self.weights[hashState(f)] = self.weights[hashState(f)] - self.getStepSize()*(Qval-(reward+self.discount*vopt))*v
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

observation = env.reset()
mdp = player(env)

'''
m=2 # no of games we want it to run
if(os.path.isfile('mdp_states_'+str(m)+'.pkl')):
    print("looks like you've already saved for m="+str(m))
    print("do you want to save again?")
    c=input()
    if(c[0]=="y"):
        print("fine, i hope you know what youre doing")
        mdp.computeStates(m)
        with open('mdp_states_'+str(m)+'.pkl', 'wb') as output:
            pickle.dump(mdp, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("time is money :P")
else:
    mdp.computeStates(m)
    with open('mdp_states_'+str(m)+'.pkl', 'wb') as output:
        pickle.dump(mdp, output, pickle.HIGHEST_PROTOCOL)

with open('mdp_states_'+str(m)+'.pkl', 'rb') as inputfile:
    mdp = pickle.load(inputfile)
'''
iters = [10,20,30,40,50]
iters_rewards = []
for i in iters:
    featureExtractor = identityFeatureExtractor
    rl = QLearningAlgorithm(mdp.actions,mdp.discount(),featureExtractor,0.2)
    rewards =simulate(mdp, rl, numTrials=i, maxIterations=1000, verbose=False,
                 sort=False)
    tempweights = rl.weights
    rl = QLearningAlgorithm(mdp.actions,mdp.discount(),featureExtractor,0)
    rl.weights = tempweights
    mdp.computeStates(1)
    rlVals = []
    for s in mdp.states:
       rlVals.append(rl.getAction(s))
    #for i in range(len(rlVals)):
    #   print rlVals[i]
    print "actions: ", set(rlVals)
    print rewards
    #print rewards
    #print float(sum(rewards))/float(len(rewards))
    iters_rewards.append(float(sum(rewards))/float(len(rewards)))
    #lst=np.arange(1,i+1)
    #plt.plot(lst.reshape((1,i)),np.asarray(rewards).reshape((1,i)))
    lst = range(1,i+ 1)
    plt.plot(lst,rewards)
    plt.ylabel("Rewards")
    plt.xlabel("Episode")
    #plt.show()
    fig_name = "numTrails_"+str(i)+".png"
    plt.savefig(fig_name)
    print "saved plot"
    plt.gcf().clear()

plt.plot(iters,iters_rewards)
plt.ylabel("Average Rewards")
plt.xlabel("Simulation: Number of Episodes")
fig_name = "avg_rewards"
plt.savefig(fig_name)

#algorithm = ValueIteration()
#algorithm.solve(mdp, .001)
#print(algorithm.pi.values())
#for i_episode in range(1):
    #print(algorithm.pi.values())
print("done")
