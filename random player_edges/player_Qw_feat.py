import numpy as np
import gym
import hashlib
from collections import defaultdict
from PIL import Image
import pickle
import os.path
import random
import math
import edge_det
from skimage import data, io, filters,color, morphology
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
env = gym.make('Enduro-v0')

def hashState(state):
    if isinstance(state,tuple):
        return (hashlib.sha256(str(state[0])).hexdigest(),hashlib.sha256(str(state[1])).hexdigest())
    return hashlib.sha256(state).hexdigest()

class MDP_Algorithm:
    def solve_mdp(self, mdp): raise NotImplementedError("Override me please")



class MDP_QL:
    # Return the start state.
    def start_State(self): raise NotImplementedError("Override me please")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me please")

    def succAnd_ProbReward(self, state, action): raise NotImplementedError("Override me please")

    def discount(self): raise NotImplementedError("Override me please")

    def computeStates(self,m):
        #self.states = set()
        self.states=list()
        queue = []
        #self.states.add(self.start_State())
        self.states.append(self.start_State())
        for i in range(m):
            queue.append(self.start_State())
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

class player(MDP_QL):
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

    def start_State(self):
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
# Return the list of rewards that we get for each trial (accummulative and for end state).
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
    last_rewards = [] #store last state reward value per trail
    for trial in range(numTrials):
        state = mdp.start_State()
        sequence = [hashState(state)]
        totalDiscount = 1
        totalReward = 0
        rewards = []
        for _ in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporate_Feedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(hashState(newState))

            rl.incorporate_Feedback(state, action, reward, hashState(newState))
            totalReward += totalDiscount * reward
            rewards.append(reward)
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        last_rewards.append(rewards[len(rewards)-1])
        totalRewards.append(totalReward)
    return totalRewards, last_rewards

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporate_Feedback()) to the RL algorithm, so it can adjust
# its parameters.
class RL_Algorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me please")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporate_Feedback(self, state, action, reward, newState): raise NotImplementedError("Override me please")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class Fixed_RLAlgorithm(RL_Algorithm):
    def __init__(self, pi): self.pi = pi

    # Just return the action given by the policy.
    def getAction(self, state): return self.pi[state]

    # Don't do anything: just stare off into space.
def incorporate_Feedback(self, state, action, reward, newState): pass



class QLearning_Algorithm(RL_Algorithm):
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
    def incorporate_Feedback(self, state, action, reward, newState):
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
def FeatureExtractor(state, action):
    Image.fromarray(observation).save('temp_state.png')
    image = io.imread('temp_state.png')
    featureKey = (state, action)
    featureValue = edge_det.img_label(edge_det.crop(color.rgb2grey(image)),edge_det.crop(image))
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
#iters = [10,20,30,40,50]
#iters = [5,20,30,40,50]
iters = [5]
iters_rewards = []
for i in iters:
    featureExtractor = FeatureExtractor
    rl = QLearning_Algorithm(mdp.actions,mdp.discount(),featureExtractor,0.2)
    total_rewards,last_rewards =simulate(mdp, rl, numTrials=i, maxIterations=1000, verbose=False,
                 sort=False)
    tempweights = rl.weights
    rl = QLearning_Algorithm(mdp.actions,mdp.discount(),featureExtractor,0)
    rl.weights = tempweights
    mdp.computeStates(1)
    rlVals = []
    for s in mdp.states:
       rlVals.append(rl.getAction(s))
    #for i in range(len(rlVals)):
    #   print rlVals[i]
    print "actions: ", set(rlVals)
    print "total rewards: ",total_rewards
    print "end state rewards: ", last_rewards
    #print rewards
    #print float(sum(rewards))/float(len(rewards))
    iters_rewards.append(float(sum(total_rewards))/float(len(total_rewards)))
    #lst=np.arange(1,i+1)
    #plt.plot(lst.reshape((1,i)),np.asarray(rewards).reshape((1,i)))
    #lst = range(1,i+ 1)
    #plt.plot(lst,total_rewards)
    #plt.ylabel("Cummulative Rewards")
    #plt.xlabel("Episode")
    #plt.show()
    #fig_name = "numTrails_"+str(i)+"cummulative_rewards.png"
    #plt.savefig(fig_name)
    #print "saved plot (total rewards)"
    #plt.gcf().clear()
    #plt.plot(lst,last_rewards)
    #plt.ylabel("End State Reward")
    #plt.xlabel("Episode")
    #fig_name = "numTrails_"+str(i)+"end_state_reward.png"
    #plt.savefig(fig_name)
    #print "saved plot (last rewards)"
    #plt.gfc().clear()
    #save data
    np.savetxt("total_rewards.csv",np.asarray(total_rewards))
    np.savetxt("last_rewards.csv",np.asarray(last_rewards))
    #if i==5:
    #    break

#plt.plot(iters,iters_rewards)
#plt.ylabel("Average Rewards")
#plt.xlabel("Simulation: Number of Episodes")
#fig_name = "avg_rewards"
#plt.savefig(fig_name)

#algorithm = ValueIteration()
#algorithm.solve(mdp, .001)
#print(algorithm.pi.values())
#for i_episode in range(1):
    #print(algorithm.pi.values())
print("done")
