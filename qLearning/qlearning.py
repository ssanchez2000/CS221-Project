#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:46:32 2017

@author: Olivier
"""

import numpy as np
import gym
import hashlib
from collections import defaultdict
from PIL import Image
import pickle
import os.path
import random
import math 
from collections import defaultdict

env = gym.make('Enduro-v0')
## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = tuple([3]*28)  # (one for each pixel)
# Number of discrete actions
NUM_ACTIONS = 3 # (left, right)
# Index of the action
ACTION_INDEX = 3

## Creating a Q-Table for each state-action pair
q_table = defaultdict(int)

# weight vector
#w = [0]*28

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

## Defining the simulation related constants

NUM_EPISODES = 500
MAX_T = 1000
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = False
render = False

final_pos = []
def reward_funct(state):
    values={}
    for i in range(10):
        A=Image.open("../10digits/"+str(i)+".png")
        A= np.array(A)
        values[(A[2,2,0],A[2,5,0],A[4,1,0],A[5,1,0])]=str(i)
    img_array=state[179:188,80:104]
    num1=img_array[:,0:8]
    num2=img_array[:,8:16]
    num3=img_array[:,16:]
    reward=values[(num1[2,2,0],num1[2,5,0],num1[4,1,0],num1[5,1,0])]
    reward1=values[(num2[2,2,0],num2[2,5,0],num2[4,1,0],num2[5,1,0])]
    reward2=values[(num3[2,2,0],num3[2,5,0],num3[4,1,0],num3[5,1,0])]
    return reward+reward1+reward2

 
def featureExtractor(array):
    #curve of the road
    curve = np.argmax(array.transpose(2,0,1)[0][52:53, ][0])-87
    
    # relative position to the right and left border
    road_pos_array = array.transpose(2,0,1)[0][145:146,][0]
    relevant_pix = [i for i in range(len(road_pos_array)) if road_pos_array[i]>190]
    left_margin = relevant_pix[1]-relevant_pix[0]
    car_pos = int(np.mean(relevant_pix))
    
    # line in front
    line = Image.fromarray(array.transpose(2,0,1)[0][90:130,55+int(curve/2):110+int(curve/2)]).resize((5, 2), Image.ANTIALIAS)
    tuple_line = list(np.fromstring(line.tobytes(), dtype=np.uint8))
    #Image.fromarray(array.transpose(2,0,1)[0][100:120,50+int(curve/5):110+int(curve/5)]).show()
    
    return tuple([curve, left_margin]+tuple_line)
    
def imageToTuple(array):
    image = Image.fromarray(array.transpose(2,0,1)[0][110:130, 55:120][::2, ::3]).resize((7, 4), Image.ANTIALIAS)
    #image.save(str(random.randint(1, 10))+".png")
    return tuple(np.fromstring(image.tobytes(), dtype=np.uint8))

def simulate():
    render = False
    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0
    
    for episode in range(NUM_EPISODES):
        #render = True
        if episode >150:
            render = True
        # Reset the environment
        obv = env.reset()
        # the initial state
        tupleState = imageToTuple(env.unwrapped.ale.getScreenGrayscale())
        state_0 = state_to_bucket(tupleState)
        t=0
        for t in range(MAX_T):
        #while True:
            #t+=1
            
            #potentially render
            if render:
                env.render()
            
            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)
            #reward = 200-int(reward_funct(obv))
            
            # Observe the result
            tupleState = featureExtractor(env.unwrapped.ale.getScreenGrayscale())
            state = state_to_bucket(tupleState)
            #print(state)

            # Update the Q based on the result
            best_q = np.amax([q_table[state+(a,)] for a in [1, 7, 8]])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])
            """prediction = sum([state_0[i]*w[i] for i in range(len(w))])
            v_opt = max([getQ(state, a) for a in [1, 7, 8]])
            target = reward + discount_factor*v_opt
            
            for i in range(len(w)):
                w[i] -= learning_rate * (prediction-target)*state[i]"""
                
            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                #print("\nEpisode = %d" % episode)
                #print("t = %d" % t)
                print("Action: %d" % action)
                #print("State: %s" % str(state))
                #print("Reward: %f" % reward)
                #print("Best Q: %f" % best_q)
                #print("Explore rate: %f" % explore_rate)
                #print("Learning rate: %f" % learning_rate)
                #print("Streaks: %d" % num_streaks)

                #print("")
            if t == MAX_T-1:
                print("Final Position", reward_funct(obv))
                final_pos.append(reward_funct(obv))
            if done:
            
               print("Episode %d finished after %f time steps" % (episode, t))
               print(200-reward)
               break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

def getQ(state, action, w):
    score = 0
    for f, v in featureExtractor(state, action):
        score += w[f] * v
    return score

def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = random.sample([1,7,8], 1)[0]
    # Select the action with the highest q
    else:
        action = np.argmax([q_table[state+(a,)] for a in [1, 7, 8]])
        if action ==0:
            action = 1
        if action ==1:
            action = 7
        if action ==2:
            action = 8
    return action


def get_explore_rate(t):
    #return 0.3
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/80)))

def get_learning_rate(t):
    #return 0.3
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/80)))

def state_to_bucket(state):
    bucket_indice = [int(state[0]/12), int(state[1]/10)]
    for i in range(2,len(state)):
        if state[i]>60:
            bucket_indice.append(1)
        else:
            bucket_indice.append(0)
    return tuple(bucket_indice)

def state_to_bucket_old(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i]<50:
            bucket_indice.append(0)
        elif state[i]<90:
            bucket_indice.append(1)
        else:
            bucket_indice.append(2)
    return tuple(bucket_indice)
            

#if __name__ == "__main__":
#    simulate()