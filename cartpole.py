#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:19:56 2017

@author: Olivier
"""

from __future__ import print_function
from PIL import Image

import sys, gym

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

env = gym.make('Enduro-v0' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
ROLLOUT_TIME = 10000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    reward = 0
    list_num = []
    while len(list_num)<10:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1
        #action = env.action_space.sample()
        obser, r, done, info = env.step(a)
        reward+=r
        #print(env.unwrapped.ale.getFrameNumber(), env.unwrapped.ale.getEpisodeFrameNumber())

        num=obser[179:188,96:104]
        Image.fromarray(num)
        env.render()
        if (not any((num == x).all() for x in list_num)):
            list_num.append(num)
        if len(list_num)==10:
            return list_num
            break
        if done:
            break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            import time
            time.sleep(0.1)

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")


l = rollout(env)


for i in range(10):
    
