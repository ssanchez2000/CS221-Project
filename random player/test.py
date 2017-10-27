import gym
import matplotlib.pyplot as plt
#31 steps per no (for 0)
#0 - none (31 steps)
#1 - accelerate (18 steps)
#2- go right (30 steps)
#3-go left (32 steps)
#4- slow? (33 steps)
#5- slow and right (34 steps)
#6- slow and left (34 steps)
#7- accelerate and right (16 steps)
#8- accelrate and left (17 steps)
game_choice='MountainCar-v0'
env = gym.make('Enduro-v0')
name="eight/"
for i_episode in range(1):
	observation = env.reset()
	for t in range(100):
		env.render()
		state=observation
		plt.imshow(observation)
		plt.savefig(name+"img"+str(t))
		action=env.action_space.sample()
		action=8
		observation,reward,done,info=env.step(action)
		if(reward !=0):
			print("reward: ",reward)
		if done:
			print("episode finished after {} timesteps".format(t+1))
