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
#name="eight/"
for i_episode in range(10):
	observation = env.reset()
	#for t in range(1000):
	t=0
	done=False
	while(not done):
		t=t+1
		env.render()
		state=observation
		action=env.action_space.sample()
		action=1
		observation,reward,done,info=env.step(action)
		if(done):
			plt.imshow(observation)
			plt.savefig("img_"+str(i_episode+1)+"_"+str(t))
		if(reward !=0):
			print("reward: ",reward)
		if done:
			print("episode finished after {} timesteps".format(t+1))
