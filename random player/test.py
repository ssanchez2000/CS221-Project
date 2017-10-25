import gym

game_choice='MountainCar-v0'
env = gym.make('Enduro-v0')

for i_episode in range(2):
	observation = env.reset()
	for t in range(1000):
		env.render()
		state=observation[1]
		#print("state: ",state)
		action=env.action_space.sample()
		#print(env.action_space)
		#action=1
		#if(state>0):
		#	action=2
		#else:
		#	action=0
		print("action:",action)
		observation,reward,done,info=env.step(action)
		reward=observation[0]
		#print("reward: ",reward)
		if done:
			print("episode finished after {} timesteps".format(t+1))
