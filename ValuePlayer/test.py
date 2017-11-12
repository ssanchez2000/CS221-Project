import gym
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
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
env = gym.make('Enduro-v0')

def reward_funct(state,t):
	key={}
	key["a"]="0"
	key["-,"]="2"
	key["m"]="1"
	key["."]="9"
	key["s"]="8"
	key["v:"]="7"
	img_array=state[179:188,80:104]
	num1=img_array[:,0:8]
	num2=img_array[:,8:16]
	num3=img_array[:,16:]

	num1=Image.fromarray(num1)
	num1=num1.convert('1')
	reward=pytesseract.image_to_string(num1,config='-psm 7')

	num2=Image.fromarray(num2)
	num2=num2.convert('1')
	reward1=pytesseract.image_to_string(num2,config='-psm 7')

	num3=Image.fromarray(num3)
	num3=num3.convert('1')
	reward2=pytesseract.image_to_string(num3,config='-psm 7')
	reward=key.setdefault(reward,reward)
	reward1=key.setdefault(reward1,reward1)
	reward2=key.setdefault(reward2,reward2)

	return reward+reward1+reward2

for i_episode in range(1):
	observation = env.reset()
	print(observation)
	t=0
	done=True
	while(not done):
		t=t+1
		#env.render()
		state=observation
		action=env.action_space.sample()
		action=1
		observation,reward,done,info=env.step(action)
		reward=reward_funct(observation,t)
		print(reward)
		if(t==1):
			break

		if done:
			print("episode finished after {} timesteps".format(t+1))
