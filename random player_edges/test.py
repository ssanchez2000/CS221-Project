import gym
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, io, filters,color, morphology
import numpy as np
import edge_det
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
values={}
for i in range(10):
	A=Image.open("../10digits/"+str(i)+".png")
	A= np.array(A)
	values[(A[2,2,0],A[2,5,0],A[4,1,0],A[5,1,0])]=str(i)

def reward_funct(state,t):
	img_array=state[179:188,80:104]
	num1=img_array[:,0:8]
	num2=img_array[:,8:16]
	num3=img_array[:,16:]
	reward=values[(num1[2,2,0],num1[2,5,0],num1[4,1,0],num1[5,1,0])]
	reward1=values[(num2[2,2,0],num2[2,5,0],num2[4,1,0],num2[5,1,0])]
	reward2=values[(num3[2,2,0],num3[2,5,0],num3[4,1,0],num3[5,1,0])]

	return reward+reward1+reward2

for i_episode in range(1):
	observation = env.reset()
	#print(observation.shape)
	#img=Image.fromarray(observation[50:158,35:138])
	#img.show()
	#print(observation)
	t=0
	done=False
	while(not done):
		t=t+1
		env.render()
		state=observation
		action=env.action_space.sample()
		action=1
		observation,reward,done,info=env.step(action)
 		#print t
	        #print edge_det.img_label(edge_det.crop(color.rgb2grey(observation)),t)	
		reward=reward_funct(observation,t)
		#if t >= 1 and t<=10:
		#    print edge_det.img_label_car(edge_det.crop_car(color.rgb2grey(observation)),t)

	#	print(reward)
	#	if(t==179):
	#		Image.fromarray(observation).save('frame_'+str(t)+'.png')
	#		break
	#	if done:
	#		print("episode finished after {} timesteps".format(t+1))










