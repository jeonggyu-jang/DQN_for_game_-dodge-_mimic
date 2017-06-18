import time
import math
import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn
import copy
import game
import os
import sys
input_size = game.input_size * game.input_size
roi_width = game.input_size
output_size = 9
#model_path = 'model_0612.ckpt-840' 
model_path = 'model_0615.ckpt-5600' 
dis = 0.99
REPLAY_MEMORY = 100000
REPLAY_MEMORY_RECENT = 500

map_data=[[1 for j in range(game.y_res)] for i in range(game.x_res)]
user_loc={'x':int(game.x_res/2),'y':int(game.y_res/2)}
ball_list = game.ball_list_init(game.ball_list)
map_data,reward,done=game.mapping2map(user_loc,ball_list,map_data,1)
roi_data = game.roi_calculation(map_data,user_loc,game.input_size)
os.system('clear')
game.game_print(map_data,roi_data)

sess=tf.Session()
mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
sess.run(tf.global_variables_initializer())
mainDQN.saver.restore(sess,model_path)

episode =0
state = roi_data
step_count = 0
step_buffer = deque()
avg_reward = 0
max_step = 0
while episode<5000:
	done = False
	action = 0
	seq_list=[(),(),()]
	next_seq_list=[(),(),()]
	for i in range(3):
		if i==2 :
			step_count += 1
			if game.ball_step != 0 :
				game.ball_step = game.ball_step - 1
			seq_list[i]=(state, action, reward, next_state, done)
			state_temp=[[[5,5,5] for j in range(roi_width)] for i3 in range(roi_width)]
			i2=0
			for state_t, action, reward, next_state_t, done in seq_list:
				for k in range(roi_width) : 
					for m in range(roi_width) : 
						state_temp[k][m][i2] = state_t[k][m]
				i2+=1
			QQ=mainDQN.predict(state_temp)
			action = np.argmax(QQ)
			f3=open("predict_log.txt","a")
			f3.write('episode : %3d, step_count : %3d, i = %3d, max_step = %d \n'%(episode,step_count,i,max_step))
			f3.close()
			print "DQN:"+str(action)+" Q:"+str(QQ)
			next_state, reward, done, ball_list,user_loc = game.step(action,game.input_size,map_data,user_loc,ball_list,1,episode)
		elif i<2 :
			print "-------------   no  choice "+str(i)+" ----------------\n"
			next_state, reward, done,ball_list,user_loc= game.step(action,game.input_size,map_data,user_loc,ball_list,0,episode)
			seq_list[i]=(state, action, reward, next_state, done)
		print "----- episode : "+str(episode)+ " reward : "+str(reward)+"  step : "+str(step_count)+" avg_reward : "+str(avg_reward)+"  max_step : "+str(max_step)+"----"	
		state = next_state
		if done == True :
			map_data=[[5 for j in range(game.y_res)] for i in range(game.x_res)]
			user_loc={'x':int(game.x_res/2),'y':int(game.y_res/2)}
			ball_list = game.ball_list_init(ball_list)
			map_data,reward,done=game.mapping2map(user_loc,ball_list,map_data,1)
			if max_step < step_count :
				max_step = step_count
			step_buffer.append(step_count)
			if len(step_buffer) > 100:
				step_buffer.popleft()
				avg_reward = np.mean(step_buffer)
			episode = episode + 1
			step_count = 0
			game.ball_step = 10
			break
