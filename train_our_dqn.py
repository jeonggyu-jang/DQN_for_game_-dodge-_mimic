import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
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
model_path = "./model_0613.ckpt"
dis = 0.99
REPLAY_MEMORY = 100000
REPLAY_MEMORY_RECENT = 500
model_path_in = 'model_0612.ckpt-840'

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
		y_stack = np.empty(0).reshape(0,mainDQN.output_size)
		x_stack = []
		for seq_data in train_batch :
			state=[[[1,1,1] for j in range(roi_width)] for i in range(roi_width)]
			next_state=[[[1,1,1] for j in range(roi_width)] for i in range(roi_width)]
			i=0
			for state_t, action, reward_t, next_state_t, done in seq_data:
				if i==2:
					reward = reward_t
				elif i>2:
					if reward >reward_t :
						reward = reward_t
				for k in range(roi_width) : 
					for m in range(roi_width) : 
						if i < 3 :
							state[k][m][i] = state_t[k][m]
						else :
							next_state[k][m][i-3] = state_t[k][m]
				i+=1
			Q = mainDQN.predict(state)
			if done:
				Q[0, action] = reward
			else:
				Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
			y_stack = np.vstack([y_stack, Q])
			x_stack.append(state)
		return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops( dest_scope_name="target", src_scope_name="main"):

		op_holder = []

		src_vars = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
		dest_vars = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

		for src_var, dest_var in zip(src_vars, dest_vars):
				op_holder.append(dest_var.assign(src_var.value()))

		return op_holder

def o_dqn(roi_data,map_data,user_loc,ball_list):
		loss =0 
		avg_reward=0
		f=open("train_log.txt","w")
		f3=open("predict_log.txt","w")
		f3.close()
		action_list=[0,1,2,3,4,5,6,7,8]
		max_episodes = 10000
		replay_buffer = deque()
		replay_buffer_recent = deque()

		last_100_game_reward = deque()

		with tf.Session() as sess:
				mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
				targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
				tf.global_variables_initializer().run()

				copy_ops = get_copy_var_ops(dest_scope_name="target",
																		src_scope_name="main")
				mainDQN.saver.restore(sess,model_path_in)
				sess.run(copy_ops)
				state = roi_data

				for episode in range(max_episodes):
						e = 1. / ((episode / 10) + 1)
						done = False
						step_count = 0
						flag = 0
						action = 0
						while not flag:
							reward_flag =1
							seq_list=[(),(),()]
							next_seq_list=[(),(),()]
							for i in range(6):
								if i==2 :
									step_count += 1
									seq_list[i]=(state, action, reward, next_state, done)
									if np.random.rand(1) < e:
											action = random.choice(action_list)
											print "------------- random choice :"+str(action)+" ----------------\n"
									else:
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
											f3.write('episode : %3d, step_count : %3d, i = %3d  '%(episode,step_count,i))
											f3.close()
											print "--- DQN choice :"+str(action)+"  Q : "+str(QQ)+" Q_sum = "
									next_state, reward, done, ball_list,user_loc = game.step(action,game.input_size,map_data,user_loc,ball_list,1,episode)
									ball_list_temp = copy.deepcopy(ball_list)
									user_loc_temp = copy.deepcopy(user_loc)
									state_temp = copy.deepcopy(next_state)
									seq_list[i]=(state, action, reward, next_state, done)
								elif i<2 :
									print "-------------   no  choice"+str(i)+"  "+str(action)+" ----------------\n"
									next_state, reward, done,ball_list,user_loc= game.step(action,game.input_size,map_data,user_loc,ball_list,0,episode)
									seq_list[i]=(state, action, reward, next_state, done)
								elif i>2 and i<5 :
									print "-------------   no  choice"+str(i)+"  "+str(action)+" ----------------\n"
									next_state, reward, done,ball_list,user_loc= game.step(action,game.input_size,map_data,user_loc,ball_list,0,-1)
									next_seq_list[i-3]=(state, action, reward, next_state, done)
								elif i==5 :
									reward = reward_flag
									next_seq_list[i-3]=(state, action, reward, next_state, done)
									next_state = state_temp
									if flag == 0 :
										ball_list = ball_list_temp
										user_loc = user_loc_temp
									elif flag == 1 :
										replay_buffer_recent.append((seq_list+next_seq_list))
										if len(replay_buffer_recent) > REPLAY_MEMORY_RECENT:
											replay_buffer_recent.popleft()
										map_data=[[5 for j in range(game.y_res)] for i in range(game.x_res)]
										user_loc={'x':int(game.x_res/2),'y':int(game.y_res/2)}
										ball_list = game.ball_list_init(ball_list)
										map_data,reward,done=game.mapping2map(user_loc,ball_list,map_data,1)
										roi_data = game.roi_calculation(map_data,user_loc,game.input_size)
										os.system('clear')
								if i < 5 :
									print "----- episode : "+str(episode)+ " reward : "+str(reward)+" done : "+str(done)+"  step : "+str(step_count)+"  Loss : "+str(loss)+" avg_reward : "+str(avg_reward)+" ----"
								
								state = next_state
								if done == True :
									flag=1
								if i >= 2 and reward_flag > reward :
									reward_flag=reward
							
							replay_buffer.append((seq_list+next_seq_list))
							if len(replay_buffer) > REPLAY_MEMORY:
									replay_buffer.popleft()

					
						f.write("Episode: {}	steps: {}\n".format(episode, step_count))
						if episode % 50 == 0 and episode > 499:	
								for train in range(50):
										t_num = 0
										b_s = 24
										max_t_num=8
										max_t = 1
										minibatch = random.sample(replay_buffer, b_s)
										minibatch2 = random.sample(replay_buffer_recent, 8)
										minibatch=minibatch+minibatch2
										loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)
										print 'training... loss = %f --%d'%(loss,train)
										f.write("Loss: {}\n".format(loss))
								sess.run(copy_ops)
						if episode % 100 == 0 :
							save_path = mainDQN.saver.save(mainDQN.session,model_path,global_step=episode)							
							print("Model(episode : ",episode,") saved in file : ",save_path)
						last_100_game_reward.append(step_count)

						if len(last_100_game_reward) > 50:
							last_100_game_reward.popleft()
							avg_reward = np.mean(last_100_game_reward)
							if avg_reward > 100:
								print("Game Cleared in {episode} episodes with avg reward {avg_reward}")
								break

map_data=[[1 for j in range(game.y_res)] for i in range(game.x_res)]
user_loc={'x':int(game.x_res/2),'y':int(game.y_res/2)}
ball_list = game.ball_list_init(game.ball_list)
map_data,reward,done=game.mapping2map(user_loc,ball_list,map_data,1)
next_roi_data = game.roi_calculation(map_data,user_loc,game.input_size)
os.system('clear')
game.game_print(map_data,next_roi_data)
while True:
	next_roi_data=game.roi_calculation(map_data,user_loc,game.input_size)
	o_dqn(next_roi_data,map_data,user_loc,ball_list)
	action = np.argmax(mainDQN.predict(next_roi_data))
	game.step(action,game.input_size,map_ata,user_loc,ball_list,1,0)
	time.sleep(delay_time)

