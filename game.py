import os
import sys
import time
import random
import readchar 
import math

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STOP = 4
LEFT2 = 5
DOWN2 = 6
RIGHT2 = 7
UP2 = 8

arrow_keys = {
	'\x1b[A': UP,
	'\x1b[B': DOWN,
	'\x1b[C': RIGHT,
	'\x1b[D': LEFT}
input_size = 41
delay_time=0.05
x_res=50
y_res=50
ball_step=12
max_ball_num=18
ball_list=[]
speed_range=[-1,0,1]
rew=-2
print_list={100:88,200:79,1:96,250:64}

def map_clear(map_data):
	for x in range(x_res):
		for y in range(y_res):
			map_data[x][y]=1
	return map_data

def game_print(map_data,roi):
	for y in range(y_res):
		for x in range(x_res): 
			print '%c'%(print_list[map_data[x][y]]),
		print '|\n',
#	for y in range(input_size-10):
#		for x in range(input_size): 
#			print '%c'%(print_list[roi[x][y+5]]),
#		print '|\n',
	time.sleep(delay_time)	

def user_mapping(new_map_data,user_x,user_y):
	new_map_data[user_x][user_y]=250
	new_map_data[user_x+1][user_y+1]=250
	new_map_data[user_x+2][user_y+2]=250
	new_map_data[user_x][user_y+2]=250
	new_map_data[user_x+2][user_y]=250
	new_map_data[user_x+1][user_y]=250
	new_map_data[user_x+1][user_y+2]=250
	new_map_data[user_x][user_y+1]=250
	new_map_data[user_x+2][user_y+1]=250
	return new_map_data

def mapping2map(user_loc,ball_list,map_data,reward):
	new_map_data = map_clear(map_data)
	for i in range(max_ball_num-ball_step):
		ball_x=ball_list[i]['x']
		ball_y=ball_list[i]['y']
		new_map_data[ball_x+1][ball_y]=200
		new_map_data[ball_x][ball_y+1]=200
		new_map_data[ball_x+1][ball_y+2]=200
		new_map_data[ball_x+2][ball_y+1]=200
		new_map_data[ball_x+1][ball_y+1]=200
	user_x=user_loc['x']
	user_y=user_loc['y']
	if check_encounter(new_map_data,user_x,user_y) == 1:
		reward=rew
		done=True
	#	time.sleep(1)	
		os.system('clear')
	else :
		if reward == rew:
			done=True
	#		time.sleep(delay_time)	
			os.system('clear')
		else :
			done=False
	new_map_data = user_mapping(new_map_data,user_x,user_y)
	return new_map_data,reward,done


def game_step_ball(ball_list):
	temp_x=[0,x_res-3]
	temp_y=[0,y_res-3]
	temp_s=[-1,1]
	print ball_step
	for i in range(max_ball_num-ball_step): 
		x=ball_list[i]['x']
		y=ball_list[i]['y']
		x_speed=ball_list[i]['x_speed']
		y_speed=ball_list[i]['y_speed']
		if (ball_list[i]['new'] == 2) : 
			if random.randrange(0,2) == 0 :
				ball_list[i]['x']=random.randrange(0,x_res-2)
				ball_list[i]['y']=random.choice(temp_y)
				s_x=random.choice(speed_range)
				s_y=random.choice(speed_range)
				if s_x == 0 and s_y == 0:
					s_x = random.choice(temp_s)
					s_y = random.choice(temp_s)
				ball_list[i]['x_speed']=s_x
				ball_list[i]['y_speed']=s_y
				ball_list[i]['new']=1
			else :
				ball_list[i]['x']=random.choice(temp_x)
				ball_list[i]['y']=random.randrange(0,y_res-2)
				s_x=random.choice(speed_range)
				s_y=random.choice(speed_range)
				if s_x == 0 and s_y == 0:
					s_x = random.choice(temp_s)
					s_y = random.choice(temp_s)
				ball_list[i]['x_speed']=s_x
				ball_list[i]['y_speed']=s_y
				ball_list[i]['new']=1
		elif x+x_speed < 0 : 
			ball_list[i]['x']=0
			ball_list[i]['new']=2
		elif x+x_speed >= x_res-2 :
			ball_list[i]['x']=x_res-3
			ball_list[i]['new']=2
		elif y+y_speed < 0 : 
			ball_list[i]['y']=0
			ball_list[i]['new']=2
		elif y+y_speed >= y_res-2:
			ball_list[i]['y']=y_res-3
			ball_list[i]['new']=2
		else :		
			ball_list[i]['x']=x+x_speed
			ball_list[i]['y']=y+y_speed
			ball_list[i]['new']=0
	return ball_list

def game_step_user(user_loc,action):
	x,y=get_key(action)
	user_x = user_loc['x']
	user_y = user_loc['y']
	if x+user_x < 0 : 
		user_loc['x']=0
		reward=rew
	elif x+user_x >= x_res-2 :
		user_loc['x']=x_res-3
		reward=rew
	elif y+user_y < 0 : 
		user_loc['y']=0
		reward=rew
	elif y+user_y >= y_res-2 :
		user_loc['y']=y_res-3
		reward=rew
	else :		
		user_loc['x']=x+user_x
		user_loc['y']=y+user_y
		reward=1
	return user_loc,reward

def get_key(action):
	if action == UP:
		x = 0
		y = -2
	elif action == DOWN:
		x = 0
		y = 2
	elif action == LEFT:
		x = -2
		y = 0
	elif action == RIGHT:
		x = 2
		y = 0
	elif action == STOP:
		x = 0
		y = 0
	elif action == UP2:
		x = -1
		y = -1
	elif action == DOWN2:
		x = 1
		y = -1
	elif action == LEFT2:
		x = -1
		y = 1
	elif action == RIGHT2:
		x = 1
		y = 1 
	return x,y
	
def ball_list_init(ball_list):
	ball_list=[]
	temp_x=[0,4,x_res-7,x_res-3]
	temp_y=[0,4,y_res-7,y_res-3]
	temp_s=[-1,1]
	for i in range(max_ball_num):
		s_x=random.choice(speed_range)
		s_y=random.choice(speed_range)
		if s_x == 0 and s_y == 0:
			s_x = random.choice(temp_s)
			s_y = random.choice(temp_s)
		if random.choice(temp_s)==1 :
			ball={'num':i,'x':random.choice(temp_x),'y':random.randrange(0,y_res-2),'x_speed':s_x,'y_speed':s_y,'new':1}
		else :
			ball={'num':i,'x':random.randrange(0,x_res-2),'y':random.choice(temp_y),'x_speed':s_x,'y_speed':s_y,'new':1}
		ball_list.append(ball)
	return ball_list

def check_encounter(map_data,user_x,user_y):
	if map_data[user_x][user_y] == 200:
		return 1
	if map_data[user_x+1][user_y] == 200:
		return 1
	if map_data[user_x+2][user_y] == 200:
		return 1
	if map_data[user_x+1][user_y+1] == 200:
		return 1
	if map_data[user_x][user_y+1] == 200:
		return 1
	if map_data[user_x][user_y+2] == 200:
		return 1
	if map_data[user_x+2][user_y+1] == 200:
		return 1
	if map_data[user_x+1][user_y+2] == 200:
		return 1
	if map_data[user_x+2][user_y+2] == 200:
		return 1
	return 0				

def roi_calculation(map_data,user_loc,size):
	next_roi_data=[[1 for j in range(size)] for i in range(size)]
	user_x=user_loc['x']
	user_y=user_loc['y']
	t_x=user_x-(size-3)/2
	t_y=user_y-(size-3)/2
	for x in range(size):
		for y in range(size):
			t_x2=t_x+x
			t_y2=t_y+y
			if (t_x2 >= 0 and t_y2 >=0) and (t_x2 < x_res and t_y2 < y_res):
				next_roi_data[x][y]=map_data[t_x2][t_y2]
			else : 
				next_roi_data[x][y]=100
	return next_roi_data

def step(action,input_size,map_data,user_loc,ball_list,opt,epi):
	reward = 1
	map_data = map_clear(map_data)
	if opt == 1 :
		user_loc,reward=game_step_user(user_loc,action)
	ball_list = game_step_ball(ball_list)
	map_data,reward,done = mapping2map(user_loc,ball_list,map_data,reward)
	next_roi_data=roi_calculation(map_data,user_loc,input_size)
	if epi != -1:
		game_print(map_data,next_roi_data)
		os.system('clear')
	return next_roi_data,reward,done,ball_list,user_loc

