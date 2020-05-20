# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:44:21 2020

@author: JLLU
"""

#from buffer import ReplayBuffer
#from maddpg import MADDPG, to_torch
import torch
import numpy as np
from unityagents import UnityEnvironment
#from utilities import transpose_list, transpose_to_tensor
from collections import deque
import matplotlib.pyplot as plt
import random
from lib import *

def seeding(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def random_move():
    for i in range(5):                                         # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    
    
#def to_torch(np_array_list):
#    return [torch.from_numpy(x).float().to(device) for x in np_array_list]

env = UnityEnvironment(file_name="Tennis_Windows_x86_64\Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

if False:
    random_move()    
    raise Exception 

    
def MADDPG_Inference(n_episodes=3, t_max=1000):
    state_size=env_info.vector_observations.shape[1]
    action_size=brain.vector_action_space_size
    num_agents=env_info.vector_observations.shape[0]
    MADDPG_obj = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents)
    MADDPG_obj.load_maddpg('2.7') #load the local network weights
    
    scores_list = []
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
        states = env_info.vector_observations                   # get the current states (for all agents)
        
        scores = np.zeros(num_agents)                          # initialize the score (for each agent in MADDPG_obj)
        for _ in range(t_max):
            actions = MADDPG_obj.act(states, i_episode=0, add_noise=False)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent in MADDPG_obj)
            rewards = env_info.rewards                         # get rewards (for each agent in MADDPG_obj)
            dones = env_info.local_done                        # see if episode finished
            scores += rewards                                  # update the score (for each agent in MADDPG_obj)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Episode {}: {}'.format(i_episode, scores))
        scores_list.append(np.max(scores))
    print('Mean score is: ', np.mean(np.array(scores_list)))

if False:
    MADDPG_Inference()




seeding(seed=55)  # 2 is default
state_size=env_info.vector_observations.shape[1]
action_size=brain.vector_action_space_size
num_agents=env_info.vector_observations.shape[0]
MADDPG_obj = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents)
###MADDPG_obj.load_maddpg() #load the local network weights


#Training
while True:
    n_episodes = 100000
    t_max = 1000
    scores_deque = deque(maxlen=100)
    scores_list = []
    scores_list_avg = []
    max_score = 0.0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                   # get the current states (for all agents)
        #MADDPG_obj.reset() #reset the MADDPG_obj OU Noise
        scores = np.zeros(num_agents)                          # initialize the score (for each agent in MADDPG)
        num_steps = 0
        for _ in range(t_max):
            actions = MADDPG_obj.act(states, i_episode)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent in MADDPG)
            rewards = env_info.rewards                         # get rewards (for each agent in MADDPG)
            dones = env_info.local_done                        # see if episode finished
            scores += rewards                                  # update the score (for each agent in MADDPG)
            MADDPG_obj.step(i_episode, states, actions, rewards, next_states, dones) #train the MADDPG_obj           
            states = next_states                               # roll over states to next time step
            num_steps += 1
            if np.any(dones):                                  # exit loop if episode finished
                break
            #print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))
        
        scores = np.max(scores)
        scores_deque.append(scores)
        scores_list.append(scores)
        scores_list_avg.append(np.mean(scores_deque))
        
        if (i_episode % 10) == 0:           
            print(f'Episode {i_episode}, Average Score: {np.mean(scores_deque):1.4f}, Current Score:{scores:1.4f}')
            if False:
                plt.figure()
                plt.plot(scores_list[:-70])
                plt.plot(scores_list_avg[:-70])
                plt.xlabel('Episode #')
                plt.legend(['Score', 'Avg Score'])
                plt.grid(True)
        
        #print('Noise Scaling: {}, Memory size: {} and Num Steps: {}'.format(MADDPG_obj.maddpg_agents[0].noise_level, len(MADDPG_obj.memory), num_steps))
        
        if scores > max_score and scores > 0.2:
            MADDPG_obj.save_maddpg(str(round(scores, 4)))
            max_score = scores
        
        if i_episode % 100 == 0:
            MADDPG_obj.save_maddpg()
            print('Saved Model: Episode {}\tAverage Score: {:1.4f}'.format(i_episode, np.mean(scores_deque)))
            
        #if np.mean(scores_deque) > 1.0 and len(scores_deque) >= 100:
        #    MADDPG_obj.save_maddpg()
        #    print('Saved Model: Episode {}\tAverage Score: {:1.4f}'.format(i_episode, np.mean(scores_deque)))
        #    break
            
    