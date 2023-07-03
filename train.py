import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from env_final import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Prosecutor_final import *
from Defence_final import *

env=CourtRoomEnvironment()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions from environmet
n_actions = len(env.action_space)
# Get the number of state observations
#state = env.reset()

prosecutor=Prosecutor_Agent(n_actions)
defence=Defence_Agent(n_actions)
#memory = ReplayMemory(10000)

'''if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50'''

while True:
    # Initialize the environment and get it's state
    # state = [i_episode]th entry of the csv file which contains case contexts and the required output which the agent should give
    state, defence_state=env.reset()
    defence_state=defence_state+ state
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    defence_state = torch.tensor(defence_state, dtype=torch.float32, device=device).unsqueeze(0)
   
    for t in range(3):
        action = prosecutor.select_action(state)
        #decode the action and show prosecutor's argument
        reward, done1 = env.step(action.item())

        '''observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated'''
        
        #reward=0 # if action == corresponding rule then reward
        #will implement after creating the csv file

        # Store the transition in memory
        prosecutor.memory.push(state, action, reward)

        # Move to the next state
        #state = next_state

        # Perform one step of the optimization (on the policy network)
        prosecutor.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict_prosecutor = prosecutor.target_net.state_dict()
        policy_net_state_dict_prosecutor = prosecutor.policy_net.state_dict()
        for key in policy_net_state_dict_prosecutor:
            target_net_state_dict_prosecutor[key] = policy_net_state_dict_prosecutor[key]*prosecutor.TAU + target_net_state_dict_prosecutor[key]*(1-prosecutor.TAU)
        prosecutor.target_net.load_state_dict(target_net_state_dict_prosecutor)

        ###################################  Defence Agent  #################################

        action = defence.select_action(defence_state)
        #decode the action and show defence's argument
        reward, done2 = env.step(action)

        # Store the transition in memory
        defence.memory.push(state, action, reward)
        # Perform one step of the optimization (on the policy network)
        defence.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict_defence = defence.target_net.state_dict()
        policy_net_state_dict_defence = defence.policy_net.state_dict()
        for key in policy_net_state_dict_defence:
            target_net_state_dict_defence[key] = policy_net_state_dict_defence[key]*defence.TAU + target_net_state_dict_defence[key]*(1-defence.TAU)
        defence.target_net.load_state_dict(target_net_state_dict_defence)

        if done1 & done2:
            break

'''print('Complete')
#plot_durations(show_result=True)
plt.ioff()
plt.show()'''