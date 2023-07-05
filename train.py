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
import pickle
import os

from Prosecutor_final import *
from Defence_final import *

env=CourtRoomEnvironment()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions from environmet
n_actions = len(env.action_space)
# Get the number of state observations
#state = env.reset()

# Define the file path for storing the variable value
file_path = "memory/steps_done_value.txt"

# To load the last value of steps_done variable
if os.path.exists(file_path):
    # If the file exists, read the value from it
    with open(file_path, "r") as file:
        steps_done = int(file.read())
else:
    # If the file doesn't exist, initialize the variable
    steps_done = 0

prosecutor=Prosecutor_Agent(n_actions, steps_done)
defence=Defence_Agent(n_actions, steps_done)
#memory = ReplayMemory(10000)

'''if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50'''

# Load prosecutor model parameters and replay memory if they exist
try:
    prosecutor.policy_net.load_state_dict(torch.load('memory/prosecutor_policy_net.pth'))
    prosecutor.target_net.load_state_dict(torch.load('memory/prosecutor_target_net.pth'))
    with open('memory/prosecutor_replay_memory.pkl', 'rb') as f:
        prosecutor.memory = pickle.load(f)
    print("Loaded prosecutor's previous model parameters and replay memory.")
except FileNotFoundError:
    print("No previous model parameters and replay memory found for prosecutor. Starting from scratch.")

# Load defence model parameters and replay memory if they exist
try:
    defence.policy_net.load_state_dict(torch.load('memory/defence_policy_net.pth'))
    defence.target_net.load_state_dict(torch.load('memory/defence_target_net.pth'))
    with open('memory/defence_replay_memory.pkl', 'rb') as f:
        defence.memory = pickle.load(f)
    print("Loaded defence's previous model parameters and replay memory.")
except FileNotFoundError:
    print("No previous model parameters and replay memory found for defence. Starting from scratch.")



while True:
    try:
        # Initialize the environment and get it's state
        # state = [i_episode]th entry of the csv file which contains case contexts and the required output which the agent should give
        state, defence_state=env.reset()
        #defence_state=defence_state+ state
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        defence_state = torch.tensor(defence_state, dtype=torch.float32, device=device).unsqueeze(0)
    
        for t in range(3):
            prosecutor_action = prosecutor.select_action(state)
            defence_action = defence.select_action(defence_state)

            prosecutor_reward, defence_reward, done = env.step(prosecutor_action.item(), defence_action.item())

            '''observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated'''
            
            #reward=0 # if action == corresponding rule then reward
            #will implement after creating the csv file

            # Store the transition in memory
            prosecutor.memory.push(state, prosecutor_action, prosecutor_reward)

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

            # Store the transition in memory
            defence.memory.push(state, defence_action, defence_reward)
            # Perform one step of the optimization (on the policy network)
            defence.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict_defence = defence.target_net.state_dict()
            policy_net_state_dict_defence = defence.policy_net.state_dict()
            for key in policy_net_state_dict_defence:
                target_net_state_dict_defence[key] = policy_net_state_dict_defence[key]*defence.TAU + target_net_state_dict_defence[key]*(1-defence.TAU)
            defence.target_net.load_state_dict(target_net_state_dict_defence)

            if done:
                break
        pass
    except KeyboardInterrupt:
        # Save prosecutor model parameters
        torch.save(prosecutor.policy_net.state_dict(), 'memory/prosecutor_policy_net.pth')
        torch.save(prosecutor.target_net.state_dict(), 'memory/prosecutor_target_net.pth')

        # Save prosecutor replay memory
        with open('memory/prosecutor_replay_memory.pkl', 'wb') as f:
            pickle.dump(prosecutor.memory, f)
        
        # Save defence model parameters
        torch.save(defence.policy_net.state_dict(), 'memory/defence_policy_net.pth')
        torch.save(defence.target_net.state_dict(), 'memory/defence_target_net.pth')

        # Save defence replay memory
        with open('memory/defence_replay_memory.pkl', 'wb') as f:
            pickle.dump(defence.memory, f)
        
        with open(file_path, "w") as file:
            file.write(str(prosecutor.steps_done))
        break



'''print('Complete')
#plot_durations(show_result=True)
plt.ioff()
plt.show()'''