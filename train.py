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

env=CourtRoomEnvironment()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions from environmet
n_actions = len(env.action_space)
# Get the number of state observations
#state = env.reset()

agent=Prosecutor_Agent(n_actions)
#memory = ReplayMemory(10000)

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    # state = [i_episode]th entry of the csv file which contains case contexts and the required output which the agent should give
    state=env.Tokenize(state)
    #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    action = agent.select_action(state)
    '''observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated'''
    
    reward=0 # if action == corresponding rule then reward
    #will implement after creating the csv file

    

    # Store the transition in memory
    agent.memory.push(state, action, reward)

    # Move to the next state
    #state = next_state

    # Perform one step of the optimization (on the policy network)
    agent.optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = agent.target_net.state_dict()
    policy_net_state_dict = agent.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*agent.TAU + target_net_state_dict[key]*(1-agent.TAU)
    agent.target_net.load_state_dict(target_net_state_dict)

    '''if done:
        episode_durations.append(t + 1)
        plot_durations()
        break'''

print('Complete')
#plot_durations(show_result=True)
plt.ioff()
plt.show()