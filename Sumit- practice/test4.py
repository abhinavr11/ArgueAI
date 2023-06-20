import gym
import numpy as np

# Create the FrozenLake-v1 environment
env = gym.make('FrozenLake-v1')

# Set the hyperparameters
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

# Initialize the Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    print(state)