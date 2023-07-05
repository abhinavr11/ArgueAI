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
    st = env.reset()
    state=st[0]
    total_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        # Choose an action using epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take the chosen action and observe the next state and reward
        # Take the chosen action and observe the next state and reward
        result = env.step(action)
        next_state=result[0]
        reward=result[1]
        done=result[3]
        #info=result[3]



        # Update the Q-table
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

        total_reward += reward
        state = next_state

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print the episode information
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}")

# Evaluate the learned policy
total_rewards = 0
num_eval_episodes = 100

for _ in range(num_eval_episodes):
    st = env.reset()
    state=st[0]
    done = False

    while not done:
        action = np.argmax(Q[state, :])
        result = env.step(action)
        next_state=result[0]
        reward=result[1]
        done=result[3]
        #state, reward, done, _ = env.step(action)
        total_rewards += reward

average_reward = total_rewards / num_eval_episodes
print(f"Average reward over {num_eval_episodes} evaluation episodes: {average_reward}")
