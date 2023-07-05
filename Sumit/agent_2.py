import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import random
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.99  # Exploration rate decay
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.gamma = 0.99  # Discount factor
        self.batch_size = 32  # Number of experiences to sample for replay
        self.memory = []  # Experience replay memory

        # Initialize the Q-networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def act(self, state):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.Tensor(state))
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Store experiences in the replay memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.Tensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        dones = torch.Tensor(dones)

        # Compute Q-values for current and next states
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Update the Q-network
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update the target network with the weights from the Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        # Decay the exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
