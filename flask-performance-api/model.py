import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 2. Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# 3. Epsilon-Greedy Policy
def select_action(state, model, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        state = torch.tensor([state], dtype=torch.float32)
        q_values = model(state)
        return q_values.max(1)[1].item()

# 4. Training Function
def train(model, target_model, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    # Q(s, a)
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

    # Max Q'(s', a') from target network
    next_q_values = target_model(next_states).max(1)[0]

    # Expected Q values
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Loss
    loss = F.mse_loss(q_values, expected_q_values.detach())

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Main Training Loop
def main():
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    hidden_dim = 128

    model = DQN(input_dim, hidden_dim, output_dim)
    target_model = DQN(input_dim, hidden_dim, output_dim)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = ReplayBuffer(10000)

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 64
    episodes = 500
    target_update = 10

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        for t in range(200):
            action = select_action(state, model, epsilon, env.action_space)
            next_state, reward, done, truncated, _ = env.step(action)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            train(model, target_model, memory, optimizer, batch_size, gamma)

            if done:
                break

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    main()
