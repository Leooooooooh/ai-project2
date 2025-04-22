# rl-hw2/experiments/run_pong.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from envs.pong_env import PongEnv
import matplotlib.pyplot as plt

class CNN_DQN(nn.Module):
    def __init__(self, action_dim):
        super(CNN_DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = x / 255.0  # normalize pixel values
        return self.net(x)

class PongAgent:
    def __init__(self, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN_DQN(action_dim).to(self.device)
        self.target = CNN_DQN(action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def store(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.model.state_dict())

def train_pong():
    env = PongEnv()
    agent = PongAgent(env.action_space.n)

    episodes = 500
    update_freq = 10
    best_reward = float('-inf')
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store((state, action, reward, next_state, float(done)))
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        if ep % update_freq == 0:
            agent.update_target()

        rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        print(f"[PONG] Episode {ep} - Reward: {total_reward:.2f} - Best: {best_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    env.close()
    agent.save("pong_dqn.pth")

    plt.plot(rewards)
    plt.title("CNN-DQN Pong Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_pong()