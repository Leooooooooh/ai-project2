# rl-hw2/agents/policy_gradient.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class PGAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.states.append(state)
        self.actions.append(action)
        return action.item()

    def train(self):
        G = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for state, action, R in zip(self.states, self.actions, returns):
            probs = self.model(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.reset()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))