# rl-hw2/experiments/run_cartpole.py

import gymnasium as gym
import numpy as np
from agents.dqn import DQNAgent
from agents.policy_gradient import PGAgent
import matplotlib.pyplot as plt
from videos.video import record_video 
import torch

def train_dqn():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n)

    episodes = 500
    update_freq = 10
    best_reward = 0
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store((state, action, reward, next_state, float(done)))
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)

        if ep % update_freq == 0:
            agent.update_target()

        if total_reward > best_reward:
            best_reward = total_reward

        print(f"[DQN] Episode {ep} - Reward: {total_reward:.2f} - Best: {best_reward:.2f}")

    env.close()
    agent.save("cartpole_dqn.pth")
    record_video("CartPole-v1", agent, video_name="cartpole_dqn_demo")

    plt.plot(rewards)
    plt.title("DQN CartPole Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

def train_pg():
    env = gym.make("CartPole-v1")
    agent = PGAgent(state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.n)

    episodes = 500
    best_reward = 0
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.train()
        rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward

        print(f"[PG] Episode {ep} - Reward: {total_reward:.2f} - Best: {best_reward:.2f}")

    env.close()
    agent.save("cartpole_pg.pth")
    record_video("CartPole-v1", agent, video_name="cartpole_pg_demo")

    plt.plot(rewards)
    plt.title("Policy Gradient CartPole Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

def rule_based():
    env = gym.make("CartPole-v1")
    episodes = 10
    best_reward = 0
    rewards = []
    actions = []

    class RuleBasedAgent:
        def select_action(self, state):
            pole_angle = state[2]
            return 1 if pole_angle > 0 else 0

    agent = RuleBasedAgent()

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        print(f"[RuleBased] Episode {ep} - Reward: {total_reward:.2f} - Best: {best_reward:.2f}")

    env.close()
    record_video("CartPole-v1", agent, video_name="cartpole_rulebased_demo")

    plt.plot(rewards)
    plt.title("Rule-Based CartPole Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_dqn()
    train_pg()
    rule_based()
