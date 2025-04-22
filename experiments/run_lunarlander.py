# rl-hw2/experiments/run_lunarlander.py

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent


def train_lunarlander():
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n)

    episodes = 500
    update_freq = 10
    best_reward = float('-inf')
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

        if ep % update_freq == 0:
            agent.update_target()

        rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        print(f"[LUNAR] Episode {ep} - Reward: {total_reward:.2f} - Best: {best_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    env.close()
    agent.save("lunarlander_dqn.pth")

    plt.plot(rewards)
    plt.title("DQN LunarLander Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_lunarlander()
