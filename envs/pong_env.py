# rl-hw2/envs/pong_env.py

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack

class PongEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make("ALE/PongNoFrameskip-v4", render_mode=render_mode)
        self.env = AtariPreprocessing(
            self.env,
            grayscale_obs=True,
            screen_size=84,
            frame_skip=1,
            scale_obs=False
        )
        self.env = FrameStack(self.env, num_stack=4)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs, info = self.env.reset()
        return np.array(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.array(obs), reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()