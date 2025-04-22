# rl-hw2/envs/pong_env.py

import gymnasium as gym
import cv2
import numpy as np
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

class PongEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make("ALE/PongNoFrameskip-v4", render_mode=render_mode)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
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