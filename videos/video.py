import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

def record_video(env_id, agent, video_name="demo", episode_len=1000):
    video_path = os.path.join("videos", video_name)
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_path, episode_trigger=lambda x: True)
    
    state, _ = env.reset()
    total_reward = 0

    for t in range(episode_len):
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    print(f"🎥 Video saved to: {video_path}")
    print(f"🏆 Total reward: {total_reward}")