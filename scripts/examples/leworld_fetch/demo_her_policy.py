import os
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")  # or "glfw" if you have a display

import numpy as np
import stable_worldmodel as swm

import gymnasium as gym
from stable_baselines3 import SAC

CKPT = "/home/mamad/PhD/stable-worldmodel/scripts/expert/policies/swm_FetchReachDict-v3_expert/best_model.zip"
_HER_KEYS = ("observation", "achieved_goal", "desired_goal")


def main():
    env_id = "swm/FetchReachDict-v3"
    # env = gym.make(env_id)
    env = gym.make(env_id, render_mode="rgb_array")

    policy = SAC.load(CKPT, env=env)
    obs, info = env.reset()
    print(f"obs keys: {list(obs.keys())}")

    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        print(f"action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        done = terminated or truncated
        print(f"done: {done}")
        print("\n\n\n")
    env.close()

if __name__ == "__main__":
    main()
