from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # registers swm/* envs with Gymnasium
from stable_baselines3 import PPO

from stable_worldmodel.policy import BasePolicy

# SB3 appends ".zip" when loading; pass the path without the suffix.
PATH = (
    Path(__file__).resolve().parents[2]
    / "expert/train_mountain_car/best_model/best_model"
)
ENV_ID = "swm/MountainCarContinuousControl-v0"



class ExpertPolicy(BasePolicy):
    def __init__(self, ckpt_path: str | Path, env_id: str, 
        device: str = "cuda", render_mode: str = ""):

        self.ckpt_path = Path(ckpt_path)
        self.device = device
        # HER checkpoints need a matching env at load time; close it after loading.
        load_env = gym.make(env_id, render_mode=render_mode)
        self._model = PPO.load(self.ckpt_path, env=load_env,
             device=self.device
        )
        load_env.close()
        self.env = None

    def get_action(self, info: dict) -> np.ndarray:
        obs = np.asarray(info["state"], dtype=np.float32)
        # World stacks a history dim: (n_envs, 1, dim) -> (n_envs, dim)
        if obs.ndim == 3 and obs.shape[1] == 1:
            obs = obs.squeeze(axis=1)
        action, _ = self._model.predict(obs, deterministic=True)
        return action


def main():
    print(f"#######10 episodes using PPO model########")

    env = gym.make(ENV_ID, render_mode="human")
    expert = ExpertPolicy(ckpt_path=PATH, env_id=ENV_ID, device="cuda")
    for i in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = expert.get_action(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
            print(f"truncated: {truncated}")
            print(f"done: {done}")
            print(f"obs: {obs}")
            print(f"info: {info}")
            print(f"action: {action}")
            print(f"--------------------------------")
    env.close()


if __name__ == "__main__":
    main()