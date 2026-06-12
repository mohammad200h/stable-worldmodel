from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
from stable_baselines3 import SAC
from typing import Any



from stable_worldmodel.policy import BasePolicy

PATH = (
    Path(__file__).resolve().parents[2]
    / "expert/train_fetch/policies/swm_FetchReachDict-v3_expert/best_model.zip"
)
ENV_ID = "swm/FetchReachDict-v3"
_HER_KEYS = ("observation", "achieved_goal", "desired_goal")


class ExpertPolicy(BasePolicy):
    def __init__(self, ckpt_path: str | Path, env_id: str, 
        device: str = "cuda", render_mode: str = ""):
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        # HER checkpoints need a matching env at load time; close it after loading.
        load_env = gym.make(env_id, render_mode=render_mode)
        self._model = SAC.load(self.ckpt_path, env=load_env, device=self.device)
        load_env.close()
        self.env = None


    def _her_obs_from_info(self, info: dict) -> dict[str, np.ndarray]:
        if all(k in info for k in _HER_KEYS):
            obs = {
                k: np.asarray(info[k], dtype=np.float32) for k in _HER_KEYS
            }
            # World stacks a history dim: (n_envs, 1, dim) -> (n_envs, dim)
            if obs["observation"].ndim == 3 and obs["observation"].shape[1] == 1:
                obs = {k: v.squeeze(axis=1) for k, v in obs.items()}
            return obs

        state = np.asarray(info["state"], dtype=np.float32)
        return {
            "observation": state[..., :-3],
            "achieved_goal": state[..., -3:],
            "desired_goal": np.asarray(info["goal_state"], dtype=np.float32),
        }

    def get_action(self, info: dict) -> np.ndarray:
        obs = self._her_obs_from_info(info)
        action, _ = self._model.predict(obs, deterministic=True)
        return action





def main():
    print(f"#######10 episodes using SAC model########")

    env = gym.make(ENV_ID, render_mode="human")
    expert = ExpertPolicy(ckpt_path=PATH, env_id=ENV_ID, device="cuda")
    for i in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = expert.get_action(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()

if __name__ == "__main__":
    main()