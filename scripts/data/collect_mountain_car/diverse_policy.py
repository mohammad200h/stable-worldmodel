from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # registers swm/* envs with Gymnasium
from stable_baselines3 import PPO

from stable_worldmodel.policy import BasePolicy

# SB3 appends ".zip" when loading; pass the path without the suffix.
PATH = (
    Path(__file__).resolve().parents[2]
    / "expert/train_mountain_car/policy_checkpoints/"
)
ENV_ID = "swm/MountainCarContinuousControl-v0"


class CollectionOfPolicies(BasePolicy):
    def __init__(
        self,
        policies_path: str | Path,
        env_id: str,
        device: str = "cuda",
        render_mode: str = "",
    ):
        self.policies_path = Path(policies_path)
        self.env_id = env_id
        self.device = device
        self.render_mode = render_mode
        self._remaining = self._get_policy_names()
        self._num_policies = len(self._remaining)
        if self._num_policies == 0:
            raise ValueError(
                f'No policy checkpoints found in {self.policies_path}'
            )
        self._current_policy_name = None
        self.current_policy_path = self._get_next_policy_path()
        load_env = gym.make(env_id, render_mode=render_mode)
        self._model = self._load_policy(self.current_policy_path, load_env)
        load_env.close()
        self.env = None

    def get_action(self, info: dict) -> np.ndarray:
        obs = np.asarray(info["state"], dtype=np.float32)
        # World stacks a history dim: (n_envs, 1, dim) -> (n_envs, dim)
        if obs.ndim == 3 and obs.shape[1] == 1:
            obs = obs.squeeze(axis=1)
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def _get_policy_names(self) -> list[str]:
        return sorted(p.stem for p in self.policies_path.glob("*.zip"))

    def _get_next_policy_path(self) -> Path:
        self._current_policy_name = self._remaining.pop(0)
        return self.policies_path / f"{self._current_policy_name}.zip"

    def _load_policy(self, checkpoint_path: str | Path, env: gym.Env):
        return PPO.load(checkpoint_path, env=env, device=self.device)

    def change_policy(self) -> None:
        if not self._remaining:
            raise RuntimeError('No more policies to switch to.')
        self.current_policy_path = self._get_next_policy_path()
        load_env = gym.make(self.env_id, render_mode=self.render_mode)
        self._model = self._load_policy(self.current_policy_path, load_env)
        load_env.close()

    @property
    def num_policies(self) -> int:
        return self._num_policies
    
    @property
    def current_policy_name(self) -> str:
        return self._current_policy_name


def main():
    print("#######2 episodes per policy using PPO models########")

    env = gym.make(ENV_ID, render_mode="human")
    expert = CollectionOfPolicies(
        policies_path=PATH, env_id=ENV_ID, device="cuda"
    )

    for j in range(expert.num_policies):
        if j > 0:
            expert.change_policy()
        for i in range(2):
            obs, info = env.reset()
            print(f"info: {info}")
            done = False
            while not done:
                action = expert.get_action(info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"policy name: {expert.current_policy_name}")
                print(f"reward: {reward}")
                print(f"terminated: {terminated}")
                print(f"truncated: {truncated}")
                print(f"done: {done}")
                print(f"obs: {obs}")
                print(f"info: {info}")
                print(f"action: {action}")
                print("--------------------------------")
    env.close()


if __name__ == "__main__":
    main()
