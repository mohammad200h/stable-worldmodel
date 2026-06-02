import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # or "glfw" if you have a display

import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy

CKPT = "/home/mamad/PhD/stable-worldmodel/scripts/expert/policies/swm_FetchReachDict-v3_expert/best_model.zip"
ENV_ID = "swm/FetchReachDict-v3"
DATASET = Path( "./datasets/fetch_reach_expert.lance")
print(f"Dataset path: {DATASET}")

_HER_KEYS = ("observation", "achieved_goal", "desired_goal")


class FetchSB3ExpertPolicy(BasePolicy):
    def __init__(self, ckpt_path: str, env_id: str, device: str = "cuda"):
        super().__init__()
        import gymnasium as gym
        from stable_baselines3 import SAC

        # HER-trained checkpoints need a matching env at load time (replay buffer setup).
        load_env = gym.make(env_id)
        self.model = SAC.load(ckpt_path, env=load_env, device=device)
        load_env.close()
        self.type = "expert"

    def get_action(self, info_dict, **kwargs):
        obs = {
            k: np.asarray(info_dict[k]).squeeze(axis=1).astype(np.float32)
            for k in _HER_KEYS
        }
        actions, _ = self.model.predict(obs, deterministic=True)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)


world = swm.World(
    ENV_ID,
    num_envs=8,
    image_shape=(224, 224),
    max_episode_steps=50,
)
world.set_policy(FetchSB3ExpertPolicy(CKPT, ENV_ID, device="cuda"))
world.collect(str(DATASET), episodes=1000, seed=0)

results = world.evaluate(episodes=50, seed=0, video="./fetch_expert_videos/")
print(f"Success rate: {results['success_rate']:.1f}%")

world.close()