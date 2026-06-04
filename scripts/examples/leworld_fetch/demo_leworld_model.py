import os
from pathlib import Path
import torch

os.environ.setdefault("MUJOCO_GL", "egl")  # or "glfw" if you have a display

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm

import gymnasium as gym
from stable_baselines3 import SAC
from torchvision.transforms import v2 as transforms

from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver


def img_transform(img_size: int = 224, dtype=torch.float32):
    """Match LeWM training (ToImage + ImageNet norm + resize)."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_world():
    world = swm.World("swm/FetchReachDict-v3", num_envs=8, image_shape=(224, 224))
    return world


def get_world_model(device: str = "cuda"):
    world_model_ckpt_path = "/home/mamad/PhD/stable-worldmodel/scripts/examples/leworld_fetch/"
    world_model = swm.wm.utils.load_pretrained(
        name="fetch_reach_lewm/weights_epoch_100.pt",
        cache_dir=world_model_ckpt_path,
    )
    world_model = world_model.to(device).eval()
    world_model.requires_grad_(False)
    return world_model

def load_rl_policy():
    CKPT = "/home/mamad/PhD/stable-worldmodel/scripts/expert/policies/swm_FetchReachDict-v3_expert/best_model.zip"
    env = gym.make("swm/FetchReachDict-v3")
    policy = SAC.load(CKPT, env=env)
    return policy


def rollout_rl_policy(policy):
    trajectory = []
  
    env = gym.make("swm/FetchReachDict-v3",render_mode="rgb_array")
    obs, info = env.reset()
    done = False

    while not done:
        obs_pixels = env.render()
        # print(f"obs_pixels::shape: {obs_pixels.shape}")
        action, _ = policy.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transitions = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "pixels": obs_pixels
        }
        trajectory.append(transitions)
        obs = next_obs
    env.close()

    return trajectory


def plan(world_model, world, img_size: int = 224):
    transform = {
        'pixels': img_transform(img_size),
        'goal': img_transform(img_size),
    }

    # model predictive control
    solver = CEMSolver(model=world_model, num_samples=300, device='cuda')
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=10,
            receding_horizon=5,
            action_block=5,  # dataset frameskip; action_encoder input_dim=20
        ),
        transform=transform,
    )

    world.set_policy(policy)
    results = world.evaluate(episodes=10, seed=0)

    print(f"Success Rate: {results['success_rate']:.1f}%")


def main():
    world = get_world()
    world_model = get_world_model()
    # world_model.eval()
    # rl_agent = load_rl_policy()
    # trajectory = rollout_rl_policy(rl_agent)

    plan(world_model, world)
 
  


if __name__ == "__main__":
    main()