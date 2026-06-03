import os
from pathlib import Path
import torch

os.environ.setdefault("MUJOCO_GL", "egl")  # or "glfw" if you have a display

import numpy as np
import stable_worldmodel as swm

import gymnasium as gym
from stable_baselines3 import SAC



def get_world_model():
    world_model_ckpt_path = "/home/mamad/PhD/stable-worldmodel/scripts/examples/leworld_fetch/"
    world_model = swm.wm.utils.load_pretrained(name="fetch_reach_lewm/weights_epoch_100.pt", cache_dir=world_model_ckpt_path)

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
        print(f"obs_pixels::shape: {obs_pixels.shape}")
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

# This is written by AI: rewrite this to be easier to understand
# The action sequence should be paddes with action repeats to match the history size
def rollout_leworld_model(world_model, trajectory):
    w_trajectory = []
    device = next(world_model.parameters()).device
    dtype = next(world_model.encoder.parameters()).dtype
    
    with torch.inference_mode():
        for transition in trajectory:
            obs_pixels = transition["pixels"]  # (H,W,C) uint8 numpy
            action = transition["action"]  # (action_dim,) numpy

            pixels = torch.from_numpy(obs_pixels).permute(2, 0, 1)  # (C,H,W)
            pixels = pixels.unsqueeze(0).unsqueeze(0)  # (B=1,T=1,C,H,W)
            pixels = pixels.to(device=device, dtype=dtype)

            act = torch.as_tensor(action, dtype=torch.float32).reshape(1, 1, -1)
            if act.size(-1) != world_model.action_encoder.input_dim:
                target_dim = int(world_model.action_encoder.input_dim)
                padded = torch.zeros((1, 1, target_dim), dtype=act.dtype)
                n = min(target_dim, act.size(-1))
                padded[..., :n] = act[..., :n]
                act = padded
            act = act.to(device=device)

            info = {"pixels": pixels, "action": act}
            w_trajectory.append(world_model.encode(info))
        
    return w_trajectory

def main():
    world_model = get_world_model()
    world_model.eval()
    rl_agent = load_rl_policy()
    trajectory = rollout_rl_policy(rl_agent)
    w_trajectory = rollout_leworld_model(world_model, trajectory)
    print(f"w_trajectory: {w_trajectory}")


if __name__ == "__main__":
    main()