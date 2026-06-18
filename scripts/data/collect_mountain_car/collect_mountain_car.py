import os
from pathlib import Path

from omegaconf import OmegaConf

# EGL supports headless rgb_array rendering across many in-process envs; GLFW segfaults.
os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm

from expert_policy import ExpertPolicy

@hydra.main(version_base=None, config_path='../config', config_name='mountain_car')
def run(cfg):
    """Collect random trajectories from the Mountain Car environment."""

    world = swm.World(cfg.env_name, **cfg.world)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)

    world.set_policy(
        ExpertPolicy(
            ckpt_path=cfg.ckpt_path,
            env_id=cfg.env_name,
            device=cfg.get("device", "cuda"),
            render_mode=cfg.get("render_mode", ""),
        )
    )

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'mountain_car_expert_rl_agent.lance',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    world.close()
    logging.success('Completed data collection for mountain car expert')

if __name__ == "__main__":
    run()