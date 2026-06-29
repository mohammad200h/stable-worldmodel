import os
from pathlib import Path

from omegaconf import OmegaConf

# EGL supports headless rgb_array rendering across many in-process envs; GLFW segfaults.
os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm

from diverse_policy import PATH as DEFAULT_POLICIES_PATH, CollectionOfPolicies


@hydra.main(version_base=None, config_path='../config', config_name='mountain_car')
def run(cfg):
    """Collect trajectories from a directory of PPO checkpoints."""

    world = swm.World(cfg.env_name, **cfg.world)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)

    policies_path = Path(cfg.policies_path or DEFAULT_POLICIES_PATH)
    episodes_per_policy = cfg.get('episodes_per_policy', 10)

    world.set_policy(
        CollectionOfPolicies(
            policies_path=policies_path,
            env_id=cfg.env_name,
            device=cfg.get("device", "cuda"),
            render_mode=cfg.get("render_mode", ""),
        )
    )

    world.collect_from_collection_of_policies(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / f'mountain_car_diverse_rl_agent_{cfg.episodes_per_policy}.lance',
        episodes_per_policy=episodes_per_policy,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    world.close()
    logging.success('Completed diverse data collection for mountain car')


if __name__ == "__main__":
    run()
