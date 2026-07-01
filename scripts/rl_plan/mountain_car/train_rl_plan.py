import os
import sys

if sys.platform == 'linux':
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import argparse
from pathlib import Path

import gymnasium as gym
from omegaconf import OmegaConf
import stable_worldmodel  # registers swm/* envs with Gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None

from env import register_mountain_car_wm_env

register_mountain_car_wm_env()

_SCRIPT_DIR = Path(__file__).resolve().parent
_WM_ENV_ID = 'swm/MountainCarWM-v0'
_WM_CONFIGS = {
    'pixels': _SCRIPT_DIR / 'rl_pixals_worldmodel.yaml',
    'state': _SCRIPT_DIR / 'rl_state_worldmodel.yaml',
}

from callbacks import EntropyDecayCallback, MountainCarEvalCallback

def _load_wm_yaml(input_state_type: str) -> dict:
    config_path = _WM_CONFIGS[input_state_type]
    return OmegaConf.to_container(OmegaConf.load(config_path))


def _resolve_wm_settings(
    input_state_type: str,
    world_model_path: str | None,
    checkpoint: str | None,
    embedding_is_made_of_pixels: bool | None = None,
) -> tuple[str, str, bool]:
    wm_cfg = _load_wm_yaml(input_state_type)
    path = world_model_path or wm_cfg['world_model_path']
    ckpt = checkpoint or wm_cfg['checkpoint']
    emb_pixels = (
        embedding_is_made_of_pixels
        if embedding_is_made_of_pixels is not None
        else wm_cfg['embedding_is_made_of_pixels']
    )
    return path, ckpt, emb_pixels


def _make_env(
    env_id: str,
    world_model_path: str | None,
    checkpoint: str | None,
    embedding_is_made_of_pixels: bool = True,
    hybrid_mode: bool = True,
):
    if env_id == _WM_ENV_ID:
        if world_model_path is None or checkpoint is None:
            raise ValueError(
                f"'{env_id}' requires world model path and checkpoint "
                f'(from --input-state-type yaml or --wm-path / --checkpoint).'
            )
        return gym.make(
            env_id,
            world_model_path=world_model_path,
            checkpoint=checkpoint,
            embedding_is_made_of_pixels=embedding_is_made_of_pixels,
            hybrid_mode=hybrid_mode,
        )
    return gym.make(env_id)


def train_expert(
    env_id: str,
    total_timesteps: int,
    seed: int = 37,
    track: bool = False,
    project_name: str = 'stable-worldmodel-rl-plan',
    wandb_name: str | None = None,
    model_name: str | None = None,
    n_steps: int = 2048,
    initial_ent_coef: float = 0.3,
    final_ent_coef: float = 0.001,
    entropy_decay_steps: int = 100_000,
    entropy_decay_start: int = 100_000,
    input_state_type: str = 'pixels',
    world_model_path: str | None = None,
    checkpoint: str | None = None,
    embedding_is_made_of_pixels: bool = True,
    hybrid_mode: bool = True,
):
    """
    Train PPO on Mountain Car with world-model latent observations.

    Uses swm/MountainCarWM-v0, which wraps swm/MountainCarContinuousControl-v0
    and exposes a flat Box embedding as the observation.
    """
    print('===================================================')
    print(f' Training Expert Policy for {env_id}')
    print(f' Setup: PPO | {total_timesteps} Timesteps | Seed: {seed}')
    print(f' WM input: {input_state_type} (pixels={embedding_is_made_of_pixels})')
    print(f' Hybrid mode: {hybrid_mode}')
    print('===================================================')

    env = Monitor(
        _make_env(
            env_id,
            world_model_path,
            checkpoint,
            embedding_is_made_of_pixels,
            hybrid_mode,
        )
    )
    eval_env = Monitor(
        _make_env(
            env_id,
            world_model_path,
            checkpoint,
            embedding_is_made_of_pixels,
            hybrid_mode,
        )
    )

    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        ent_coef=initial_ent_coef,
        verbose=1,
        seed=seed,
        tensorboard_log=(
            f'./logs/tensorboard/{env_id.replace("/", "_")}_ppo/'
        ),
    )

    save_path = (
        f'./policies/{model_name}'
        if model_name
        else f'./policies/{env_id.replace("/", "_")}_expert'
    )
    os.makedirs(save_path, exist_ok=True)

    eval_callback = MountainCarEvalCallback(
        eval_env,
        video_folder=f'{save_path}/videos',
        log_video_to_wandb=track,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )
    entropy_decay_callback = EntropyDecayCallback(
        initial_ent_coef=initial_ent_coef,
        final_ent_coef=final_ent_coef,
        decay_steps=entropy_decay_steps,
        start_after_timesteps=entropy_decay_start,
    )

    callbacks = [eval_callback, entropy_decay_callback]

    if track and wandb is None:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        )

    if track:
        wandb.init(
            project=project_name,
            name=wandb_name or f'PPO_{env_id.replace("/", "_")}',
            config={
                'env': env_id,
                'algo': 'PPO',
                'seed': seed,
                'timesteps': total_timesteps,
                'n_steps': n_steps,
                'initial_ent_coef': initial_ent_coef,
                'final_ent_coef': final_ent_coef,
                'entropy_decay_steps': entropy_decay_steps,
                'entropy_decay_start': entropy_decay_start,
                'input_state_type': input_state_type,
                'embedding_is_made_of_pixels': embedding_is_made_of_pixels,
                'world_model_path': world_model_path,
                'checkpoint': checkpoint,
                'hybrid_mode': hybrid_mode,
                'model_name': model_name,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            model_save_path=save_path, model_save_freq=5000, verbose=2
        )
        callbacks.append(wandb_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(f'{save_path}/final_model')

    if track:
        wandb.finish()

    print(f'Training complete. Models saved to {save_path}')
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Train a PPO expert on Mountain Car with world-model latent '
            'observations (swm/MountainCarWM-v0).'
        )
    )
    parser.add_argument(
        '--env',
        type=str,
        default=_WM_ENV_ID,
        help=f'Env id (default: {_WM_ENV_ID}).',
    )
    parser.add_argument(
        '--input-state-type',
        type=str,
        default='pixels',
        choices=['pixels', 'state'],
        help=(
            'World model input type: pixels loads rl_pixals_worldmodel.yaml, '
            'state loads rl_state_worldmodel.yaml.'
        ),
    )
    parser.add_argument(
        '--wm-path',
        type=str,
        default=None,
        help=(
            f'Override world model checkpoint directory for {_WM_ENV_ID} '
            f'(default from --input-state-type yaml)'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            f'Override checkpoint filename inside --wm-path for {_WM_ENV_ID} '
            f'(default from --input-state-type yaml)'
        ),
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1_000_000,
        help='Total environment steps to execute',
    )
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2048,
        help='PPO rollout length per update',
    )
    parser.add_argument(
        '--track',
        action='store_true',
        help='Log training metrics natively to Weights & Biases',
    )
    parser.add_argument(
        '--project',
        type=str,
        default='stable-worldmodel',
        help='WandB Cloud project name',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Policy save directory name under ./policies/ (default: env-based)',
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='WandB run name (default: PPO_<env>)',
    )
    parser.add_argument(
        '--embedding-is-made-of-pixels',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Whether WM embeddings come from pixels (default: from yaml)',
    )
    parser.add_argument(
        '--hybrid-mode',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Use original env reward/done with WM observations (default: true). '
            'Set --no-hybrid-mode to use WM reward and continue heads.'
        ),
    )

    args = parser.parse_args()

    wm_path, ckpt, emb_pixels = _resolve_wm_settings(
        args.input_state_type,
        args.wm_path,
        args.checkpoint,
        args.embedding_is_made_of_pixels,
    )

    train_expert(
        args.env,
        args.timesteps,
        args.seed,
        args.track,
        args.project,
        wandb_name=args.wandb_name,
        model_name=args.model_name,
        n_steps=args.n_steps,
        input_state_type=args.input_state_type,
        world_model_path=wm_path,
        checkpoint=ckpt,
        embedding_is_made_of_pixels=emb_pixels,
        hybrid_mode=args.hybrid_mode,
    )
