import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import stable_worldmodel  # noqa: F401 — register gym envs

from pusht_callbacks import PushTEvalCallback
from pusht_env_wrappers import make_pushT_env, wrap_pushT_env


CONFIG_PATH = Path(__file__).resolve().parent / 'config.json'

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None


def load_config(path: Path | str | None = None) -> dict:
    config_path = Path(path) if path is not None else CONFIG_PATH
    with open(config_path) as f:
        return json.load(f)


def observation_channels(env_cfg: dict) -> int:
    base_channels = 1 if env_cfg['grayscale'] else 3
    return base_channels * env_cfg['frame_stack']


def estimate_replay_buffer_gb(
    buffer_size: int,
    *,
    n_channels: int,
    resolution: int,
) -> float:
    """Approximate RAM for SB3's image replay buffer (obs + next_obs, float32)."""
    bytes_per_transition = n_channels * resolution * resolution * np.dtype(np.float32).itemsize
    return buffer_size * bytes_per_transition * 2 / (1024**3)


def clamp_buffer_size(
    buffer_size: int,
    *,
    n_channels: int,
    resolution: int,
    max_gb: float,
) -> int:
    estimated_gb = estimate_replay_buffer_gb(
        buffer_size,
        n_channels=n_channels,
        resolution=resolution,
    )
    if estimated_gb <= max_gb:
        return buffer_size

    bytes_per_transition = (
        n_channels * resolution * resolution * np.dtype(np.float32).itemsize * 2
    )
    clamped = max(int(max_gb * (1024**3) / bytes_per_transition), 1000)
    print(
        'WARNING: '
        f'buffer_size={buffer_size} needs ~{estimated_gb:.1f} GB for the replay buffer; '
        f'clamping to {clamped} '
        f'(~{estimate_replay_buffer_gb(clamped, n_channels=n_channels, resolution=resolution):.1f} GB)'
    )
    return clamped


def make_env(env_cfg: dict, seed: int, rank: int):
    def _init():
        return make_pushT_env(
            env_cfg['id'],
            seed,
            rank,
            resolution=env_cfg['resolution'],
            max_episode_steps=env_cfg['max_episode_steps'],
            grayscale=env_cfg['grayscale'],
            frame_stack=env_cfg['frame_stack'],
            normalize_pixels=env_cfg['normalize_pixels'],
        )

    return _init


def make_vec_env(env_cfg: dict, seed: int):
    n_envs = env_cfg['n_envs']
    env_fns = [make_env(env_cfg, seed, i) for i in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def make_eval_env(env_cfg: dict):
    env = gym.make(
        env_cfg['id'],
        render_mode='rgb_array',
        resolution=env_cfg['resolution'],
    )
    env = wrap_pushT_env(
        env,
        max_episode_steps=env_cfg['max_episode_steps'],
        grayscale=env_cfg['grayscale'],
        frame_stack=env_cfg['frame_stack'],
        normalize_pixels=env_cfg['normalize_pixels'],
    )
    return Monitor(env)


def train_expert(
    config: dict,
    track: bool = False,
    project_name: str = 'stable-worldmodel',
    *,
    run_name: str | None = None,
    save_path: str | None = None,
    record_eval_video: bool = True,
    return_metrics: bool = False,
):
    """
    Train a vision-based Soft Actor-Critic (SAC) expert policy on PushT.
    """
    env_cfg = config['env']
    sac_cfg = config['sac']
    training_cfg = config['training']

    env_id = env_cfg['id']
    seed = env_cfg['seed']
    n_envs = env_cfg['n_envs']
    total_timesteps = training_cfg['total_timesteps']

    print('===================================================')
    print(f' Training Vision Expert Policy for {env_id}')
    print(
        f' Setup: SAC + {sac_cfg["policy"]} | {total_timesteps} Timesteps | '
        f'{n_envs} Envs | {env_cfg["resolution"]}px | '
        f'grayscale={env_cfg["grayscale"]} | '
        f'frame_stack={env_cfg["frame_stack"]} | '
        f'normalize_pixels={env_cfg["normalize_pixels"]} | '
        f'{env_cfg["max_episode_steps"]} Steps | Seed: {seed}'
    )
    print('===================================================')

    env = make_vec_env(env_cfg, seed)
    eval_env = make_eval_env(env_cfg)
    eval_freq = max(training_cfg['eval_freq_steps'] // n_envs, 1)

    n_channels = observation_channels(env_cfg)
    buffer_size = clamp_buffer_size(
        sac_cfg['buffer_size'],
        n_channels=n_channels,
        resolution=env_cfg['resolution'],
        max_gb=training_cfg.get('max_replay_buffer_gb', 48.0),
    )

    policy_kwargs = {}
    if env_cfg['normalize_pixels']:
        policy_kwargs['normalize_images'] = False

    sac_kwargs = {
        'verbose': sac_cfg['verbose'],
        'seed': seed,
        'learning_starts': sac_cfg['learning_starts'],
        'buffer_size': buffer_size,
        'batch_size': sac_cfg['batch_size'],
        'policy_kwargs': policy_kwargs,
        'tensorboard_log': (
            f'./logs/tensorboard/{env_id.replace("/", "_")}_sac_vision/'
        ),
    }
    for key in ('learning_rate', 'gradient_steps', 'ent_coef', 'tau'):
        if key in sac_cfg:
            sac_kwargs[key] = sac_cfg[key]

    model = SAC(sac_cfg['policy'], env, **sac_kwargs)

    if save_path is None:
        save_path = f'./policies/{env_id.replace("/", "_")}_expert_vision'
    os.makedirs(save_path, exist_ok=True)

    eval_callback = PushTEvalCallback(
        eval_env,
        video_folder=f'{save_path}/videos',
        record_video=record_eval_video,
        log_video_to_wandb=track and record_eval_video,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = [eval_callback]

    if track and wandb is None:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        )

    if track:
        if run_name is None:
            run_name = f'SAC_vision_{env_id.replace("/", "_")}'
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                'env': env_id,
                'algo': 'SAC',
                'policy': sac_cfg['policy'],
                'seed': seed,
                'n_envs': n_envs,
                'timesteps': total_timesteps,
                'resolution': env_cfg['resolution'],
                'max_episode_steps': env_cfg['max_episode_steps'],
                'grayscale': env_cfg['grayscale'],
                'frame_stack': env_cfg['frame_stack'],
                'normalize_pixels': env_cfg['normalize_pixels'],
                'learning_starts': sac_cfg['learning_starts'],
                'buffer_size': buffer_size,
                'buffer_size_requested': sac_cfg['buffer_size'],
                'batch_size': sac_cfg['batch_size'],
                **{
                    key: sac_cfg[key]
                    for key in ('learning_rate', 'gradient_steps', 'ent_coef', 'tau')
                    if key in sac_cfg
                },
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            model_save_path=save_path,
            model_save_freq=eval_freq,
            verbose=2,
        )
        callbacks.append(wandb_callback)

    metrics = {
        'best_eval_success_rate': 0.0,
        'final_eval_success_rate': 0.0,
        'best_eval_mean_reward': float('-inf'),
        'final_eval_mean_reward': float('-inf'),
    }

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

        model.save(f'{save_path}/final_model')
        metrics['best_eval_success_rate'] = eval_callback.best_success_rate
        metrics['final_eval_success_rate'] = eval_callback.last_success_rate
        metrics['best_eval_mean_reward'] = eval_callback.best_mean_reward
        metrics['final_eval_mean_reward'] = eval_callback.last_mean_reward

        if track and wandb.run is not None:
            wandb.run.summary['best_eval_success_rate'] = (
                metrics['best_eval_success_rate']
            )
            wandb.run.summary['final_eval_success_rate'] = (
                metrics['final_eval_success_rate']
            )
            wandb.run.summary['best_eval_mean_reward'] = (
                metrics['best_eval_mean_reward']
            )
            wandb.run.summary['final_eval_mean_reward'] = (
                metrics['final_eval_mean_reward']
            )
    finally:
        env.close()
        eval_env.close()
        if track:
            wandb.finish()

    print(f'Training complete. Models saved to {save_path}')

    if return_metrics:
        return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a vision-based RL Expert Policy for PushT'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=CONFIG_PATH,
        help='Path to JSON config with env and SAC parameters',
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

    args = parser.parse_args()
    config = load_config(args.config)

    train_expert(config, track=args.track, project_name=args.project)
