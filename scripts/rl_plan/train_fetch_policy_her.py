import os
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import stable_worldmodel  # registers swm/* envs with Gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

_EXPERT_FETCH = Path(__file__).resolve().parent.parent / 'expert' / 'train_fetch'
if str(_EXPERT_FETCH) not in sys.path:
    sys.path.insert(0, str(_EXPERT_FETCH))

from callbacks import FetchEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her import HerReplayBuffer

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None

from env import register_fetch_wm_env

register_fetch_wm_env()

_WM_ENV_ID = 'swm/FetchReachWM-v0'
_HER_OBS_KEYS = ('observation', 'achieved_goal', 'desired_goal')


def _dict_env_id(env_id: str) -> str:
    """Map flattened swm/Fetch* env ids to their Dict HER variants."""
    if env_id == _WM_ENV_ID:
        return env_id
    if env_id.endswith('Dict-v3'):
        return env_id
    if env_id.endswith('Dense-v3'):
        raise ValueError(
            f"HER with sparse relabeling expects Dict sparse envs (e.g. "
            f"'swm/FetchReachDict-v3' or '{_WM_ENV_ID}'), not dense reward "
            f"'{env_id}'. Use train_fetch_policy.py for dense SAC."
        )
    if env_id.startswith('swm/Fetch') and env_id.endswith('-v3'):
        base = env_id.removeprefix('swm/Fetch').removesuffix('-v3')
        return f'swm/Fetch{base}Dict-v3'
    raise ValueError(
        f"Unsupported env_id '{env_id}' for HER. Use a registered "
        f"swm/Fetch*Dict-v3 id (e.g. swm/FetchReachDict-v3) or "
        f"'{_WM_ENV_ID}'."
    )


def _make_env(env_id: str, world_model_path: str | None, checkpoint: str | None):
    if env_id == _WM_ENV_ID:
        if world_model_path is None or checkpoint is None:
            raise ValueError(
                f"'{env_id}' requires --wm-path and --checkpoint."
            )
        return gym.make(
            env_id,
            world_model_path=world_model_path,
            checkpoint=checkpoint,
        )
    return gym.make(env_id)


def _check_her_obs_space(env: gym.Env) -> None:
    obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Dict):
        raise ValueError(
            'HER requires a Dict observation space with keys '
            f'{_HER_OBS_KEYS}. Got {type(obs_space).__name__}. '
            'Use a *Dict-v3 env (e.g. swm/FetchReachDict-v3), not the '
            'flattened swm/FetchReach-v3 variant.'
        )
    missing = [k for k in _HER_OBS_KEYS if k not in obs_space.spaces]
    if missing:
        raise ValueError(
            f'Observation space missing HER keys {missing}. '
            f'Available keys: {list(obs_space.spaces.keys())}'
        )


def train_expert(
    env_id: str,
    total_timesteps: int,
    seed: int = 42,
    track: bool = False,
    project_name: str = 'stable-worldmodel',
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = 'future',
    learning_starts: int = 1000,
    world_model_path: str | None = None,
    checkpoint: str | None = None,
):
    """
    Train SAC with Hindsight Experience Replay on sparse Fetch tasks.

    The environment must expose goal-conditioned Dict observations
    (observation / achieved_goal / desired_goal). Use swm/Fetch*Dict-v3
    or swm/FetchReachWM-v0 for world-model latent observations.
    """
    env_id = _dict_env_id(env_id)

    print('===================================================')
    print(f' Training Expert Policy for {env_id}')
    print(
        f' Setup: SAC+HER | {total_timesteps} Timesteps | Seed: {seed}'
    )
    print(
        f' HER: strategy={goal_selection_strategy}, '
        f'n_sampled_goal={n_sampled_goal}'
    )
    print('===================================================')

    env = Monitor(_make_env(env_id, world_model_path, checkpoint))
    eval_env = Monitor(_make_env(env_id, world_model_path, checkpoint))
    _check_her_obs_space(env)
    _check_her_obs_space(eval_env)

    model = SAC(
        'MultiInputPolicy',
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
        ),
        learning_starts=learning_starts,
        verbose=1,
        seed=seed,
        tensorboard_log=(
            f'./logs/tensorboard/{env_id.replace("/", "_")}_sac_her/'
        ),
    )

    save_path = f'./policies/{env_id.replace("/", "_")}_expert'
    os.makedirs(save_path, exist_ok=True)

    eval_callback = FetchEvalCallback(
        eval_env,
        video_folder=f'{save_path}/videos',
        log_video_to_wandb=track,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]

    if track and wandb is None:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        )

    if track:
        wandb.init(
            project=project_name,
            name=f'SAC_HER_{env_id.replace("/", "_")}',
            config={
                'env': env_id,
                'algo': 'SAC+HER',
                'seed': seed,
                'timesteps': total_timesteps,
                'n_sampled_goal': n_sampled_goal,
                'goal_selection_strategy': goal_selection_strategy,
                'learning_starts': learning_starts,
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
            'Train a SAC+HER expert on sparse Fetch envs '
            '(requires swm/FetchReachWM-v0 observation space)'
        )
    )
    parser.add_argument(
        '--env',
        type=str,
        default=_WM_ENV_ID,
        help=(
            'Env id: swm/FetchReachWM-v0 (default), swm/Fetch*Dict-v3, or a '
            'flat swm/Fetch*-v3 id (auto-mapped to *Dict-v3).'
        ),
    )
    parser.add_argument(
        '--wm-path',
        type=str,
        default=None,
        help=f'World model checkpoint directory (required for {_WM_ENV_ID})',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=f'Checkpoint filename inside --wm-path (required for {_WM_ENV_ID})',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100_000,
        help='Total environment steps to execute',
    )
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument(
        '--n-sampled-goal',
        type=int,
        default=4,
        help='HER: virtual transitions sampled per real transition',
    )
    parser.add_argument(
        '--goal-strategy',
        type=str,
        default='future',
        choices=('future', 'final', 'episode'),
        help='HER goal selection strategy',
    )
    parser.add_argument(
        '--learning-starts',
        type=int,
        default=1000,
        help=(
            'Steps before gradient updates (should exceed max_episode_steps=50)'
        ),
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

    train_expert(
        args.env,
        args.timesteps,
        args.seed,
        args.track,
        args.project,
        n_sampled_goal=args.n_sampled_goal,
        goal_selection_strategy=args.goal_strategy,
        learning_starts=args.learning_starts,
        world_model_path=args.wm_path,
        checkpoint=args.checkpoint,
    )
