import argparse

import gymnasium as gym
import stable_worldmodel  # registers swm/* envs with Gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from statistic import PolicyComparisonCallback

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None

# https://medium.com/@emikea03/the-power-of-ppo-how-proximal-policy-optimization-solves-a-range-of-rl-problems-10076d9da34e


class EntropyDecayCallback(BaseCallback):
    """Decay ent_coef linearly from initial to final value over decay_steps, starting after start_after_timesteps."""

    def __init__(
        self,
        initial_ent_coef: float,
        final_ent_coef: float,
        decay_steps: int,
        start_after_timesteps: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_steps = decay_steps
        self.start_after_timesteps = start_after_timesteps

    def _on_step(self) -> bool:
        total = self.num_timesteps
        if total < self.start_after_timesteps:
            return True
        elapsed = total - self.start_after_timesteps
        if elapsed >= self.decay_steps:
            self.model.ent_coef = self.final_ent_coef
        else:
            alpha = elapsed / self.decay_steps
            self.model.ent_coef = self.initial_ent_coef + alpha * (self.final_ent_coef - self.initial_ent_coef)
        return True


N_ENVS = 4
MAX_EPISODE_STEPS = 1000
LOG_DIR = "logs/"
ENV_ID = "swm/MountainCarContinuousControl-v0"


def make_env(record_video: bool = False):
    env = gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env, LOG_DIR)  # needed so SB3 can log rollout/ep_rew_mean and rollout/ep_len_mean
    return env
    


def make_eval_env():
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env, LOG_DIR)
    env = RecordVideo(
            env,
            video_folder="videos",
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix="mountaincar",
        )
    return env


def train(
    total_timesteps: int = 1_000_000,
    seed: int = 37,
    track: bool = False,
    project_name: str = "stable-worldmodel",
):
    env_id = ENV_ID
    initial_ent_coef = 0.3
    final_ent_coef = 0.001
    decay_steps = 100_000
    start_after_timesteps = 100_000

    train_env = SubprocVecEnv([lambda: make_env(record_video=False) for _ in range(N_ENVS)])
    eval_env = make_eval_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        name_prefix="ppo_mountain_car",
        save_path="./checkpoints/",
    )

    policy_compare_callback = PolicyComparisonCallback(
        make_eval_env=make_eval_env,
        save_path="./policy_checkpoints/",
        name_prefix="policy",
        compare_freq=10000,
        steps_per_collect=500,
        deterministic=True,
        mmd_threshold=0.05,
        verbose=1,
    )

    entropy_decay_callback = EntropyDecayCallback(
        initial_ent_coef=initial_ent_coef,
        final_ent_coef=final_ent_coef,
        decay_steps=decay_steps,
        start_after_timesteps=start_after_timesteps,
    )

    callbacks = [
        eval_callback,
        checkpoint_callback,
        policy_compare_callback,
        entropy_decay_callback,
    ]

    if track and wandb is None:
        raise ImportError(
            "wandb is required for tracking. Install it with: pip install wandb"
        )

    if track:
        wandb.init(
            project=project_name,
            name="PPO_MountainCarContinuousControl",
            config={
                "env": env_id,
                "algo": "PPO",
                "seed": seed,
                "timesteps": total_timesteps,
                "n_envs": N_ENVS,
                "max_episode_steps": MAX_EPISODE_STEPS,
                "n_steps": 2048,
                "initial_ent_coef": initial_ent_coef,
                "final_ent_coef": final_ent_coef,
                "decay_steps": decay_steps,
                "start_after_timesteps": start_after_timesteps,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            model_save_path="./checkpoints/",
            model_save_freq=10000,
            verbose=2,
        )
        callbacks.append(wandb_callback)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=2,
        n_steps=2048,
        seed=seed,
        tensorboard_log=LOG_DIR,
        ent_coef=initial_ent_coef,
    )
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save("ppo_mountain_car")

    if track:
        wandb.finish()

    train_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Train a PPO expert on {ENV_ID}"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps to execute",
    )
    parser.add_argument("--seed", type=int, default=37, help="RNG seed")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log training metrics natively to Weights & Biases",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="stable-worldmodel",
        help="WandB Cloud project name",
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        track=args.track,
        project_name=args.project,
    )