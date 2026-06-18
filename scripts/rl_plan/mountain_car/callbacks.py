"""Callbacks for Mountain Car expert policy training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from stable_worldmodel.plot import save_video

try:
    import wandb
except ImportError:
    wandb = None


class MountainCarEvalCallback(EvalCallback):
    """Evaluate like :class:`EvalCallback` and record video from the same eval run."""

    def __init__(
        self,
        eval_env: gym.Env,
        *,
        video_folder: str | Path | None = None,
        record_video: bool = True,
        fps: int = 25,
        log_video_to_wandb: bool = False,
        **eval_callback_kwargs,
    ):
        """
        Args:
            eval_env: Environment used for periodic evaluation (passed to
                :class:`EvalCallback`).
            video_folder: Directory for mp4 recordings. Required when
                ``record_video=True``.
            record_video: Save a video of the first eval episode each time
                :class:`EvalCallback` runs.
            fps: Video frame rate.
            log_video_to_wandb: Upload the latest video to Weights & Biases.
            **eval_callback_kwargs: Forwarded to :class:`EvalCallback` (e.g.
                ``best_model_save_path``, ``log_path``, ``eval_freq``).
        """
        if record_video and video_folder is None:
            raise ValueError('video_folder is required when record_video=True')

        self.video_folder = Path(video_folder) if video_folder is not None else None
        self.record_video = record_video
        self.fps = fps
        self.log_video_to_wandb = log_video_to_wandb
        self._capturing_video = False
        self._video_frames: list = []
        self.last_success_rate = 0.0
        self.best_success_rate = -1.0

        super().__init__(eval_env, **eval_callback_kwargs)

    def _init_callback(self) -> None:
        super()._init_callback()
        if self.record_video and self.video_folder is not None:
            self.video_folder.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if is_eval_step and self.record_video:
            self._video_frames = []
            self._capturing_video = True

        continue_training = super()._on_step()

        if is_eval_step:
            self._log_eval_success_rate()

        if is_eval_step and self.record_video:
            self._capturing_video = False
            if self._video_frames:
                self._save_eval_video()

        return continue_training

    def _log_eval_success_rate(self) -> None:
        if len(self._is_success_buffer) == 0:
            return

        success_rate = float(np.mean(self._is_success_buffer))
        self.last_success_rate = success_rate

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            if self.verbose >= 1:
                print(
                    f'MountainCarEvalCallback: new best success rate '
                    f'{100 * success_rate:.2f}%'
                )

        if (
            self.log_video_to_wandb
            and wandb is not None
            and wandb.run is not None
        ):
            wandb.log(
                {'eval/success_rate': success_rate},
                step=self.num_timesteps,
            )

    def _log_success_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        super()._log_success_callback(locals_, globals_)

        if not self._capturing_video:
            return

        episode_counts = locals_['episode_counts']
        if episode_counts[0] > 0:
            return

        frame = self._render_eval_frame(locals_['env'])
        if frame is not None:
            self._video_frames.append(frame)

    @staticmethod
    def _render_eval_frame(vec_env: VecEnv):
        env = vec_env.envs[0] if hasattr(vec_env, 'envs') else vec_env
        return env.unwrapped.render()

    def _save_eval_video(self) -> None:
        assert self.video_folder is not None

        video_path = self.video_folder / f'eval_{self.num_timesteps:08d}.mp4'
        save_video(video_path, self._video_frames, fps=self.fps)

        if self.verbose >= 1:
            print(
                f'MountainCarEvalCallback: saved {video_path} '
                f'(first eval episode, frames={len(self._video_frames)}, '
                f'success_rate={100 * self.last_success_rate:.2f}%, '
                f'mean_reward={self.last_mean_reward:.2f})'
            )

        if (
            self.log_video_to_wandb
            and wandb is not None
            and wandb.run is not None
        ):
            wandb.log(
                {
                    'eval/video': wandb.Video(
                        str(video_path), fps=self.fps, format='mp4'
                    ),
                },
                step=self.num_timesteps,
            )


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
