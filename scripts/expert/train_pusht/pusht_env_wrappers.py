"""Gymnasium wrappers for training vision-based SAC policies on PushT."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import GrayScaleObservation
from stable_baselines3.common.monitor import Monitor


DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_FRAME_STACK = 4


class PushTVisionObsWrapper(gym.ObservationWrapper):
    """Expose rendered RGB frames as the policy observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        render_size = getattr(env.unwrapped, 'render_size', 224)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(render_size, render_size, 3),
            dtype=np.uint8,
        )

    def observation(self, observation):
        del observation
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                'PushT render() returned None. Create the env with render_mode="rgb_array".'
            )
        return frame.astype(np.uint8, copy=False)


class SuccessInfoWrapper(gym.Wrapper):
    """Expose ``is_success`` in ``info`` when an episode ends.

    Stable-Baselines3 logs ``rollout/success_rate`` and ``eval/success_rate``
    from this key. For PushT, success is ``terminated`` (goal reached), not
    a timeout ``truncated`` from TimeLimit.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            info['is_success'] = bool(terminated)
        return obs, reward, terminated, truncated, info


class ChannelFirstObsWrapper(gym.ObservationWrapper):
    """Transpose observations to channel-first and optionally scale pixels to [0, 1]."""

    def __init__(self, env: gym.Env, *, normalize_pixels: bool = True):
        super().__init__(env)
        height, width, channels = env.observation_space.shape
        if normalize_pixels:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(channels, height, width),
                dtype=np.float32,
            )
        else:
            low = (
                np.min(env.observation_space.low)
                if np.isscalar(env.observation_space.low)
                else np.min(env.observation_space.low)
            )
            high = (
                np.max(env.observation_space.high)
                if np.isscalar(env.observation_space.high)
                else np.max(env.observation_space.high)
            )
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=(channels, height, width),
                dtype=env.observation_space.dtype,
            )
        self.normalize_pixels = normalize_pixels

    def observation(self, observation):
        if self.normalize_pixels:
            observation = observation.astype(np.float32) / 255.0
        return np.transpose(observation, (2, 0, 1))


class ChannelFrameStack(gym.Wrapper):
    """Stack the last ``num_stack`` frames along the channel dimension."""

    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        if num_stack < 1:
            raise ValueError(f'num_stack must be >= 1, got {num_stack}')

        self.num_stack = num_stack
        obs_shape = env.observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(
                f'ChannelFrameStack expects (H, W, C) observations, got shape {obs_shape}'
            )

        height, width, channels = obs_shape
        low = (
            np.min(env.observation_space.low)
            if np.isscalar(env.observation_space.low)
            else np.min(env.observation_space.low)
        )
        high = (
            np.max(env.observation_space.high)
            if np.isscalar(env.observation_space.high)
            else np.max(env.observation_space.high)
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(height, width, channels * num_stack),
            dtype=env.observation_space.dtype,
        )
        self.frames: deque[np.ndarray] = deque(maxlen=num_stack)

    def _get_observation(self) -> np.ndarray:
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        self.frames.extend([obs] * self.num_stack)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info


def wrap_pushT_env(
    env: gym.Env,
    *,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    grayscale: bool = True,
    frame_stack: int = DEFAULT_FRAME_STACK,
    normalize_pixels: bool = True,
) -> gym.Env:
    """Apply TimeLimit, vision, grayscale, frame-stack, and normalize wrappers."""
    if env.spec is None or env.spec.max_episode_steps is None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    env = PushTVisionObsWrapper(env)

    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)

    if frame_stack > 1:
        env = ChannelFrameStack(env, frame_stack)

    env = ChannelFirstObsWrapper(env, normalize_pixels=normalize_pixels)
    env = SuccessInfoWrapper(env)

    return env


def make_pushT_env(
    env_id: str,
    seed: int,
    rank: int,
    *,
    resolution: int = 96,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    grayscale: bool = True,
    frame_stack: int = DEFAULT_FRAME_STACK,
    normalize_pixels: bool = True,
) -> gym.Env:
    """Build a monitored PushT env that exposes vision observations."""
    import stable_worldmodel  # noqa: F401 — register gym envs in worker

    env = gym.make(
        env_id,
        render_mode='rgb_array',
        resolution=resolution,
    )
    env = wrap_pushT_env(
        env,
        max_episode_steps=max_episode_steps,
        grayscale=grayscale,
        frame_stack=frame_stack,
        normalize_pixels=normalize_pixels,
    )
    env = Monitor(env)
    env.reset(seed=seed + rank)
    return env
