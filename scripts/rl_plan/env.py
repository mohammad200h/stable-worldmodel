import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm  # registers swm/* envs for gym.make
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


class FetchWorldModelEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        world_model_path,
        checkpoint,
        device=None,
        img_size=224,
        env_id='swm/FetchReachDict-v3',
        **kwargs,
    ):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.wm_config = self._load_wm_config(world_model_path)
        self.worldmodel = self._load_worldmodel(world_model_path, checkpoint)
        self.pixel_transform = self._make_pixel_transform(img_size)

        self._env_id = env_id
        self.original_env = self._load_original_env(self._env_id)
        self.action_space = self.original_env.action_space

        emb_dim = self._infer_emb_dim(img_size)
        orig_obs_space = self.original_env.observation_space
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(emb_dim,),
                    dtype=np.float32,
                ),
                'achieved_goal': orig_obs_space['achieved_goal'],
                'desired_goal': orig_obs_space['desired_goal'],
            }
        )

        self.frameskip = self._infer_frameskip()
        self.wm_action_dim = self.worldmodel.action_encoder.input_dim
        self.action_processor = self._load_action_processor()
        self._action_buf = deque(maxlen=self.frameskip)
        self.obs_embedding = None

        # original env state
        self.original_env_state = {
            "obs":None,
            "action":None,
            "next_obs":None,
            "reward":None,
            "done":False,
            "truncated":False,
            "info":{}
        }

    def _load_wm_config(self, path):
        config_path = Path(path) / 'config.json'
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            return json.load(f)

    def _load_original_env(self, env_name):
        return gym.make(env_name, render_mode='rgb_array')

    def _load_worldmodel(self, path, checkpoint):
        ckpt_path = Path(path) / checkpoint
        model = swm.wm.utils.load_pretrained(str(ckpt_path))
        model = model.to(self.device).eval()
        model.requires_grad_(False)
        return model

    def _infer_emb_dim(self, img_size):
        with torch.no_grad():
            dtype = next(self.worldmodel.parameters()).dtype
            pixels = torch.zeros(
                1, 1, 3, img_size, img_size, device=self.device, dtype=dtype
            )
            encoded = self.worldmodel.encode({'pixels': pixels})
        return int(encoded['emb'].shape[-1])

    def _her_obs(self, obs_embedding, raw_obs):
        return {
            'observation': np.asarray(obs_embedding, dtype=np.float32),
            'achieved_goal': np.asarray(
                raw_obs['achieved_goal'], dtype=np.float64
            ),
            'desired_goal': np.asarray(
                raw_obs['desired_goal'], dtype=np.float64
            ),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Forward sparse reward for HER goal relabeling."""
        return self.original_env.get_wrapper_attr('compute_reward')(
            achieved_goal, desired_goal, info
        )

    def render(self):
        return self.original_env.render()

    def _infer_frameskip(self):
        env_action_dim = self.original_env.action_space.shape[0]
        wm_action_dim = self.worldmodel.action_encoder.input_dim
        if wm_action_dim % env_action_dim != 0:
            raise ValueError(
                f'WM action dim {wm_action_dim} is not divisible by env '
                f'action dim {env_action_dim}'
            )
        return wm_action_dim // env_action_dim

    def _load_action_processor(self):
        dataset_name = (
            self.wm_config.get('data', {})
            .get('dataset', {})
            .get('name')
        )
        if dataset_name is None:
            return None
        dataset = swm.data.load_dataset(
            dataset_name,
            keys_to_load=['action'],
            num_steps=1,
            frameskip=1,
        )
        processor = preprocessing.StandardScaler()
        processor.fit(dataset.get_col_data('action'))
        return processor

    def _make_pixel_transform(self, img_size):
        dtype = next(self.worldmodel.parameters()).dtype
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(dtype, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=img_size),
            ]
        )

    def _prepare_pixels(self, pixels):
        pixels = self.pixel_transform(pixels)
        return pixels.unsqueeze(0).unsqueeze(0).to(self.device)

    def _prepare_wm_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action_dim = self.action_space.shape[0]

        if action.shape[0] == env_action_dim:
            self._action_buf.append(action.copy())
            while len(self._action_buf) < self.frameskip:
                self._action_buf.appendleft(action.copy())
            block = np.stack(list(self._action_buf), axis=0)
        elif action.shape[0] == self.wm_action_dim:
            block = action.reshape(self.frameskip, env_action_dim)
        else:
            raise ValueError(
                f'Expected action dim {env_action_dim} or {self.wm_action_dim}, '
                f'got {action.shape[0]}'
            )

        if self.action_processor is not None:
            block = self.action_processor.transform(block)

        return block.reshape(-1)

    def reset(self, seed=None, options=None):
        self._action_buf.clear()
        raw_obs, pixels = self._reset_env_obs(seed=seed, options=options)
        encoded = self.worldmodel.encode({'pixels': pixels})
        self.obs_embedding = encoded['emb']
        obs_embedding = (
            self.obs_embedding.squeeze(0).squeeze(0).detach().cpu().numpy()
        )
        return self._her_obs(obs_embedding, raw_obs), {}

    def step(self, action):
        wm_action = self._prepare_wm_action(action)
        next_embedding = self._step_wm(self.obs_embedding, wm_action)
        self.obs_embedding = next_embedding
        obs_embedding = (
            next_embedding.squeeze(0).squeeze(0).detach().cpu().numpy()
        )

        raw_obs, reward, done, truncated, info = self._step_env(action)
        return (
            self._her_obs(obs_embedding, raw_obs),
            reward,
            done,
            truncated,
            info,
        )

    def close(self):
        self.original_env.close()

    def _step_wm(self, obs_embedding, action):
        """Predict the next latent state given embedding history and action(s).

        Args:
            obs_embedding: (B, T, D) or (T, D) embedding history.
            action: (A,), (B, A), or (B, T, A) actions aligned with history.

        Returns:
            (B, 1, D) predicted next embedding.
        """
        dtype = next(self.worldmodel.parameters()).dtype

        if not torch.is_tensor(obs_embedding):
            obs_embedding = torch.as_tensor(
                obs_embedding, device=self.device, dtype=dtype
            )
        else:
            obs_embedding = obs_embedding.to(device=self.device, dtype=dtype)

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device, dtype=dtype)
        else:
            action = action.to(device=self.device, dtype=dtype)

        if obs_embedding.ndim == 2:
            obs_embedding = obs_embedding.unsqueeze(0)

        if action.ndim == 1:
            emb = obs_embedding[:, -1:, :]
            action = action.unsqueeze(0).unsqueeze(0)
        elif action.ndim == 2:
            emb = obs_embedding[:, -1:, :]
            action = action.unsqueeze(1)
        elif action.ndim == 3:
            emb = obs_embedding[:, -action.shape[1] :, :]
        else:
            raise ValueError(f'Unexpected action shape: {tuple(action.shape)}')

        act_emb = self.worldmodel.action_encoder(action)
        with torch.no_grad():
            return self.worldmodel.predict(emb, act_emb)[:, -1:]

    def _step_env(self, action):
        obs, reward, done, truncated, info = self.original_env.step(action)
        self.original_env_state["next_obs"] = obs
        self.original_env_state["reward"] = reward
        self.original_env_state["done"] = done
        self.original_env_state["truncated"] = truncated
        self.original_env_state["info"] = info
        return obs, reward, done, truncated, info
    
    def _reset_env_obs(self, seed=None, options=None):
        obs, info = self.original_env.reset(seed=seed, options=options)
        self.original_env_state['obs'] = obs
        self.original_env_state['info'] = info
        pixels = self.original_env.render()
        return obs, self._prepare_pixels(pixels)


def register_fetch_wm_env():
    try:
        gym.spec('swm/FetchReachWM-v0')
    except gym.error.Error:
        gym.register(
            id='swm/FetchReachWM-v0',
            entry_point='env:FetchWorldModelEnv',
        )


register_fetch_wm_env()

