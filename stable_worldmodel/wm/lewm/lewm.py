import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .module import (
    ContinuePredictor,
    RewardPredictor,
    RewardPredictionMode,
)


def _head_cfg(cfg: dict | None) -> dict:
    return cfg or {}


class LeWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        pixel_encoding=True,
        reward_prediction=None,
        continue_prediction=None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.reward_prediction_cfg = _head_cfg(reward_prediction)
        self.continue_prediction_cfg = _head_cfg(continue_prediction)
        self.reward_predictor = None
        self.continue_predictor = None
        if self.reward_prediction_cfg.get('enabled', False):
            self.reward_predictor = RewardPredictor(
                mode=self.reward_prediction_cfg['mode'],
                embed_dim=self.reward_prediction_cfg['embed_dim'],
                action_emb_dim=self.reward_prediction_cfg['action_emb_dim'],
                hidden_dim=self.reward_prediction_cfg.get('hidden_dim'),
            )
        if self.continue_prediction_cfg.get('enabled', False):
            self.continue_predictor = ContinuePredictor(
                mode=self.continue_prediction_cfg['mode'],
                embed_dim=self.continue_prediction_cfg['embed_dim'],
                action_emb_dim=self.continue_prediction_cfg['action_emb_dim'],
                hidden_dim=self.continue_prediction_cfg.get('hidden_dim'),
            )

        self.pixel_encoding = pixel_encoding
        if pixel_encoding:
            self.encode = self._encode_pixels
            self.rollout = self._rollout_pixels
        else:
            self.encode = self._encode_state
            self.rollout = self._rollout_state
    

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    @property
    def reward_prediction_enabled(self) -> bool:
        return self.reward_predictor is not None

    @property
    def continue_prediction_enabled(self) -> bool:
        return self.continue_predictor is not None

    @property
    def transition_head_mode(self) -> RewardPredictionMode | None:
        if self.reward_predictor is not None:
            return self.reward_predictor.mode
        if self.continue_predictor is not None:
            return self.continue_predictor.mode
        return None

    @property
    def reward_prediction_mode(self) -> RewardPredictionMode | None:
        if self.reward_predictor is None:
            return None
        return self.reward_predictor.mode

    def predict_reward(
        self,
        z_cur: torch.Tensor,
        z_pred: torch.Tensor | None = None,
        act_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict reward for a latent transition."""
        if self.reward_predictor is None:
            raise RuntimeError(
                'Reward prediction is disabled. '
                'Enable reward_prediction in the model config.'
            )
        return self.reward_predictor(z_cur, z_pred=z_pred, act_emb=act_emb)

    def predict_continue(
        self,
        z_cur: torch.Tensor,
        z_pred: torch.Tensor | None = None,
        act_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict episode-continue logit for a latent transition."""
        if self.continue_predictor is None:
            raise RuntimeError(
                'Continue prediction is disabled. '
                'Enable continue_prediction in the model config.'
            )
        return self.continue_predictor(z_cur, z_pred=z_pred, act_emb=act_emb)

    def align_transition_batch(
        self,
        emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        act_emb: torch.Tensor,
        n_preds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align latent tensors with predictor outputs for head training."""
        n_steps = tgt_emb.size(1)
        offset = n_preds - 1
        z_cur = emb[:, offset : offset + n_steps]
        z_next = tgt_emb[:, :n_steps]
        act = act_emb[:, offset : offset + n_steps]
        return z_cur, z_next, act

    def align_reward_batch(
        self,
        emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        act_emb: torch.Tensor,
        n_preds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.align_transition_batch(emb, tgt_emb, act_emb, n_preds)

    def _encode_pixels(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def _encode_state(self, info):
        """Encode numeric state into embeddings via an MLP encoder.

        Expects info['state'] of shape (B, T, state_dim) with no pixels.
        """
        state = info['state'].float()
        b, t = state.shape[:2]
        emb = self.encoder(state)
        emb = self.projector(rearrange(emb, 'b t d -> (b t) d'))
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b, t=t)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    

    ####################
    ## Inference only ##
    ####################

    def _rollout_pixels(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )  # (BS, T, A_emb)

        # rollout predictor autoregressively for n_steps + 1 (final) steps
        # emb_list holds individual (BS, D) frames, each with its own grad_fn
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + n_steps + 1, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info

    def _rollout_state(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        state: (B, S, T, state_dim)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        assert 'state' in info, 'state not in info_dict'
        H = info['state'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )  # (BS, T, A_emb)

        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + n_steps + 1, D)
        info['predicted_emb'] = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict['predicted_emb']  # (B,S, T-1, dim)
        goal_emb = info_dict['goal_emb']  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert 'goal' in info_dict, 'goal not in info_dict'

        # encode goal state, or reuse cached embedding from a prior call
        if 'goal_emb' not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            obs_key = 'pixels' if self.pixel_encoding else 'state'
            goal[obs_key] = goal['goal']

            for k in info_dict:
                if k.startswith('goal_'):
                    goal[k[len('goal_') :]] = goal.pop(k)

            goal.pop('action')
            goal = self.encode(goal)

            info_dict['goal_emb'] = goal['emb']

        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost


__all__ = ['LeWM']
