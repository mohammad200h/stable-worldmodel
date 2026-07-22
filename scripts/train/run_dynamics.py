"""Train LEWM dynamics (+ DreamerV3 ``symexp_twohot`` reward head).

Analogous to OfflineRL-Kit ``run_example/run_dynamics.py``, but for pixel
world models via LeWM. Reward is trained with DreamerV3 soft two-hot CE
(``log_prob``), not soft-mean regression on decoded scalars.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Save model checkpoint after each epoch via ``save_pretrained``."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def dynamics_forward(self, batch, stage, cfg):
    """Encode, predict next emb, optional DreamerV3 reward CE, SIGReg."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    rew_w = cfg.loss.get('reward_weight', 1.0)
    cont_w = cfg.loss.get('continue_weight', 1.0)

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    if 'reward' in batch:
        batch['reward'] = torch.nan_to_num(batch['reward'], 0.0)
    if 'cont' in batch:
        batch['cont'] = torch.nan_to_num(batch['cont'], 1.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    # DreamerV3 reward: CE / log_prob on predicted latents (no soft-mean decode).
    if self.model.reward_head is not None and 'reward' in batch:
        # Align reward with predicted next-state slots: pred covers
        # timesteps [0..ctx_len) predicting [n_preds..n_preds+ctx_len).
        # Use rewards at the predicted timesteps.
        rew = batch['reward']
        if rew.ndim == 3 and rew.size(-1) == 1:
            rew = rew.squeeze(-1)
        rew_tgt = rew[:, n_preds : n_preds + pred_emb.size(1)]
        rew_feat = pred_emb if cfg.loss.get('reward_grad', True) else pred_emb.detach()
        rew_dist = self.model.predict_reward(rew_feat)
        output['reward_loss'] = rew_dist.nll(rew_tgt).mean()
        output['loss'] = output['loss'] + rew_w * output['reward_loss']

    # DreamerV3 continue: Bernoulli CE on predicted latents (contdisc soft labels).
    if self.model.continue_head is not None and 'cont' in batch:
        cont = batch['cont']
        if cont.ndim == 3 and cont.size(-1) == 1:
            cont = cont.squeeze(-1)
        cont_tgt = cont[:, n_preds : n_preds + pred_emb.size(1)]
        if cfg.loss.get('contdisc', True):
            horizon = cfg.loss.get('horizon', 333)
            cont_tgt = cont_tgt * (1 - 1 / horizon)
        cont_feat = (
            pred_emb if cfg.loss.get('continue_grad', True) else pred_emb.detach()
        )
        cont_dist = self.model.predict_continue(cont_feat)
        output['continue_loss'] = cont_dist.nll(cont_tgt).mean()
        output['loss'] = output['loss'] + cont_w * output['continue_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='run_dynamics')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels') or col == 'reward':
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if getattr(cfg, 'require_reward_head', True) and world_model.reward_head is None:
        raise ValueError(
            'run_dynamics expects model.reward_head (DreamerV3 symexp_twohot). '
            'Set it in config or pass require_reward_head=false.'
        )
    if getattr(cfg, 'require_continue_head', False) and world_model.continue_head is None:
        raise ValueError(
            'run_dynamics expects model.continue_head (DreamerV3 binary). '
            'Set it in config or pass require_continue_head=false.'
        )

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(dynamics_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[save_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
