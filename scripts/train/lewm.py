import os
import argparse
import sys
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained

_CLI = argparse.Namespace(
    track=False, project=None, output_model_name=None, wandb_name=None
)


def resolve_run_names(cfg):
    """Set output_model_name and wandb.name from prefix + embed_dim when needed."""
    with open_dict(cfg):
        if _CLI.output_model_name:
            cfg.output_model_name = _CLI.output_model_name
        elif not cfg.get('output_model_name') and cfg.get(
            'output_model_name_prefix'
        ):
            embed_dim = cfg.embed_dim
            if not OmegaConf.is_list(embed_dim):
                cfg.output_model_name = (
                    f'{cfg.output_model_name_prefix}_{embed_dim}'
                )

        if _CLI.wandb_name:
            cfg.wandb.name = _CLI.wandb_name
        elif not cfg.wandb.get('name') and cfg.get('output_model_name'):
            cfg.wandb.name = cfg.output_model_name


def setup_wandb_logger(cfg):
    """Initialize W&B the same way as train_fetch_policy_her.py (--track)."""
    track = (
        _CLI.track
        or cfg.get('track', False)
        or cfg.wandb.get('enabled', False)
    )
    if not track:
        return None

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        ) from exc

    project = (
        _CLI.project
        or cfg.wandb.get('project', 'stable-worldmodel')
    )
    name = cfg.wandb.get('name', cfg.output_model_name)
    logger = WandbLogger(
        project=project,
        name=name,
        save_code=True,
        log_model=False,
    )
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    return logger


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

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

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    if 'state' in batch:
        batch['state'] = torch.nan_to_num(batch['state'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    resolve_run_names(cfg)

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    pixel_encoding = cfg.model.get('pixel_encoding', True)
    transforms = []
    if pixel_encoding:
        transforms.append(
            get_img_preprocessor(
                source='pixels', target='pixels', img_size=cfg.img_size
            )
        )

    with open_dict(cfg):
        norm_cols = (
            ('state', 'action')
            if not pixel_encoding
            else cfg.data.dataset.keys_to_load
        )
        for col in norm_cols:
            if col not in cfg.data.dataset.keys_to_load:
                continue
            if col.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

        if not pixel_encoding:
            cfg.model.encoder.input_dim = dataset.get_dim('state')

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
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = setup_wandb_logger(cfg)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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


def parse_cli_args(argv=None):
    """Parse CLI flags before Hydra, like train_fetch_policy_her.py."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--track',
        action='store_true',
        help='Log training metrics natively to Weights & Biases',
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='WandB Cloud project name',
    )
    parser.add_argument(
        '--output-model-name',
        type=str,
        default=None,
        help='Checkpoint / run name override (Hydra: output_model_name=...)',
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='WandB run name override (Hydra: wandb.name=...)',
    )
    args, hydra_args = parser.parse_known_args(argv)
    _CLI.track = args.track
    _CLI.project = args.project
    _CLI.output_model_name = args.output_model_name
    _CLI.wandb_name = args.wandb_name
    return hydra_args


if __name__ == '__main__':
    sys.argv = [sys.argv[0], *parse_cli_args()]
    run()
