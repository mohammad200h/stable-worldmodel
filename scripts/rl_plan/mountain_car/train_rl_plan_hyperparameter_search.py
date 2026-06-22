"""Grid search over embed_dim by launching train_rl_plan.py once per value."""

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / 'train_rl_plan.py'
DEFAULT_CONFIG = SCRIPT_DIR / 'rl_state_worldmodel_hyperparameter_search.yaml'


def load_search_config(config_path: Path):
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    return OmegaConf.load(config_path)


def iter_embed_dims(embed_dim):
    if isinstance(embed_dim, (list, ListConfig)):
        return list(embed_dim)
    return [embed_dim]


def resolve_run_settings(cfg, embed_dim: int) -> dict:
    run_cfg = OmegaConf.merge(cfg, OmegaConf.create({'embed_dim': embed_dim}))
    OmegaConf.resolve(run_cfg)
    return {
        'world_model_path': f'{run_cfg.world_model_path_prefix}{embed_dim}',
        'checkpoint': run_cfg.checkpoint,
        'embedding_is_made_of_pixels': run_cfg.embedding_is_made_of_pixels,
        'model_name': run_cfg.model_name,
        'wandb_project': run_cfg.wandb.project,
        'wandb_name': run_cfg.wandb.name,
        'timesteps': run_cfg.get('timesteps', 1_000_000),
        'seed': run_cfg.get('seed', 42),
    }


def run_train_rl_plan(
    settings: dict,
    track: bool,
    train_args: list[str],
) -> None:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--input-state-type',
        'state',
        '--wm-path',
        settings['world_model_path'],
        '--checkpoint',
        settings['checkpoint'],
        '--model-name',
        settings['model_name'],
        '--wandb-name',
        settings['wandb_name'],
        '--project',
        settings['wandb_project'],
        '--timesteps',
        str(settings['timesteps']),
        '--seed',
        str(settings['seed']),
        *train_args,
    ]
    if settings['embedding_is_made_of_pixels']:
        cmd.append('--embedding-is-made-of-pixels')
    else:
        cmd.append('--no-embedding-is-made-of-pixels')
    if track:
        cmd.append('--track')

    print(
        f'\n=== embed_dim={settings["embed_dim"]} '
        f'({settings["model_name"]}) ==='
    )
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Grid search RL-plan embed_dim via train_rl_plan.py'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help='Search config with embed_dim list and wandb settings',
    )
    parser.add_argument(
        '--track',
        action='store_true',
        help='Enable Weights & Biases logging for each run',
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='WandB project override (default: wandb.project from config)',
    )
    parser.add_argument(
        'train_args',
        nargs='*',
        help='Extra arguments passed to each train_rl_plan.py run',
    )
    args = parser.parse_args(argv)

    cfg = load_search_config(args.config)
    embed_dims = iter_embed_dims(cfg.embed_dim)
    wandb_project = args.project or cfg.wandb.get('project')

    print(
        f'Starting grid search: embed_dims={embed_dims}, '
        f'project={wandb_project}'
    )

    for embed_dim in embed_dims:
        settings = resolve_run_settings(cfg, embed_dim)
        settings['embed_dim'] = embed_dim
        if wandb_project:
            settings['wandb_project'] = wandb_project
        run_train_rl_plan(settings, track=args.track, train_args=args.train_args)


if __name__ == '__main__':
    main()
