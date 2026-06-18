"""Grid search over embed_dim by launching lewm.py once per value."""

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
LEWM_SCRIPT = SCRIPT_DIR / 'lewm.py'
DEFAULT_CONFIG_NAME = 'lewm_mountain_car_state_hyperparameter_search'


def load_search_config(config_name: str):
    config_path = SCRIPT_DIR / 'config' / f'{config_name}.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    return OmegaConf.load(config_path)


def iter_embed_dims(embed_dim):
    if isinstance(embed_dim, (list, ListConfig)):
        return list(embed_dim)
    return [embed_dim]


def build_run_name(prefix: str, embed_dim: int) -> str:
    return f'{prefix}_{embed_dim}'


def run_lewm(
    config_name: str,
    embed_dim: int,
    run_name: str,
    wandb_project: str | None,
    track: bool,
    overrides: list[str],
) -> None:
    cmd = [
        sys.executable,
        str(LEWM_SCRIPT),
        f'--config-name={config_name}',
        f'embed_dim={embed_dim}',
        f'output_model_name={run_name}',
        f'wandb.name={run_name}',
        *overrides,
    ]
    if track:
        cmd.append('--track')
    if wandb_project:
        cmd.extend(['--project', wandb_project])

    print(f'\n=== embed_dim={embed_dim} ({run_name}) ===')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Grid search LeWM embed_dim via lewm.py'
    )
    parser.add_argument(
        '--config-name',
        default=DEFAULT_CONFIG_NAME,
        help='Hydra config with embed_dim list and output_model_name_prefix',
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
        'overrides',
        nargs='*',
        help='Extra Hydra overrides passed to each lewm.py run',
    )
    args = parser.parse_args(argv)

    cfg = load_search_config(args.config_name)
    prefix = cfg.output_model_name_prefix
    embed_dims = iter_embed_dims(cfg.embed_dim)
    wandb_project = args.project or cfg.wandb.get('project')

    print(
        f'Starting grid search: prefix={prefix}, '
        f'embed_dims={embed_dims}, project={wandb_project}'
    )

    for embed_dim in embed_dims:
        run_name = build_run_name(prefix, embed_dim)
        run_lewm(
            config_name=args.config_name,
            embed_dim=embed_dim,
            run_name=run_name,
            wandb_project=wandb_project,
            track=args.track,
            overrides=args.overrides,
        )


if __name__ == '__main__':
    main()
