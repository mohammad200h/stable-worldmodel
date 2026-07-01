"""Grid search over embed_dim and rl_prediction_heads_input via train_rl_plan.py."""

import argparse
import itertools
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


def iter_search_values(value):
    if isinstance(value, (list, ListConfig)):
        return list(value)
    if value is None:
        return [None]
    return [value]


def get_embed_dims(cfg):
    return iter_search_values(cfg.get('embed_dim'))


def get_rl_prediction_heads_input_modes(cfg):
    if 'rl_prediction_heads_input' in cfg:
        return iter_search_values(cfg.rl_prediction_heads_input)
    return [None]


def resolve_run_settings(
    cfg,
    embed_dim: int,
    rl_prediction_heads_input: str | None,
) -> dict:
    overrides = {'embed_dim': embed_dim}
    if rl_prediction_heads_input is not None:
        overrides['rl_prediction_heads_input'] = rl_prediction_heads_input
    run_cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    OmegaConf.resolve(run_cfg)

    wm_checkpoint_path = Path(run_cfg.world_model_path)
    return {
        'embed_dim': embed_dim,
        'rl_prediction_heads_input': rl_prediction_heads_input,
        'world_model_path': str(wm_checkpoint_path.parent),
        'checkpoint': run_cfg.checkpoint,
        'embedding_is_made_of_pixels': run_cfg.embedding_is_made_of_pixels,
        'hybrid_mode': run_cfg.get('hybrid_mode', True),
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
    if settings['hybrid_mode']:
        cmd.append('--hybrid-mode')
    else:
        cmd.append('--no-hybrid-mode')
    if track:
        cmd.append('--track')

    label = f'embed_dim={settings["embed_dim"]}'
    if settings['rl_prediction_heads_input'] is not None:
        label += f', rl_prediction_heads_input={settings["rl_prediction_heads_input"]}'
    print(f'\n=== {label} ({settings["model_name"]}) ===')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Grid search RL-plan embed_dim and rl_prediction_heads_input '
            'via train_rl_plan.py'
        )
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help='Search config with embed_dim / rl_prediction_heads_input grids',
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
    embed_dims = get_embed_dims(cfg)
    heads_input_modes = get_rl_prediction_heads_input_modes(cfg)
    hybrid_mode = cfg.get('hybrid_mode', True)
    wandb_project = args.project or cfg.wandb.get('project')

    print(
        f'Starting grid search: embed_dims={embed_dims}, '
        f'rl_prediction_heads_input={heads_input_modes}, '
        f'hybrid_mode={hybrid_mode}, project={wandb_project}'
    )

    for embed_dim, heads_input in itertools.product(embed_dims, heads_input_modes):
        settings = resolve_run_settings(cfg, embed_dim, heads_input)
        if wandb_project:
            settings['wandb_project'] = wandb_project
        run_train_rl_plan(settings, track=args.track, train_args=args.train_args)


if __name__ == '__main__':
    main()
