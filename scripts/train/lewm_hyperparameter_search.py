"""Grid search over LeWM hyperparameters by launching lewm.py once per combo."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

from lewm import build_lewm_run_name

SCRIPT_DIR = Path(__file__).resolve().parent
LEWM_SCRIPT = SCRIPT_DIR / 'lewm.py'
DEFAULT_CONFIG_NAME = 'lewm_mountain_car_state_hyperparameter_search'


def load_search_config(config_name: str):
    config_path = SCRIPT_DIR / 'config' / f'{config_name}.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    return OmegaConf.load(config_path)


def iter_search_values(value):
    if isinstance(value, (list, ListConfig)):
        return list(value)
    if value is None:
        return [None]
    return [value]


def is_head_enabled(cfg, section: str) -> bool:
    return bool(OmegaConf.select(cfg, f'{section}.enabled', default=False))


def get_embed_dims(cfg):
    return iter_search_values(cfg.get('embed_dim'))


def get_rl_prediction_heads_input_modes(cfg):
    reward_on = is_head_enabled(cfg, 'reward_prediction')
    continue_on = is_head_enabled(cfg, 'continue_prediction')
    if not reward_on and not continue_on:
        return [None]
    if 'rl_prediction_heads_input' in cfg:
        return iter_search_values(cfg.rl_prediction_heads_input)
    mode = OmegaConf.select(cfg, 'reward_prediction.mode', default=None)
    if mode is None:
        mode = OmegaConf.select(cfg, 'continue_prediction.mode', default=None)
    return iter_search_values(mode)


def build_run_name(
    prefix: str,
    embed_dim: int,
    heads_input: str | None,
    reward_enabled: bool,
    continue_enabled: bool,
    reward_mode: str | None = None,
    continue_mode: str | None = None,
) -> str:
    return build_lewm_run_name(
        prefix=prefix,
        embed_dim=embed_dim,
        reward_enabled=reward_enabled,
        continue_enabled=continue_enabled,
        reward_mode=reward_mode,
        continue_mode=continue_mode,
        heads_input=heads_input,
    )


def run_lewm(
    config_name: str,
    embed_dim: int,
    run_name: str,
    wandb_project: str | None,
    track: bool,
    heads_input: str | None,
    reward_enabled: bool,
    continue_enabled: bool,
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
    if reward_enabled and heads_input is not None:
        cmd.extend(
            [
                'reward_prediction.enabled=true',
                f'reward_prediction.mode={heads_input}',
            ]
        )
    if continue_enabled and heads_input is not None:
        cmd.extend(
            [
                'continue_prediction.enabled=true',
                f'continue_prediction.mode={heads_input}',
            ]
        )
    if track:
        cmd.append('--track')
    if wandb_project:
        cmd.extend(['--project', wandb_project])

    label = f'embed_dim={embed_dim}'
    if heads_input is not None:
        label += f', rl_prediction_heads_input={heads_input}'
    print(f'\n=== {label} ({run_name}) ===')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Grid search LeWM hyperparameters via lewm.py'
    )
    parser.add_argument(
        '--config-name',
        default=DEFAULT_CONFIG_NAME,
        help='Hydra config with search grids and output_model_name_prefix',
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
    embed_dims = get_embed_dims(cfg)
    heads_input_modes = get_rl_prediction_heads_input_modes(cfg)
    reward_enabled = is_head_enabled(cfg, 'reward_prediction')
    continue_enabled = is_head_enabled(cfg, 'continue_prediction')
    wandb_project = args.project or cfg.wandb.get('project')

    print(
        f'Starting grid search: prefix={prefix}, '
        f'embed_dims={embed_dims}, '
        f'reward_prediction.enabled={reward_enabled}, '
        f'continue_prediction.enabled={continue_enabled}, '
        f'rl_prediction_heads_input={heads_input_modes}, '
        f'project={wandb_project}'
    )

    for embed_dim, heads_input in itertools.product(embed_dims, heads_input_modes):
        run_name = build_run_name(
            prefix,
            embed_dim,
            heads_input,
            reward_enabled,
            continue_enabled,
            reward_mode=cfg.reward_prediction.get('mode'),
            continue_mode=cfg.continue_prediction.get('mode'),
        )
        run_lewm(
            config_name=args.config_name,
            embed_dim=embed_dim,
            run_name=run_name,
            wandb_project=wandb_project,
            track=args.track,
            heads_input=heads_input,
            reward_enabled=reward_enabled,
            continue_enabled=continue_enabled,
            overrides=args.overrides,
        )


if __name__ == '__main__':
    main()
