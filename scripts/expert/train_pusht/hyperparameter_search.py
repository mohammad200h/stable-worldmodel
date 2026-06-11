"""W&B hyperparameter sweep for PushT vision SAC training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import wandb

from train_pusht_policy import CONFIG_PATH, load_config, train_expert


SCRIPT_DIR = Path(__file__).resolve().parent
BEST_CONFIG_PATH = SCRIPT_DIR / 'best_config.json'

SWEEP_PARAM_MAP: dict[str, tuple[str, str]] = {
    'learning_rate': ('sac', 'learning_rate'),
    'learning_starts': ('sac', 'learning_starts'),
    'buffer_size': ('sac', 'buffer_size'),
    'batch_size': ('sac', 'batch_size'),
    'resolution': ('env', 'resolution'),
    'grayscale': ('env', 'grayscale'),
    'n_envs': ('env', 'n_envs'),
    'total_timesteps': ('training', 'total_timesteps'),
}

DEFAULT_SWEEP_CONFIG: dict[str, Any] = {
    'method': 'bayes',
    'metric': {
        'name': 'best_eval_success_rate',
        'goal': 'maximize',
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'learning_starts': {'values': [10000, 25000, 50000]},
        # Vision replay buffers are RAM-heavy: ~0.5 MB per transition at 128px RGB stack.
        'buffer_size': {'values': [50000, 100000, 200000]},
        'batch_size': {'values': [128, 256, 512]},
        'resolution': {'values': [64, 96, 128]},
        'grayscale': {'values': [True, False]},
        'n_envs': {'values': [8, 16]},
        'total_timesteps': {'value': 200000},
    },
}


def build_sweep_config(
    *,
    search_timesteps: int,
    metric_name: str = 'best_eval_success_rate',
) -> dict[str, Any]:
    sweep_config = copy.deepcopy(DEFAULT_SWEEP_CONFIG)
    sweep_config['metric']['name'] = metric_name
    sweep_config['parameters']['total_timesteps'] = {'value': search_timesteps}
    return sweep_config


def apply_sweep_params(
    base_config: dict,
    sweep_params: dict[str, Any],
) -> dict:
    config = copy.deepcopy(base_config)
    for sweep_key, value in sweep_params.items():
        if sweep_key not in SWEEP_PARAM_MAP:
            continue
        section, field = SWEEP_PARAM_MAP[sweep_key]
        config[section][field] = value
    return config


def config_to_wandb_params(config: dict) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for sweep_key, (section, field) in SWEEP_PARAM_MAP.items():
        if field in config.get(section, {}):
            params[sweep_key] = config[section][field]
    return params


def write_best_config(
    *,
    base_config: dict,
    best_run: Any,
    sweep_id: str,
    output_path: Path,
) -> dict:
    best_config = apply_sweep_params(base_config, dict(best_run.config))
    payload = copy.deepcopy(best_config)
    payload['_search_metadata'] = {
        'best_eval_success_rate': best_run.summary.get(
            'best_eval_success_rate',
            best_run.summary.get('eval/success_rate'),
        ),
        'final_eval_success_rate': best_run.summary.get(
            'final_eval_success_rate'
        ),
        'best_eval_mean_reward': best_run.summary.get('best_eval_mean_reward'),
        'final_eval_mean_reward': best_run.summary.get(
            'final_eval_mean_reward'
        ),
        'wandb_run_id': best_run.id,
        'wandb_run_name': best_run.name,
        'wandb_run_url': best_run.url,
        'sweep_id': sweep_id,
        'sweep_url': best_run.sweep.url if best_run.sweep is not None else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    return payload


def log_best_config_to_wandb(
    *,
    project: str,
    entity: str | None,
    sweep_id: str,
    best_config_payload: dict,
    output_path: Path,
) -> None:
    metadata = best_config_payload['_search_metadata']
    with wandb.init(
        project=project,
        entity=entity,
        job_type='sweep_best_config',
        name=f'best_config_{sweep_id}',
        config=config_to_wandb_params(best_config_payload),
    ) as run:
        run.summary.update(metadata)
        wandb.log(
            {
                'best_eval_success_rate': metadata.get(
                    'best_eval_success_rate', 0.0
                ),
                'best_eval_mean_reward': metadata.get('best_eval_mean_reward'),
            }
        )
        artifact = wandb.Artifact(
            name=f'best_config_{sweep_id}',
            type='config',
            description='Best PushT SAC config from hyperparameter sweep',
        )
        artifact.add_file(str(output_path))
        run.log_artifact(artifact)


def fetch_best_run(api: wandb.Api, entity: str, project: str, sweep_id: str):
    sweep_path = f'{entity}/{project}/{sweep_id}'
    sweep = api.sweep(sweep_path)
    best_run = sweep.best_run()
    if best_run is None:
        raise RuntimeError(f'No completed runs found for sweep {sweep_path}')
    return best_run


def make_train_fn(
    *,
    base_config: dict,
    project: str,
    record_eval_video: bool,
):
    def train() -> float:
        run = wandb.init()
        assert run is not None

        trial_config = apply_sweep_params(base_config, dict(wandb.config))
        env_id = trial_config['env']['id'].replace('/', '_')
        run_name = f'sweep_{env_id}_{run.id}'
        save_path = f'./policies/sweep_{env_id}_{run.id}'

        metrics = train_expert(
            trial_config,
            track=True,
            project_name=project,
            run_name=run_name,
            save_path=save_path,
            record_eval_video=record_eval_video,
            return_metrics=True,
        )
        assert metrics is not None

        best_success_rate = float(metrics['best_eval_success_rate'])
        wandb.log({'eval/success_rate': metrics['final_eval_success_rate']})
        return best_success_rate

    return train


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run a W&B hyperparameter sweep for PushT SAC training'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=CONFIG_PATH,
        help='Base JSON config merged with sweep parameters',
    )
    parser.add_argument(
        '--project',
        type=str,
        default='stable-worldmodel',
        help='W&B project name',
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='W&B entity (username or team). Defaults to the logged-in user.',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='Number of sweep trials to run',
    )
    parser.add_argument(
        '--search-timesteps',
        type=int,
        default=200000,
        help='Training timesteps per sweep trial',
    )
    parser.add_argument(
        '--sweep-id',
        type=str,
        default=None,
        help='Existing sweep ID to continue instead of creating a new sweep',
    )
    parser.add_argument(
        '--best-config',
        type=Path,
        default=BEST_CONFIG_PATH,
        help='Where to write the best config after the sweep finishes',
    )
    parser.add_argument(
        '--no-eval-video',
        action='store_true',
        help='Disable eval video recording during sweep trials',
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    sweep_config = build_sweep_config(search_timesteps=args.search_timesteps)

    if args.sweep_id is None:
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity,
        )
        print(f'Created sweep: {sweep_id}')
    else:
        sweep_id = args.sweep_id
        print(f'Using existing sweep: {sweep_id}')

    train_fn = make_train_fn(
        base_config=base_config,
        project=args.project,
        record_eval_video=not args.no_eval_video,
    )
    wandb.agent(
        sweep_id,
        function=train_fn,
        count=args.count,
        project=args.project,
        entity=args.entity,
    )

    api = wandb.Api()
    entity = args.entity or api.default_entity
    best_run = fetch_best_run(api, entity, args.project, sweep_id)
    best_config_payload = write_best_config(
        base_config=base_config,
        best_run=best_run,
        sweep_id=sweep_id,
        output_path=args.best_config,
    )
    log_best_config_to_wandb(
        project=args.project,
        entity=entity,
        sweep_id=sweep_id,
        best_config_payload=best_config_payload,
        output_path=args.best_config,
    )

    metadata = best_config_payload['_search_metadata']
    print(f'Sweep complete. Best run: {metadata["wandb_run_name"]} ({metadata["wandb_run_id"]})')
    print(
        'Best eval success rate: '
        f'{100 * float(metadata.get("best_eval_success_rate") or 0):.2f}%'
    )
    print(f'Wrote {args.best_config}')


if __name__ == '__main__':
    main()
