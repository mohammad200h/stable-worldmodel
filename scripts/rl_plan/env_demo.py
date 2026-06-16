import argparse
import sys

import hydra
import stable_worldmodel  # registers swm/* envs for the underlying Fetch env

from env import FetchWorldModelEnv

@hydra.main(version_base=None, config_path='./', config_name='env_demo_pixels')
def run_demo_pixels(cfg):
    env = FetchWorldModelEnv(cfg.world_model_path, cfg.checkpoint)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)

    print(f"next obs embedding shape: {next_obs['observation'].shape}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    env.close()

@hydra.main(version_base=None, config_path='./', config_name='env_demo_state')
def run_demo_state(cfg):
    env = FetchWorldModelEnv(cfg.world_model_path, cfg.checkpoint, embedding_is_made_of_pixels=False)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"next obs embedding shape: {next_obs['observation'].shape}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    env.close()

def parse_demo_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--input-state-type',
        type=str,
        default='pixels',
        choices=['pixels', 'state'],
    )
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, hydra_args = parse_demo_args()
    sys.argv = [sys.argv[0], *hydra_args]

    if args.input_state_type == 'pixels':
        run_demo_pixels()
    elif args.input_state_type == 'state':
        run_demo_state()
    else:
        raise ValueError(
            f"Invalid input state type: {args.input_state_type} passed. "
            "Pass either 'pixels' or 'state'."
        )