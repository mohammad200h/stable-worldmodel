import hydra
import stable_worldmodel  # registers swm/* envs for the underlying Fetch env

from env import FetchWorldModelEnv

@hydra.main(version_base=None, config_path='./', config_name='env_demo')
def main(cfg):
    env = FetchWorldModelEnv(cfg.world_model_path, cfg.checkpoint)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)

    print(f"next obs embedding shape: {next_obs['observation'].shape}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    # print(f"truncated: {truncated}")
    # print(f"info: {info}")

    
    
 
    env.close()

if __name__ == "__main__":
    main()