# What I am trying to do?
1. I am trying to train an RL policy that can solve Fetch
2. I am going to use that policy to collect data
3. I am then going to use that data to train a leworld model
4. I am going to evalute the performance

5. I am going to wirte a new version of leworld model that only takes proprioception
6. Train it using collected data
7. evaluate 

8. Implement Ideas from MOPO to predict uncertainty
7. Implement Ideas from DreamerV3 so that RL agnet can learn using worldmodel

# Train RL policy
```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/expert/train_fetch
python3 train_fetch_policy_her.py --track
```

# Collect data

Demo how expert behaves and it setup!
```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/data/collect_fetch
python3 python3 expert_policy
```

Collect data using expert
```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/data/collect_fetch
python3 collect_fetch.py
```
The data get stored in
```bash
/home/mamad/.stable_worldmodel/datasets/fetch_reach_expert_rl_agent.lance
```

# Train world model on Cluster
```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/train
python3 lewm.py data=fetch_rl # if on cluster and have beefy gpu
```

# Train world model on workstation
```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/train
python3 lewm.py --config-name lewm_local data=fetch_rl 
```

# Evaluate world model
```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/rl_plan
python train_fetch_policy_her.py \
  --env swm/FetchReachWM-v0 \
  --wm-path /home/mamad/.stable_worldmodel/checkpoints/lewm \
  --checkpoint weights_epoch_100.pt \
  --timesteps 100000
```

# To wathch GPU performance:
```bash
watch -n1 nvidia-smi
```