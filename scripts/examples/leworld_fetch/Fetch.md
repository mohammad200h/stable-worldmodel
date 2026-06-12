# What I am trying to do?
1. I am trying to train an RL policy that can solve Fetch
2. I am going to use that policy to collect data
3. I am then going to use that data to train a leworld model
4. I am going to evalute the performance

5. I am going to wirte a new version of leworld model that only takes proprioception
6. Train it using collected data
7. evaluate 

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
/home/mamad/.stable_worldmodel/datasets/fetch_reach_expert.lance
```