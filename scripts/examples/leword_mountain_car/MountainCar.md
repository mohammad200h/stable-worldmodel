# Train RL expert

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/expert/train_mountain_car
python3  train.py --track 
```

# Data Collection

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/data/collect_mountain_car
python3 collect_mountain_car.py 
python3 collect_diverse_mountain_car.py
```

# Train world model on workstation

## Pixels

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/train
python3 lewm.py --config-name lewm_local data=mountain_car_rl --track
```

## State Vector (No pixels)

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/train
python3 lewm.py --config-name lewm_state_local data=mountain_car_rl_state.yaml --track
```

## Train World Model Searching for hyperprameter doing Grid search

Grid over `embed_dim` × `rl_prediction_heads_input` for each enabled RL head (`reward_prediction`, `continue_prediction`). See `lewm_mountain_car_state_hyperparameter_search.yaml`.

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/train
python3 lewm_hyperparameter_search.py data=mountain_car_rl_diverse_state --track
```

# Train RL plan

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/rl_plan/mountain_car
python3 train_rl_plan.py --input-state-type state --track
```

## Train RL plan Searching for hyperprameter doing Grid search

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/rl_plan/mountain_car
python3 train_rl_plan_hyperparameter_search.py --track 
```

