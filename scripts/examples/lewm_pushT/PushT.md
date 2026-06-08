## What I am trying to do?

I am trying to create a whole process of training a world model:
    - Creating dataset
    - Training WorldModel
    - Planning

## Collecting Data for the WorldModel

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/data
python3 collect_weak_pusht.py
```

## Training world model

1. Download expert PushT from Hugging Face and convert to Lance (training expects `pusht_expert_train.lance` under `~/.stable_worldmodel/datasets/`):

```bash
mkdir -p ~/.stable_worldmodel/datasets

# download (~13 GB). `huggingface-cli` is deprecated — use `hf` instead:
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst \
  --repo-type dataset \
  --local-dir ~/.stable_worldmodel/datasets

zstd -d -f -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  ~/.stable_worldmodel/datasets/pusht_expert_train.h5.zst

# HDF5 → Lance (scripts/data/convert.py requires --source and --dest)
cd /home/mamad/PhD/stable-worldmodel/scripts/data
python3 convert.py \
  --source ~/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  --dest ~/.stable_worldmodel/datasets/pusht_expert_train.lance \
  --dest-format lance \
  --mode overwrite
```

2. Train LeWM (use the local 2×4090 profile — the default `lewm.yaml` settings
   are tuned for larger clusters and can OOM a workstation):

```bash
cd /home/mamad/PhD/stable-worldmodel/scripts/train
python3 lewm.py --config-name lewm_local data=pusht
```

`lewm_local.yaml` sets `batch_size=48` per GPU, `num_workers=2`,
`prefetch_factor=2`, `persistent_workers=False`, and clears `keys_to_cache`
(Lance already does efficient random access).

If training is still unstable, scale down further before adding the second GPU:

```bash
python3 lewm.py --config-name lewm_local data=pusht \
  trainer.devices=1 loader.batch_size=32 loader.num_workers=0
```

After a crash, confirm OOM in kernel logs:

```bash
dmesg -T | tail -30
# look for: "Out of memory: Killed process ..."
```

To wathch GPU performance:
```bash
watch -n1 nvidia-smi
```

The model will be saved here
```bash
/home/mamad/.stable_worldmodel/checkpoints/lewm/
```

## Evaluting the world model
```bash
python eval_wm.py \
  policy=lewm/weights_epoch_100.pt \
  bf16=true \
  eval.num_eval=50
```
