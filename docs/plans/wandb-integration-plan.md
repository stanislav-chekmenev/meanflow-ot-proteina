# WandB Integration Improvement Plan

## Current State

The project already has basic WandB support in `proteinfoundation/train.py`:
- Imports `wandb` and `WandbLogger` from Lightning
- Config section `log.wandb_project`, `log.log_wandb` in YAML
- `WandbLogger` init with project and run id
- Config artifact logging via `log_configs()`
- Metric logging via `self.log()` in the training step

## Issues Found

1. **Bug in `log_configs`** (line 67): References global `run_name` which is undefined in function scope — should receive it as a parameter.
2. **No DDP-safe wandb init**: On multi-node SLURM jobs, non-rank-0 workers will try to init wandb and may deadlock.
3. **No error alerting**: Training crashes silently without notifying wandb dashboard.
4. **No `wandb.finish()`**: Runs may not finalize properly on crash.
5. **No `entity` config**: Cannot direct runs to a specific wandb team/entity.
6. **No `run_name` override**: Run name is always the config `run_name_`, no way to customize display name separately.
7. **No LearningRateMonitor**: LR schedule changes are not tracked.
8. **No full config logged**: Only artifacts are logged, not the resolved config dict.
9. **No CSV fallback logger**: If wandb is disabled, no logging at all.

## Plan

### Step 1: Fix `log_configs` bug
- File: `proteinfoundation/train.py`
- Pass `run_name` as a parameter to `log_configs()` instead of relying on a global.

### Step 2: Add `wandb` config subsection to YAML configs
- Files: `configs/experiment_config/training_ca.yaml`, `training_ca_debug.yaml`, `training_ca_motif.yaml`
- Add under `log:`:
  ```yaml
  wandb_entity: null    # W&B team/entity (null = personal)
  wandb_run_name: null  # Display name override (null = use run_name_)
  ```

### Step 3: DDP-safe wandb init
- File: `proteinfoundation/train.py`
- Before creating `WandbLogger`, disable wandb on non-rank-0 SLURM workers:
  ```python
  local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
  if local_rank != 0:
      os.environ["WANDB_MODE"] = "disabled"
  ```

### Step 4: Pass full resolved config to WandbLogger
- File: `proteinfoundation/train.py`
- Add `config=OmegaConf.to_container(cfg_exp, resolve=True)` to `WandbLogger()` init.
- Add `entity` and `name` params from config.

### Step 5: Add LearningRateMonitor callback
- File: `proteinfoundation/train.py`
- Import `LearningRateMonitor` from lightning
- Add to callbacks list when wandb logging is enabled.

### Step 6: Add error alerting and proper shutdown
- File: `proteinfoundation/train.py`
- Wrap `trainer.fit()` in try/except
- On exception: call `wandb.run.alert()` and `wandb.finish(exit_code=1)`

### Step 7: Add CSVLogger fallback
- File: `proteinfoundation/train.py`
- When `nolog` is set or wandb is disabled, use `CSVLogger` so metrics are still saved.

### Step 8: Add `wandb` to `.gitignore`
- File: `.gitignore`
- Add `wandb/` directory if not already present.

## Files Modified
- `proteinfoundation/train.py` (main changes)
- `configs/experiment_config/training_ca.yaml`
- `configs/experiment_config/training_ca_debug.yaml`
- `configs/experiment_config/training_ca_motif.yaml`
- `.gitignore`
