# Periodic-every-N-step retained checkpoints

**Date:** 2026-04-29
**Status:** approved, ready for implementation plan

## Problem

Today, [`_build_ckpt_callbacks`](../../../proteinfoundation/train.py) in `proteinfoundation/train.py:119-176` emits two `EmaModelCheckpoint` callbacks:

1. A `last.ckpt` callback that fires every `log.last_ckpt_every_n_steps` optimizer steps and **overwrites itself** (used for requeuing).
2. A top-K callback (when `log.top_k_by_val_loss > 0`) that monitors `val/raw_loss_mf_epoch` and keeps the K best by validation loss.

There is no way to keep a *retained* snapshot every N optimizer steps that survives independently of validation rank. Practical effect: aside from the K val-best ckpts, you can only resume from `last.ckpt` — there is no chronological trail of intermediate weights for post-hoc inspection.

A separate small problem: `log.checkpoint_every_n_steps: 10000` is present in `configs/experiment_config/training_ca.yaml:85` and `training_ca_motif.yaml:72`, but is **dead** — referenced only in `_build_ckpt_callbacks`'s docstring, never read into a callback.

## Solution

Wire `log.checkpoint_every_n_steps` into a new third `EmaModelCheckpoint` that:

- Fires every `checkpoint_every_n_steps` optimizer steps (same unit as `last_ckpt_every_n_steps`).
- Retains the M most recent firings, where M is a new config knob `log.keep_last_periodic`.
- Uses a distinct filename prefix (`periodic_…`) so it cannot collide with the existing top-K (`chk_…`) or `last.ckpt`.
- Is independent of and coexists with the existing two callbacks.

`checkpoint_every_n_steps <= 0` disables the periodic callback (early-return; no callback added).

## Config changes

Files: `configs/experiment_config/training_ca.yaml`, `training_ca_debug.yaml`, `training_ca_motif.yaml`.

Existing key, now wired (no rename):

```yaml
log:
  checkpoint_every_n_steps: 10000  # Optimizer steps between retained periodic ckpts. <=0 disables the periodic callback.
```

New key:

```yaml
log:
  keep_last_periodic: 5  # How many most-recent periodic ckpts to retain on disk.
```

`training_ca_debug.yaml` currently lacks both keys (per the existing "training_ca schema gap" memory). Both keys are added there too with values appropriate for debug runs (e.g. small `checkpoint_every_n_steps`, small `keep_last_periodic`). The exact debug values are an implementation choice; the design only requires both keys be present so the schemas stay in sync.

## Code changes

File: `proteinfoundation/train.py`, function `_build_ckpt_callbacks` (around lines 119-176).

After the existing `last` callback is appended and before the top-K block, add:

```python
periodic_every = int(cfg_log.get("checkpoint_every_n_steps", 0))
if periodic_every > 0:
    keep_last = int(cfg_log.get("keep_last_periodic", 5))
    args_ckpt_periodic = {
        "dirpath": checkpoint_path_store,
        "save_last": False,
        "save_weights_only": False,
        "filename": "periodic_{epoch:08d}_{step:012d}",
        "every_n_train_steps": periodic_every,
        "save_top_k": keep_last,
        "monitor": "step",
        "mode": "max",
        "save_on_train_epoch_end": False,
    }
    callbacks.append(EmaModelCheckpoint(**args_ckpt_periodic))
```

Why `monitor="step"` + `mode="max"` + `save_top_k=keep_last`: this is Lightning's idiomatic pattern to retain the M most-recent step-triggered checkpoints. The global `step` is always populated in `callback_metrics`, so the monitor is always available when `every_n_train_steps` fires.

Update the existing INFO log so it also reports periodic cadence + retention when enabled (and stays silent on it when disabled).

Update the docstring of `_build_ckpt_callbacks` to describe the third callback truthfully (currently the docstring mentions `checkpoint_every_n_steps` but the function never reads it).

## Tests

File: `tests/test_ckpt_topk_by_val_loss.py` (extend, do not rewrite — existing tests stay green).

Add three cases:

1. **Periodic enabled produces a third callback.** With `checkpoint_every_n_steps=2000`, `keep_last_periodic=4`, `top_k_by_val_loss=3`: callback list length == 3; the third callback has `every_n_train_steps == 2000`, `save_top_k == 4`, `monitor == "step"`, and a filename starting with `"periodic_"`.
2. **Periodic disabled.** With `checkpoint_every_n_steps=0`: no `periodic_`-filename callback in the returned list. Length matches the existing "top-K only" case.
3. **All three coexist.** Same as case 1 but assert each of the three callbacks is distinguishable by filename prefix (`last`/`chk_`/`periodic_`) — no two callbacks write to the same path pattern.

No new test file is created; the existing one is the right home (it already centralises `_build_ckpt_callbacks` test coverage).

## Out of scope

- Refactoring or renaming `last_ckpt_every_n_steps` / `top_k_by_val_loss`.
- Touching `EmaModelCheckpoint` itself.
- Resume-logic changes — the existing `resume_from_last_ckpt` flow continues to use `last.ckpt`; periodic ckpts are for inspection/branching, not the requeue path.
- Any `sbatch` script changes — the new keys live entirely inside the Hydra config and require no flags to be threaded through `train.sbatch`.

## Rollout

Single PR. No migration: the new `keep_last_periodic` key is read with `.get(..., 5)` so older configs without it still run with the documented default. `checkpoint_every_n_steps` keeps its current value of `10000` in checked-in configs, which means the moment this lands, training runs start producing one retained periodic ckpt every 10k steps (capped at `keep_last_periodic` of them).
