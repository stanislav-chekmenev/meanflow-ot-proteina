# Periodic-every-N-step Retained Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third checkpoint callback that fires every N optimizer steps and retains the M most-recent firings, alongside the existing `last.ckpt` and top-K-by-val-loss callbacks. Wire the existing-but-dead `log.checkpoint_every_n_steps` config key.

**Architecture:** Extend `_build_ckpt_callbacks` in `proteinfoundation/train.py` to append a third `EmaModelCheckpoint` when `cfg_log.checkpoint_every_n_steps > 0`, using `every_n_train_steps=N`, `monitor="step"`, `mode="max"`, `save_top_k=keep_last_periodic`, and a distinct `periodic_…` filename prefix. Add a new `keep_last_periodic` config key (default 5). Read both with `.get(...)` so older configs without them still work.

**Tech Stack:** PyTorch Lightning `ModelCheckpoint` (via `EmaModelCheckpoint` subclass), Hydra/OmegaConf, pytest.

---

## File Structure

- **Modify:** `proteinfoundation/train.py` — extend `_build_ckpt_callbacks` (~lines 119-176) to build the periodic callback. Update its docstring to describe the third callback.
- **Modify:** `configs/experiment_config/training_ca.yaml` — keep `checkpoint_every_n_steps: 10000` (already there), add `keep_last_periodic: 5`.
- **Modify:** `configs/experiment_config/training_ca_motif.yaml` — already has `checkpoint_every_n_steps: 10000`; add `keep_last_periodic: 5`.
- **Modify:** `configs/experiment_config/training_ca_debug.yaml` — add `checkpoint_every_n_steps` and `keep_last_periodic` for schema parity (`checkpoint: False` means they're inert, but parity matters per the project memory).
- **Modify:** `tests/test_ckpt_topk_by_val_loss.py` — extend `_make_log_cfg` and add three new test functions.

No new files.

---

## Task 1: Extend the test helper to accept the new config keys

**Files:**
- Modify: `tests/test_ckpt_topk_by_val_loss.py:31-38`

The existing `_make_log_cfg` helper must accept and propagate `checkpoint_every_n_steps` and `keep_last_periodic` so the new tests can drive `_build_ckpt_callbacks`. We must keep the old default for `ckpt_every` so existing tests still pass — they call `_make_log_cfg(top_k)` with no kwargs.

Critical: the existing tests pass `ckpt_every=10000` by default but never read it back, because `_build_ckpt_callbacks` doesn't currently look at it. After this plan lands, `checkpoint_every_n_steps > 0` will create a third callback. So the default in `_make_log_cfg` must be **0** going forward (so callers that don't opt in still get the old 1-or-2-callback behaviour). Tests that already exist do not assert callback list lengths beyond the top-K case, so flipping the default to 0 won't break them — verified below before changing.

- [ ] **Step 1: Inspect existing assertions to confirm flipping the default is safe**

Run: `grep -n "len(cbs" tests/test_ckpt_topk_by_val_loss.py`

Expected output (the assertions on list length):

```
96:    assert len(cbs_zero) == 1
103:    assert len(cbs_three) == 2
130:    assert len(cbs) == 2
```

These all assume the periodic callback is OFF (lengths 1 and 2). With `ckpt_every` default flipped to `0`, the periodic callback won't be added, so `len(cbs_zero) == 1` and `len(cbs_three) == 2` still hold. Safe to proceed.

- [ ] **Step 2: Update the helper signature and defaults**

Replace lines 31-38 of `tests/test_ckpt_topk_by_val_loss.py`:

```python
def _make_log_cfg(
    top_k: int,
    *,
    ckpt_every: int = 0,
    last_every: int = 3000,
    keep_last_periodic: int = 5,
):
    return OmegaConf.create(
        {
            "checkpoint_every_n_steps": ckpt_every,
            "last_ckpt_every_n_steps": last_every,
            "top_k_by_val_loss": top_k,
            "keep_last_periodic": keep_last_periodic,
        }
    )
```

Note: the **default** of `ckpt_every` flips from `10000` to `0` so the periodic callback is opt-in within tests. The new key `keep_last_periodic` defaults to `5`, matching the production config.

- [ ] **Step 3: Run all existing tests to confirm no regression**

Activate the venv and set PYTHONPATH per project memory:

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py -v
```

Expected: all 7 existing tests still pass. The helper change is signature-compatible and the new default of `ckpt_every=0` does not add any new callbacks under existing tests (they were all already not asserting periodic-callback presence).

- [ ] **Step 4: Commit**

```bash
git add tests/test_ckpt_topk_by_val_loss.py
git commit -m "test(ckpt): extend _make_log_cfg with periodic-ckpt knobs"
```

---

## Task 2: Failing test — periodic enabled produces a third callback with correct args

**Files:**
- Modify: `tests/test_ckpt_topk_by_val_loss.py` (append a new test at the bottom)

- [ ] **Step 1: Append the new test**

Add the following at the end of `tests/test_ckpt_topk_by_val_loss.py`:

```python
# ---------------------------------------------------------------------------
# Test 8: checkpoint_every_n_steps > 0 adds a third "periodic" callback
# ---------------------------------------------------------------------------


def test_periodic_callback_added_when_checkpoint_every_n_steps_positive(tmp_path):
    """checkpoint_every_n_steps > 0 must produce a 3rd callback with the right knobs.

    Filename starts with 'periodic_', monitor='step' (mode='max') so Lightning
    keeps the M most-recent step-triggered ckpts when save_top_k=M.
    """
    from proteinfoundation.train import _build_ckpt_callbacks
    from proteinfoundation.utils.ema_utils.ema_callback import EmaModelCheckpoint

    cfg = _make_log_cfg(top_k=3, ckpt_every=2000, keep_last_periodic=4)
    cbs = _build_ckpt_callbacks(str(tmp_path), cfg)

    assert len(cbs) == 3, f"expected 3 callbacks; got {len(cbs)}"
    assert all(isinstance(c, EmaModelCheckpoint) for c in cbs)

    periodic_cb = next(
        c for c in cbs if c.filename and c.filename.startswith("periodic_")
    )
    assert periodic_cb._every_n_train_steps == 2000
    assert periodic_cb.save_top_k == 4
    assert periodic_cb.monitor == "step"
    assert periodic_cb.mode == "max"
    assert periodic_cb.save_last is False
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py::test_periodic_callback_added_when_checkpoint_every_n_steps_positive -v
```

Expected: FAIL with `assert len(cbs) == 3` failure (currently `_build_ckpt_callbacks` returns 2 callbacks for `top_k=3`, no periodic logic exists yet). `StopIteration` from the `next(...)` is also acceptable as the failure mode if the length assertion is changed in test maintenance.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_ckpt_topk_by_val_loss.py
git commit -m "test(ckpt): add failing test for periodic ckpt callback"
```

---

## Task 3: Implement the periodic callback in `_build_ckpt_callbacks`

**Files:**
- Modify: `proteinfoundation/train.py:119-176`

- [ ] **Step 1: Update the `_build_ckpt_callbacks` body to add the periodic callback**

Replace the body of `_build_ckpt_callbacks` (lines 119-176 in `proteinfoundation/train.py`) with:

```python
def _build_ckpt_callbacks(checkpoint_path_store, cfg_log):
    """Build the list of checkpoint callbacks.

    Always returns the "last" ckpt callback (overwrites itself every
    ``last_ckpt_every_n_steps`` steps, used for requeuing).

    If ``cfg_log.checkpoint_every_n_steps`` > 0, additionally returns a
    periodic callback that fires every N optimizer steps and retains the
    ``keep_last_periodic`` most-recent firings (via ``monitor="step",
    mode="max", save_top_k=keep_last_periodic``). Filename prefix is
    ``periodic_`` so it cannot collide with ``last.ckpt`` or the top-k file.

    If ``cfg_log.top_k_by_val_loss`` > 0, additionally returns a top-k
    callback that monitors ``val/raw_loss_mf_epoch`` (lower-is-better) and
    keeps the best K checkpoints. ``val/raw_loss_mf_epoch`` is the core
    MeanFlow objective logged with ``on_epoch=True, sync_dist=True`` in
    ``model_trainer_base.py``; Lightning emits it into ``callback_metrics``
    automatically at validation-end on every DDP rank — no custom broadcast
    needed. Lightning evaluates the monitor at validation-end, which is the
    natural cadence; do NOT set ``every_n_train_steps`` on the top-k callback
    — it would couple to train steps and fire before the monitor is populated.

    Args:
        checkpoint_path_store: directory where checkpoints are written.
        cfg_log: OmegaConf-like log block with keys
            ``last_ckpt_every_n_steps``, ``checkpoint_every_n_steps``,
            ``keep_last_periodic`` (default 5), ``top_k_by_val_loss``
            (default 3).

    Returns:
        list of EmaModelCheckpoint callbacks (length 1, 2, or 3).
    """
    args_ckpt_last = {
        "dirpath": checkpoint_path_store,
        "save_weights_only": False,
        "filename": "ignore",
        "every_n_train_steps": cfg_log.last_ckpt_every_n_steps,
        "save_last": True,
    }
    callbacks = [EmaModelCheckpoint(**args_ckpt_last)]

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

    top_k = int(cfg_log.get("top_k_by_val_loss", 3))
    if top_k > 0:
        args_ckpt_topk = {
            "dirpath": checkpoint_path_store,
            "save_last": False,
            "save_weights_only": False,
            "filename": "chk_{epoch:08d}_{step:012d}_{val/raw_loss_mf_epoch:.6f}",
            "monitor": "val/raw_loss_mf_epoch",
            "mode": "min",
            "save_top_k": top_k,
            "save_on_train_epoch_end": False,
        }
        callbacks.append(EmaModelCheckpoint(**args_ckpt_topk))

    # Build a single INFO log line summarising what was configured.
    parts = [f"last ckpt every {cfg_log.last_ckpt_every_n_steps} steps"]
    if periodic_every > 0:
        parts.append(
            f"periodic ckpt every {periodic_every} steps "
            f"(keep last {int(cfg_log.get('keep_last_periodic', 5))})"
        )
    if top_k > 0:
        parts.append(f"top-{top_k} by val/raw_loss_mf_epoch (mode=min)")
    log_info("Checkpointing: " + " + ".join(parts) + ".")

    return callbacks
```

Key points in this implementation:
- The periodic block is inserted **between** the `last` callback and the top-k callback so callback ordering matches the spec/test (last, periodic, top-k).
- Both new keys are read via `cfg_log.get(..., default)` so older configs without them still work.
- The original two `log_info(...)` branches are merged into one structured line — easier to scan and avoids three separate emissions.

- [ ] **Step 2: Run the new failing test — it should now pass**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py::test_periodic_callback_added_when_checkpoint_every_n_steps_positive -v
```

Expected: PASS.

- [ ] **Step 3: Run the full file to confirm no regressions**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py -v
```

Expected: all 8 tests pass (the original 7 + the new one).

- [ ] **Step 4: Commit**

```bash
git add proteinfoundation/train.py
git commit -m "feat(ckpt): add periodic every-N-step retained checkpoints"
```

---

## Task 4: Failing test — periodic disabled when `checkpoint_every_n_steps <= 0`

**Files:**
- Modify: `tests/test_ckpt_topk_by_val_loss.py` (append)

This test guards the off-switch contract: setting `checkpoint_every_n_steps: 0` (or negative) must skip creating the periodic callback. It must already be green after Task 3 (the off-switch logic is already in the implementation), so the TDD ordering here is "explicit regression-guard test" rather than "failing-first." We still write it, run it, see it pass.

- [ ] **Step 1: Append the new test**

Add to the bottom of `tests/test_ckpt_topk_by_val_loss.py`:

```python
# ---------------------------------------------------------------------------
# Test 9: checkpoint_every_n_steps <= 0 disables the periodic callback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ckpt_every", [0, -1])
def test_periodic_callback_disabled_when_checkpoint_every_n_steps_le_zero(
    tmp_path, ckpt_every
):
    """checkpoint_every_n_steps <= 0 must NOT add a periodic-prefix callback.

    Length matches the prior 'top-K only' world: 1 (last) + 1 (top-k) = 2.
    """
    from proteinfoundation.train import _build_ckpt_callbacks

    cfg = _make_log_cfg(top_k=3, ckpt_every=ckpt_every)
    cbs = _build_ckpt_callbacks(str(tmp_path), cfg)

    assert len(cbs) == 2, (
        f"expected 2 callbacks (last + top-k) when ckpt_every={ckpt_every}; "
        f"got {len(cbs)}"
    )
    periodic_filenames = [
        c.filename for c in cbs if c.filename and c.filename.startswith("periodic_")
    ]
    assert periodic_filenames == [], (
        f"no callback should have a 'periodic_' filename; got {periodic_filenames}"
    )
```

- [ ] **Step 2: Run the new test**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py::test_periodic_callback_disabled_when_checkpoint_every_n_steps_le_zero -v
```

Expected: PASS for both parametrized cases (`0` and `-1`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ckpt_topk_by_val_loss.py
git commit -m "test(ckpt): periodic ckpt off-switch regression guard"
```

---

## Task 5: Failing test — all three callbacks coexist with distinct filename prefixes

**Files:**
- Modify: `tests/test_ckpt_topk_by_val_loss.py` (append)

- [ ] **Step 1: Append the new test**

Add:

```python
# ---------------------------------------------------------------------------
# Test 10: all three callbacks coexist with distinct filename prefixes
# ---------------------------------------------------------------------------


def test_three_callbacks_have_distinct_filename_prefixes(tmp_path):
    """When all three callback knobs are on, no two callbacks share a filename pattern.

    Filename templates must be distinguishable so saved files cannot collide:
      - last.ckpt  (save_last=True; filename='ignore' is unused for last.ckpt)
      - periodic_…
      - chk_…
    """
    from proteinfoundation.train import _build_ckpt_callbacks

    cfg = _make_log_cfg(top_k=3, ckpt_every=2000, keep_last_periodic=4)
    cbs = _build_ckpt_callbacks(str(tmp_path), cfg)
    assert len(cbs) == 3

    # Identify each callback by a stable trait.
    last_cb = next(c for c in cbs if c.save_last is True)
    periodic_cb = next(
        c for c in cbs if c.monitor == "step"
    )
    topk_cb = next(c for c in cbs if c.monitor == "val/raw_loss_mf_epoch")

    # Sanity: they're three distinct objects.
    assert {id(last_cb), id(periodic_cb), id(topk_cb)} == {id(c) for c in cbs}

    # Filename prefix uniqueness (last_cb's 'filename' is unused for last.ckpt
    # but should still not collide with the others).
    assert periodic_cb.filename.startswith("periodic_")
    assert topk_cb.filename.startswith("chk_")
    assert not last_cb.filename.startswith("periodic_")
    assert not last_cb.filename.startswith("chk_")
```

- [ ] **Step 2: Run the new test**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py::test_three_callbacks_have_distinct_filename_prefixes -v
```

Expected: PASS.

- [ ] **Step 3: Run the full test file to confirm no regressions**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py -v
```

Expected: 10 tests pass (7 original + 3 new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_ckpt_topk_by_val_loss.py
git commit -m "test(ckpt): three-callback coexistence + filename uniqueness"
```

---

## Task 6: Wire the new key into `training_ca.yaml`

**Files:**
- Modify: `configs/experiment_config/training_ca.yaml:87`

- [ ] **Step 1: Add `keep_last_periodic` next to `top_k_by_val_loss`**

Edit `configs/experiment_config/training_ca.yaml`. Replace the line:

```yaml
  top_k_by_val_loss: 3  # Keep this many best-val-loss checkpoints (mode=min on val/raw_loss_mf_epoch). 0 = keep only the "last" ckpt.
```

with:

```yaml
  top_k_by_val_loss: 3  # Keep this many best-val-loss checkpoints (mode=min on val/raw_loss_mf_epoch). 0 = keep only the "last" ckpt.
  keep_last_periodic: 5  # How many most-recent periodic ckpts to retain (paired with checkpoint_every_n_steps). Periodic ckpts use filename prefix 'periodic_'.
```

Also update the comment above `checkpoint_every_n_steps` (line 85) to mention the off-switch:

Replace:

```yaml
  checkpoint_every_n_steps: 10000  # Optimizer steps (trainer.global_step) — unlike val_check_interval, NOT affected by accumulate_grad_batches
```

with:

```yaml
  checkpoint_every_n_steps: 10000  # Optimizer steps between retained periodic ckpts (trainer.global_step). 0 disables the periodic callback. NOT affected by accumulate_grad_batches.
```

- [ ] **Step 2: Verify the YAML still loads with Hydra/OmegaConf**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/experiment_config/training_ca.yaml'); print(c.log.checkpoint_every_n_steps, c.log.keep_last_periodic, c.log.top_k_by_val_loss)"
```

Expected output: `10000 5 3`

- [ ] **Step 3: Commit**

```bash
git add configs/experiment_config/training_ca.yaml
git commit -m "config(training_ca): add keep_last_periodic, doc periodic off-switch"
```

---

## Task 7: Mirror the new key into `training_ca_motif.yaml`

**Files:**
- Modify: `configs/experiment_config/training_ca_motif.yaml:72-74`

- [ ] **Step 1: Add `keep_last_periodic`**

Edit `configs/experiment_config/training_ca_motif.yaml`. Replace the line:

```yaml
  top_k_by_val_loss: 3  # Keep this many best-val-loss checkpoints (mode=min on val/raw_loss_mf_epoch). 0 = keep only the "last" ckpt.
```

with:

```yaml
  top_k_by_val_loss: 3  # Keep this many best-val-loss checkpoints (mode=min on val/raw_loss_mf_epoch). 0 = keep only the "last" ckpt.
  keep_last_periodic: 5  # How many most-recent periodic ckpts to retain (paired with checkpoint_every_n_steps). Periodic ckpts use filename prefix 'periodic_'.
```

Update the `checkpoint_every_n_steps` comment the same way as in Task 6:

```yaml
  checkpoint_every_n_steps: 10000  # Optimizer steps between retained periodic ckpts (trainer.global_step). 0 disables the periodic callback. NOT affected by accumulate_grad_batches.
```

- [ ] **Step 2: Verify the YAML still loads**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/experiment_config/training_ca_motif.yaml'); print(c.log.checkpoint_every_n_steps, c.log.keep_last_periodic, c.log.top_k_by_val_loss)"
```

Expected output: `10000 5 3`

- [ ] **Step 3: Commit**

```bash
git add configs/experiment_config/training_ca_motif.yaml
git commit -m "config(training_ca_motif): add keep_last_periodic"
```

---

## Task 8: Mirror schema into `training_ca_debug.yaml`

**Files:**
- Modify: `configs/experiment_config/training_ca_debug.yaml:74-82`

The debug config has `checkpoint: False`, so checkpoints are not actually written, but per the project memory "training_ca schema gap (2026-04-20)" we mirror keys for parity to keep the schemas aligned. Both new keys go in with comments noting they're inert in this config.

- [ ] **Step 1: Add both keys for schema parity**

Edit `configs/experiment_config/training_ca_debug.yaml`. After the existing `top_k_by_val_loss: 3` line (line 82), add:

Replace:

```yaml
  top_k_by_val_loss: 3  # Unused (checkpoint: False above); kept for schema parity with training_ca.yaml
```

with:

```yaml
  top_k_by_val_loss: 3  # Unused (checkpoint: False above); kept for schema parity with training_ca.yaml
  checkpoint_every_n_steps: 10000  # Unused (checkpoint: False above); kept for schema parity with training_ca.yaml
  keep_last_periodic: 5  # Unused (checkpoint: False above); kept for schema parity with training_ca.yaml
  last_ckpt_every_n_steps: 5000  # Unused (checkpoint: False above); kept for schema parity with training_ca.yaml
```

`last_ckpt_every_n_steps` is also added because it was missing — the `_build_ckpt_callbacks` function reads it as a hard attribute (`cfg_log.last_ckpt_every_n_steps`, no `.get`) and would crash on a debug run that exercises the helper, even though `checkpoint=False` means the helper is currently skipped. Adding it now closes the schema gap pre-emptively.

- [ ] **Step 2: Verify the YAML still loads**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/experiment_config/training_ca_debug.yaml'); print(c.log.checkpoint_every_n_steps, c.log.keep_last_periodic, c.log.last_ckpt_every_n_steps, c.log.top_k_by_val_loss)"
```

Expected output: `10000 5 5000 3`

- [ ] **Step 3: Commit**

```bash
git add configs/experiment_config/training_ca_debug.yaml
git commit -m "config(training_ca_debug): mirror periodic ckpt + last-ckpt keys for schema parity"
```

---

## Task 9: End-to-end verification — full test file passes, INFO log line is correct

**Files:** none (verification only)

- [ ] **Step 1: Run the full ckpt test file**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD pytest tests/test_ckpt_topk_by_val_loss.py -v
```

Expected: 10 tests pass (test 9 is parametrized into 2 cases, so pytest will report 11 passed in some pytest versions; either way, all green).

- [ ] **Step 2: Verify the INFO log line emits the expected three-part summary**

```bash
source .venv/bin/activate && PYTHONPATH=$PWD python -c "
from omegaconf import OmegaConf
from proteinfoundation.train import _build_ckpt_callbacks
import tempfile, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
cfg = OmegaConf.create({
    'checkpoint_every_n_steps': 2000,
    'last_ckpt_every_n_steps': 3000,
    'top_k_by_val_loss': 3,
    'keep_last_periodic': 4,
})
with tempfile.TemporaryDirectory() as d:
    cbs = _build_ckpt_callbacks(d, cfg)
    print('num_callbacks=', len(cbs))
"
```

Expected: an INFO line of the form

```
Checkpointing: last ckpt every 3000 steps + periodic ckpt every 2000 steps (keep last 4) + top-3 by val/raw_loss_mf_epoch (mode=min).
num_callbacks= 3
```

(`log_info` may write to stdout or stderr depending on project setup — either is fine; the substring `periodic ckpt every 2000 steps` and `(keep last 4)` must appear.)

- [ ] **Step 3: No commit required (verification only)**

This task is the finish line — no code change.

---

## Self-Review Notes

- **Spec coverage:** Config changes (training_ca + motif + debug) → Tasks 6/7/8. Code change → Task 3. Tests (3 new cases) → Tasks 2/4/5. Docstring update → Task 3. Wiring `checkpoint_every_n_steps` → Task 3. Adding `keep_last_periodic` knob → Tasks 3/6/7/8. INFO log update → Task 3. End-to-end verification → Task 9. All spec sections covered.
- **Placeholder scan:** No TBD/TODO/“implement later”. All code blocks are concrete.
- **Type consistency:** `keep_last_periodic` is read with `.get(..., 5)` in Task 3; defaulted to `5` in `_make_log_cfg` (Task 1) and in all configs (Tasks 6/7/8). `checkpoint_every_n_steps` is read with `.get(..., 0)` and the off-switch uses the same `> 0` check in both code (Task 3) and tests (Task 4). Filename prefixes: `periodic_` in code (Task 3) and asserted in tests (Tasks 2/4/5).
- **Out-of-scope guardrails:** `EmaModelCheckpoint` itself is untouched; resume logic is untouched; `train.sbatch` is untouched.
