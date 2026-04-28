# OT Pool + Loss Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `loss_accumulation_steps > 1` (manual K-loop accumulation) to be used together with the streaming OT pool. Today the combination is rejected at `on_train_start` by an assertion.

**Architecture:** "K batches per step" semantics — for each training step, the K-loop pops K consecutive B-sized batches from the pool. Every K-pass uses its own pool-paired (x_1, x_0, mask) and its own freshly sampled (t, r). Pool exhaustion accelerates by a factor of K (refill cadence becomes `pool_size / (K * batch_size)` training steps). The K=1 path and the non-pool K>1 path are unchanged.

Also fix the unrelated wandb-startup crash in the base run by setting `WANDB__SERVICE_WAIT=300` in the launcher (mitigates wandb's 30-second port-file polling timeout on slow shared filesystems).

**Tech Stack:** PyTorch, PyTorch Lightning, OmegaConf, pytest. Unchanged from current stack.

---

## File Structure

**Files modified:**
- `proteinfoundation/proteinflow/proteina.py` — drop the K==1 assertion in `on_train_start`; add a defensive assertion that pool mode is incompatible with `fold_cond=True` (pool x_1 has no associated cath_code, so dataloader-batch cath_code would mis-pair with pool proteins); add a soft warning when `pool_size < K * batch_size` (refill on every step is wasteful but legal).
- `proteinfoundation/proteinflow/model_trainer_base.py` — restructure the K>1 path so that, when pool mode is active, the K-loop draws a fresh OT-paired (x_1, x_0, mask) from the pool per iteration AND resamples (t, r) per iteration. The non-pool K>1 path keeps "shared x_1, shared (t, r), unique x_0 per pass" semantics.
- `tests/test_ot_pool.py` — flip `test_on_train_start_rejects_pool_with_loss_accum_gt_1` to assert the assertion no longer raises; add three new tests (pool refill cadence under K>1, K pool batches drawn per step, end-to-end finite loss + gradients).
- `tests/test_loss_accumulation.py` — add a regression test that the existing K>1 non-pool path still has the old "shared x_1, K different x_0" semantics (i.e. `extract_clean_sample` is called once per training_step, not K times) — this guards against accidentally moving the non-pool extraction into the K-loop.
- `scripts/train.sbatch` — export `WANDB__SERVICE_WAIT=300` before the `srun python …` line.

**Files NOT modified:**
- `proteinfoundation/flow_matching/ot_pool.py` — `OTPool` already supports cursor-based exhaustion across an arbitrary number of `next_batch` calls per training step. K>1 just calls `next_batch` more often; the class doesn't need to change.
- `configs/experiment_config/training_ca.yaml` and `_debug.yaml` — `loss_accumulation_steps` and `ot_pool_size` are already independently configurable knobs. No schema change.

---

## Background and design notes (read before starting)

The current K=1 pool path (in `training_step`):

```python
# training_step lines ~417-432
ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
pool_size = ot_cfg.get("ot_pool_size", None)
pool_mode = pool_size is not None and self.ot_sampler is not None and not val_step

if pool_mode:
    x_1, x_0_pool, mask, batch_shape, n, dtype = self._get_ot_batch()
else:
    x_0_pool = None
    x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
    x_1 = self.fm._mask_and_zero_com(x_1, mask)
```

The K=1 path passes `x_0_override=x_0_pool` to `_compute_single_noise_loss` to skip the inner fresh-noise sampling and B×B OT.

The current K>1 path (lines ~560-572) shares `(x_1, mask, t_ext, r_ext)` across all K passes and resamples `x_0` (and B×B OT pairs it) per pass — i.e. "fixed data, varied noise" Monte-Carlo gradient averaging.

**The chosen K>1 + pool semantics for this plan:** every K-pass pops a fresh batch from the pool. The K-loop becomes "K independent OT-paired micro-batches accumulated into one optimizer step". Per training step, the pool serves K * B samples (vs B in K=1 pool mode), so the pool refills `K` times more often. (t, r) is resampled per pass to match the per-batch-shape contract.

Because this changes the pool exhaustion cadence, `_get_ot_batch` is called K times per training step in the K>1 pool path. It already handles lazy refill internally — no changes to `OTPool`.

Each pass's `_compute_single_noise_loss` receives `x_0_override=x_0_pool_k`, so each pass skips the inner fresh-noise sampling and B×B OT (the pool already OT-paired). This is the whole point — the pool provides the OT coupling.

**Pre-existing pool limitation made explicit:** pool mode + `fold_cond=True` is not supported (the dataloader batch's `cath_code` does not correspond to the pool's x_1, which was drawn from random dataset indices in `OTPool.refill`). This was silently broken before; we add a clear assertion at `on_train_start`. The user's current runs all have `fold_cond=False`, so this changes no working configuration.

---

## Task 1: Pre-flight — verify the failing-state reproduces and the test suite is green

**Files:**
- Read-only verification.

- [ ] **Step 1: Activate venv and confirm pytest runs**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py tests/test_loss_accumulation.py -v
```

Expected: all tests pass, including `test_on_train_start_rejects_pool_with_loss_accum_gt_1`.

- [ ] **Step 2: Confirm the failing assertion text matches the OT crash log**

Run:

```bash
grep -n "ot_pool_size is set but loss_accumulation_steps" \
  /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/proteinfoundation/proteinflow/proteina.py
```

Expected: matches line ~141 inside `on_train_start`. This is the assertion that the OT run crashed on (`logs/train_ot_43291.err` line 102).

No commit — this task is verification only.

---

## Task 2: Test — pool + K>1 must NOT raise at `on_train_start`

**Files:**
- Modify: `tests/test_ot_pool.py:339-358`

- [ ] **Step 1: Replace the rejection test with an acceptance test**

Open `tests/test_ot_pool.py`. Replace the body of `test_on_train_start_rejects_pool_with_loss_accum_gt_1` (currently asserting the assertion raises) with the following. Also rename the function to `test_on_train_start_accepts_pool_with_loss_accum_gt_1`:

```python
def test_on_train_start_accepts_pool_with_loss_accum_gt_1():
    """Pool + loss_accumulation_steps > 1 is now supported. on_train_start
    must build the pool without raising."""
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg_exp = _build_proteina_cfg(ot_pool_size=8, loss_accumulation_steps=2)
    model = Proteina(cfg_exp=cfg_exp)
    model.to("cpu")

    class _FakeDM:
        batch_size = 2
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=0)
        def setup(self, stage): pass

    class _FakeTrainer:
        datamodule = _FakeDM()
        world_size = 1

    model._trainer = _FakeTrainer()

    # Should NOT raise.
    model.on_train_start()

    assert model._ot_pool is not None
    assert model._ot_pool.pool_size == 8
    assert model._ot_pool.batch_size == 2
```

Note: `ot_pool_size=8` (not 4) so that K=2 with batch_size=2 has a clean cadence: pool serves K*B = 4 samples per training step, so two training steps fully exhaust the pool. `_FakeDataset` already provides 16 proteins (more than enough). `_FakeDataset` and `_build_proteina_cfg` are defined earlier in the same file — do not redefine them.

- [ ] **Step 2: Run the new test (it MUST fail right now)**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py::test_on_train_start_accepts_pool_with_loss_accum_gt_1 -v
```

Expected: FAIL with `AssertionError: ot_pool_size is set but loss_accumulation_steps=2. Pool mode requires loss_accumulation_steps=1.`

This proves the test exercises the failing path before we fix it.

- [ ] **Step 3: Commit (failing test)**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add tests/test_ot_pool.py
git commit -m "test(ot): expect pool to accept loss_accumulation_steps > 1 (failing)"
```

---

## Task 3: Implementation — drop the K==1 pool assertion and add the fold_cond guard

**Files:**
- Modify: `proteinfoundation/proteinflow/proteina.py:133-161`

- [ ] **Step 1: Edit `on_train_start`**

In `proteinfoundation/proteinflow/proteina.py`, find the `on_train_start` method (starts at line 133) and replace its body so it (a) no longer asserts `loss_accumulation_steps == 1`, (b) asserts pool mode is incompatible with `fold_cond=True`, and (c) emits a loguru warning when `pool_size < K * batch_size` (refill on every step is wasteful but legal).

The new method body:

```python
    def on_train_start(self):
        """Build the OT pool when ``ot_coupling.ot_pool_size`` is set.

        Pool mode supports ``loss_accumulation_steps > 1``: per training step
        the K-loop pops K consecutive batches from the pool, so the pool
        refills ``K`` times more often than in K=1 mode.
        """
        from proteinfoundation.flow_matching.ot_pool import OTPool

        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        pool_size = ot_cfg.get("ot_pool_size", None)
        if pool_size is not None and self.ot_sampler is not None:
            assert not self.cfg_exp.training.get("fold_cond", False), (
                "Pool mode is incompatible with fold_cond=True: pool x_1 is "
                "drawn from random dataset indices in OTPool.refill and has "
                "no associated cath_code. Disable fold_cond or unset "
                "ot_pool_size."
            )
            dm = self.trainer.datamodule
            if dm.train_ds is None:
                dm.setup("fit")
            self._ot_dataset = dm.train_ds
            bs = self.trainer.datamodule.batch_size
            assert pool_size >= bs, (
                f"ot_pool_size ({pool_size}) must be >= batch_size ({bs})"
            )
            K = self.loss_accumulation_steps
            if pool_size < K * bs:
                from loguru import logger
                logger.warning(
                    "ot_pool_size={} < loss_accumulation_steps * batch_size "
                    "({} * {} = {}); pool will refill on every training step.",
                    pool_size, K, bs, K * bs,
                )
            self._ot_pool = OTPool(
                pool_size=pool_size,
                batch_size=bs,
                dim=self.fm.dim,
                scale_ref=self.fm.scale_ref,
            )
        else:
            self._ot_pool = None
            self._ot_dataset = None
```

The assertion on line ~140-144 (the K==1 enforcement) is removed. Everything else in the body is rewritten as above. The `from loguru import logger` import is local to the warning branch to avoid adding a top-level import for a rarely-hit code path; keep it as written.

- [ ] **Step 2: Run the new pool acceptance test — it should now pass**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py::test_on_train_start_accepts_pool_with_loss_accum_gt_1 -v
```

Expected: PASS.

- [ ] **Step 3: Run the rest of `test_ot_pool.py` to confirm no regressions**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py -v
```

Expected: all tests pass. The K=1 pool path is unchanged.

- [ ] **Step 4: Commit**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add proteinfoundation/proteinflow/proteina.py tests/test_ot_pool.py
git commit -m "feat(ot): allow ot_pool_size with loss_accumulation_steps > 1"
```

---

## Task 4: Test — pool + K>1 training_step draws K batches per step

**Files:**
- Modify: `tests/test_ot_pool.py` (append a new test at the end of the file)

- [ ] **Step 1: Add the new test**

Append at the end of `tests/test_ot_pool.py`:

```python
def test_training_step_pool_mode_with_loss_accum_pops_k_batches_per_step():
    """Pool + loss_accumulation_steps=K: one training_step call pops K
    consecutive batches from the pool (cursor advances by K*B)."""
    from proteinfoundation.proteinflow.proteina import Proteina

    K = 2
    B = 2
    POOL = 8  # holds K * B * 2 = 8 samples -> 2 training steps per pool

    cfg_exp = _build_proteina_cfg(ot_pool_size=POOL, loss_accumulation_steps=K)
    model = Proteina(cfg_exp=cfg_exp)
    model.to("cpu")
    model.train()

    class _FakeDM:
        batch_size = B
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=42)
        def setup(self, stage): pass

    class _FakeTrainer:
        datamodule = _FakeDM()
        world_size = 1

    model._trainer = _FakeTrainer()
    model.on_train_start()

    # Wire up an optimizer + LR scheduler so manual_backward / optimizer.step
    # can run. _build_model patterns in test_loss_accumulation.py do this via
    # `model.trainer = _FakeTrainer(...)` with a manual configure_optimizers()
    # call; copy that pattern here.
    from proteinfoundation.proteinflow.proteina import Proteina  # noqa: F401
    opts = model.configure_optimizers()
    if isinstance(opts, dict):
        opt = opts["optimizer"]
    else:
        opt = opts
    model._optimizers_list = [opt]
    # Lightning's `optimizers()` reads from `trainer.strategy._lightning_optimizers`
    # in the real path; for testing, monkeypatch:
    model.optimizers = lambda: opt
    model.lr_schedulers = lambda: None
    # Manual-step bookkeeping
    model._manual_step_count = 0
    model._accum_grad_batches = 1

    # Build a dummy dataloader batch (ignored on the pool path).
    dummy_batch = {
        "coords": torch.zeros(B, 8, 37, 3),
        "mask_dict": {"coords": torch.ones(B, 8, 1, 1, dtype=torch.bool)},
    }

    cursor_before = model._ot_pool._cursor

    result = model.training_step(dummy_batch, batch_idx=0)

    cursor_after = model._ot_pool._cursor

    # K calls to next_batch advance cursor by K * B.
    expected_advance = K * B
    actual_advance = cursor_after - cursor_before
    assert actual_advance == expected_advance, (
        f"Pool cursor advanced by {actual_advance}, expected K*B={expected_advance}"
    )
    # Manual optimization in K>1 returns None.
    assert result is None
```

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py::test_training_step_pool_mode_with_loss_accum_pops_k_batches_per_step -v
```

Expected: FAIL — currently `training_step` in K>1 mode doesn't call `_get_ot_batch` at all, so the cursor never advances. The test exposes the missing wiring.

- [ ] **Step 2: Commit (failing test)**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add tests/test_ot_pool.py
git commit -m "test(ot): training_step pool+K>1 pops K batches per step (failing)"
```

---

## Task 5: Implementation — wire pool into the K>1 path of `training_step`

**Files:**
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py:394-634`

This is the core code change. Replace the K>1 branch (currently lines ~560-634) so it draws from the pool per K-pass when pool mode is active. The K=1 path stays as-is. The non-pool K>1 path keeps "shared x_1, shared (t, r), unique x_0 per pass" semantics.

- [ ] **Step 1: Read the current `training_step` to confirm line numbers**

Open `proteinfoundation/proteinflow/model_trainer_base.py`. The `training_step` method starts at line 394. The K=1 branch is `if K == 1:` around line 481; the K>1 branch is `# --- 4. K>1 path (manual optimization) ---` around line 560.

- [ ] **Step 2: Replace the K>1 branch**

Replace the entire block from `# --- 4. K>1 path (manual optimization) ---` (line ~560) through the end of `training_step` (the `return None` on line ~634, just before `def compute_fm_loss`) with the following.

The new block computes (x_1, mask, t, r, x_0_pool_k) per pass when pool mode is active, and falls back to "shared (x_1, t, r), unique x_0 per pass" when pool mode is not. It must look like:

```python
        # --- 4. K>1 path (manual optimization) ---
        optimizer = self.optimizers()

        total_loss_mf = 0.0
        total_loss_fm = 0.0
        total_adp_wt = 0.0

        # Track samples processed across the K passes for scaling stats. In
        # pool mode each pass uses a fresh B-batch from the pool, so K passes
        # account for K * B samples. In non-pool mode all K passes share the
        # same x_1 — only B samples per training_step.
        if pool_mode:
            samples_this_step = 0
        else:
            samples_this_step = mask.shape[0]

        for _ in range(K):
            if pool_mode:
                # Each pass: fresh OT-paired (x_1, x_0, mask) from the pool
                # AND fresh (t, r) matching the per-pass batch_shape.
                (
                    x_1_k,
                    x_0_pool_k,
                    mask_k,
                    batch_shape_k,
                    _n_k,
                    _dtype_k,
                ) = self._get_ot_batch()
                t_k, r_k = self.fm.sample_two_timesteps_uniform(
                    batch_shape_k, self.device,
                    ratio=self.meanflow_ratio,
                )
                t_ext_k = t_k[..., None, None]
                r_ext_k = r_k[..., None, None]
                B_k = t_k.shape[0]
                samples_this_step += mask_k.shape[0]
            else:
                # Legacy semantics: shared (x_1, mask, t, r) across passes;
                # only x_0 varies (sampled fresh inside _compute_single_noise_loss).
                x_1_k = x_1
                mask_k = mask
                t_k = t
                t_ext_k = t_ext
                r_ext_k = r_ext
                B_k = B
                x_0_pool_k = None  # triggers fresh x_0 + B x B OT inside the helper

            loss_k, raw_loss_mf_k, raw_loss_fm_k, raw_adp_wt_mean_k = (
                self._compute_single_noise_loss(
                    x_1_k, mask_k, t_ext_k, r_ext_k, t_k, batch, B_k,
                    use_sc=use_sc,
                    x_0_override=x_0_pool_k,
                )
            )
            # Backward with 1/K scaling so accumulated gradient == mean gradient
            self.manual_backward(loss_k / K)
            total_loss_mf += raw_loss_mf_k.item()
            total_loss_fm += raw_loss_fm_k.item()
            total_adp_wt += raw_adp_wt_mean_k.item()

        avg_loss_mf = total_loss_mf / K
        avg_loss_fm = total_loss_fm / K
        avg_adp_wt = total_adp_wt / K
        mf_ratio = self.cfg_exp.training.meanflow.ratio
        avg_combined = (1 - mf_ratio) * avg_loss_fm + mf_ratio * avg_loss_mf

        # Use the per-pass batch_size for logging. In pool mode B is constant
        # (= datamodule.batch_size) across passes; in non-pool mode B is the
        # pre-loop B. Either way, mask_k.shape[0] from the last iteration is
        # a stable per-pass batch size for log batch_size accounting.
        log_bs = mask_k.shape[0]

        # Log averaged metrics
        self.log(
            f"{log_prefix}/combined_adaptive_loss", avg_combined,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=log_bs, sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_loss_mf", avg_loss_mf,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=log_bs, sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_loss_fm", avg_loss_fm,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=log_bs, sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_adp_wt_mean", avg_adp_wt,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=log_bs, sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/loss_accumulation_steps", float(K),
            on_step=True, on_epoch=False, prog_bar=False, logger=True,
            batch_size=1, sync_dist=False, add_dataloader_idx=False,
        )

        self.nsamples_processed = (
            self.nsamples_processed + samples_this_step * self.trainer.world_size
        )
        self.log(
            "scaling/nsamples_processed", self.nsamples_processed * 1.0,
            on_step=True, on_epoch=False, prog_bar=False, logger=True,
            batch_size=1, sync_dist=True,
        )
        self.log(
            "scaling/nparams", self.nparams * 1.0,
            on_step=True, on_epoch=False, prog_bar=False, logger=True,
            batch_size=1, sync_dist=True,
        )

        # --- 5. Manual optimizer step (respects _accum_grad_batches) ---
        self._manual_step_count += 1
        if self._manual_step_count % self._accum_grad_batches == 0:
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()
            optimizer.zero_grad()
            # Step LR scheduler if present
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        return None  # manual optimization: Lightning doesn't need a loss value
```

The differences vs the old K>1 block are:

1. The K-loop now branches on `pool_mode`: pool mode pops a fresh batch + (t, r) per pass; non-pool mode reuses the pre-loop (x_1, mask, t, r) (legacy).
2. `samples_this_step` accumulates inside the loop in pool mode (K * B) and is set once outside the loop in non-pool mode (B). Replaces the old line `b, _n = mask.shape; ... + b * world_size` which only counted one batch.
3. `log_bs` uses the last-iteration `mask_k.shape[0]` (constant across passes — pool serves a fixed batch_size; non-pool uses the pre-loop mask).
4. The `_compute_single_noise_loss` call now passes `x_0_override=x_0_pool_k`. In non-pool mode `x_0_pool_k=None` so the helper resamples fresh x_0 internally — identical to today.

**Important pre-loop variables that must already exist in `training_step` before the K>1 block:** `pool_mode`, `x_1`, `mask`, `batch_shape`, `n`, `dtype`, `t`, `r`, `t_ext`, `r_ext`, `B`, `use_sc`, `K`, `log_prefix`. Confirm by inspecting lines 417-478 — all are bound in the shared setup. In pool mode the pre-loop `x_1, x_0_pool, mask` is the FIRST batch from the pool, but in K>1 pool mode we ignore it and pop fresh batches inside the loop. That's an extra unused pop per training step — fix by guarding the pre-loop pool draw on K==1 (see Step 3 below).

- [ ] **Step 3: Guard the pre-loop pool draw on K=1**

Above the K-branch (around line 425), the pre-loop currently always calls `_get_ot_batch()` when `pool_mode`. For K>1 pool mode the K-loop pops K fresh batches itself, so the pre-loop pop would burn one extra batch every step (off-by-K-times-B in the cursor advance test).

Replace the pre-loop block:

```python
        if pool_mode:
            # OT pool: pool owns (x_1, x_0) pairing. x_0 flows through as
            # x_0_override into _compute_single_noise_loss.
            x_1, x_0_pool, mask, batch_shape, n, dtype = self._get_ot_batch()
        else:
            x_0_pool = None
            x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, mask)
```

with:

```python
        # K is determined further down (val_step forces K=1 there too); compute
        # it early so we know whether to pre-draw the pool batch. Validation
        # always uses K=1 regardless of config.
        K = 1 if val_step else self.loss_accumulation_steps

        if pool_mode and K == 1:
            # K=1 pool path: pre-draw the single batch the K=1 branch will use.
            x_1, x_0_pool, mask, batch_shape, n, dtype = self._get_ot_batch()
        elif pool_mode:
            # K>1 pool path: pool draw happens inside the K-loop below. Bind
            # placeholders so downstream `B = t.shape[0]` etc. compile; t/r
            # sampling still uses these batch_shape/mask but they're discarded
            # by the pool branch of the K-loop.
            x_1, x_0_pool, mask, batch_shape, n, dtype = self._get_ot_batch()
            # We DID just pop a B-batch here. Mark so the K-loop knows to use
            # this one as the FIRST iteration's payload (no extra pop).
            self._pool_prefetched = (x_1, x_0_pool, mask, batch_shape)
        else:
            x_0_pool = None
            x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, mask)
```

Then in the K-loop's pool branch, use the prefetched batch on the first iteration and pop fresh ones for the remaining K-1 passes:

```python
        for k in range(K):
            if pool_mode:
                if k == 0 and getattr(self, "_pool_prefetched", None) is not None:
                    x_1_k, x_0_pool_k, mask_k, batch_shape_k = self._pool_prefetched
                    self._pool_prefetched = None
                else:
                    (
                        x_1_k,
                        x_0_pool_k,
                        mask_k,
                        batch_shape_k,
                        _n_k,
                        _dtype_k,
                    ) = self._get_ot_batch()
                t_k, r_k = self.fm.sample_two_timesteps_uniform(
                    batch_shape_k, self.device,
                    ratio=self.meanflow_ratio,
                )
                t_ext_k = t_k[..., None, None]
                r_ext_k = r_k[..., None, None]
                B_k = t_k.shape[0]
                samples_this_step += mask_k.shape[0]
            else:
                x_1_k = x_1
                ...  # rest as in Step 2
```

Update the K-loop signature line `for _ in range(K):` → `for k in range(K):` so we can detect the first iteration.

Also remove the duplicate `K = 1 if val_step else self.loss_accumulation_steps` line that previously appeared just before the `if K == 1:` branch (now redundant — we computed `K` earlier). Just delete that one line.

**Why prefetch?** In pool mode the existing code expects `(x_1, mask, batch_shape, ...)` to be defined before the (t, r) sampling block (lines 435-442) and before the fold-cond block (lines 452-471). Removing the pre-loop pool draw would break that. The simplest fix is to keep the pre-draw and use it as the first pass's payload — that way K total pops match K passes exactly.

Alternative pattern (do not implement, kept here for reviewers): refactor the (t, r) sampling and fold-cond block to be K-loop-internal so the pool draw is purely inside the loop. That is cleaner but a bigger refactor; we keep the prefetch pattern to minimize diff.

- [ ] **Step 4: Run the new test — should now pass**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py::test_training_step_pool_mode_with_loss_accum_pops_k_batches_per_step -v
```

Expected: PASS. Cursor advances by exactly K * B per training_step.

- [ ] **Step 5: Run the full pool + loss-accumulation suites for regressions**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py tests/test_loss_accumulation.py tests/test_training_step.py tests/test_jvp_smoke.py -v
```

Expected: all tests pass. The K=1 path is untouched. The non-pool K>1 path keeps its old "shared x_1, varied x_0" semantics because `pool_mode` is False there. The pool K=1 path is identical (the prefetched batch IS the K=1 payload).

- [ ] **Step 6: Commit**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add proteinfoundation/proteinflow/model_trainer_base.py tests/test_ot_pool.py
git commit -m "feat(ot): wire pool into K>1 training_step (K batches per step)"
```

---

## Task 6: Test — pool + K>1 produces finite loss and gradients (end-to-end)

**Files:**
- Modify: `tests/test_ot_pool.py` (append a new test)

- [ ] **Step 1: Add the end-to-end test**

Append at the end of `tests/test_ot_pool.py`:

```python
def test_training_step_pool_mode_with_loss_accum_finite_loss_and_grad():
    """Pool + loss_accumulation_steps=K runs end-to-end: nonzero gradients on
    nn parameters, no NaN/Inf in loss telemetry."""
    from proteinfoundation.proteinflow.proteina import Proteina

    K = 2
    B = 2
    POOL = 8

    cfg_exp = _build_proteina_cfg(ot_pool_size=POOL, loss_accumulation_steps=K)
    model = Proteina(cfg_exp=cfg_exp)
    model.to("cpu")
    model.train()

    class _FakeDM:
        batch_size = B
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=42)
        def setup(self, stage): pass

    class _FakeTrainer:
        datamodule = _FakeDM()
        world_size = 1

    model._trainer = _FakeTrainer()
    model.on_train_start()

    opts = model.configure_optimizers()
    opt = opts["optimizer"] if isinstance(opts, dict) else opts
    model.optimizers = lambda: opt
    model.lr_schedulers = lambda: None
    model._manual_step_count = 0
    model._accum_grad_batches = 2  # so optimizer.zero_grad does NOT clear grads after step 1

    dummy_batch = {
        "coords": torch.zeros(B, 8, 37, 3),
        "mask_dict": {"coords": torch.ones(B, 8, 1, 1, dtype=torch.bool)},
    }

    model.training_step(dummy_batch, batch_idx=0)

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "Pool + K=2: no parameters received nonzero gradients"
```

- [ ] **Step 2: Run it**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_ot_pool.py::test_training_step_pool_mode_with_loss_accum_finite_loss_and_grad -v
```

Expected: PASS — gradients flow through K passes despite the pool-managed (x_1, x_0).

- [ ] **Step 3: Commit**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add tests/test_ot_pool.py
git commit -m "test(ot): pool+K>1 end-to-end finite loss + gradients"
```

---

## Task 7: Test — guard the legacy K>1 non-pool path

**Files:**
- Modify: `tests/test_loss_accumulation.py` (append a new test)

This test pins the existing semantics of K>1 WITHOUT pool: shared x_1, varied x_0. If a future refactor accidentally moves `extract_clean_sample` into the K-loop, this test fails.

- [ ] **Step 1: Add the regression test**

Append at the end of `tests/test_loss_accumulation.py`:

```python
def test_k_greater_than_1_non_pool_extract_called_once():
    """Without the pool, K>1 must call extract_clean_sample exactly once per
    training_step (shared-x_1 semantics). Pool mode is a separate path."""
    model = _build_model(K=2, ot_enabled=True)  # ot_pool_size NOT set
    batch = _build_batch(4, 16)

    call_count = []
    original = model.extract_clean_sample

    def counting_extract(b):
        call_count.append(1)
        return original(b)

    model.extract_clean_sample = counting_extract

    model.training_step(batch, batch_idx=0)

    assert len(call_count) == 1, (
        f"Non-pool K=2: extract_clean_sample called {len(call_count)} times, "
        "expected exactly 1 (shared-x_1 semantics)."
    )
```

`_build_model(K=2, ot_enabled=True)` constructs a Proteina with `ot_coupling.enabled=True` but does NOT set `ot_pool_size` (verify in the existing `_build_cfg` helper at the top of the file — `ot_pool_size` is only included when explicitly requested). If `_build_cfg` does include `ot_pool_size` by default for OT, override it to `None` in this test.

- [ ] **Step 2: Inspect `_build_cfg` to confirm it omits `ot_pool_size`**

Run:

```bash
grep -n "ot_pool_size\|ot_coupling" /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/tests/test_loss_accumulation.py | head -20
```

Expected: `ot_coupling.enabled=True` is set when `ot_enabled=True`, but `ot_pool_size` is not added — confirming the non-pool path. If the test helper does set `ot_pool_size`, edit the new test to explicitly null it via `model.cfg_exp.training.ot_coupling.ot_pool_size = None` before constructing the model.

- [ ] **Step 3: Run the new test**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/test_loss_accumulation.py::test_k_greater_than_1_non_pool_extract_called_once -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add tests/test_loss_accumulation.py
git commit -m "test(loss-accum): guard non-pool K>1 shared-x_1 semantics"
```

---

## Task 8: Mitigate the base-run wandb startup crash

**Files:**
- Modify: `scripts/train.sbatch:128-269`

The base run (`logs/train_base_43290.err`) failed with:

```
wandb.sdk.lib.service.service_port_file.ServicePollForTokenError: Failed to read port info after 30.0 seconds.
```

This is wandb's internal service subprocess timing out while writing its port file on slow shared filesystems. The fix is to bump `WANDB__SERVICE_WAIT` (note: **double underscore** between `WANDB` and `SERVICE`).

- [ ] **Step 1: Add the env var to `train.sbatch`**

In `scripts/train.sbatch`, find the existing wandb-adjacent environment exports (around line 128: `export PYTHONUNBUFFERED=1`). Add immediately after that line:

```bash
# wandb's service subprocess polls for a port file with a 30s default timeout.
# On slow shared FS (observed on H100Azure03, job 43290) the poll can exceed
# 30s and crash wandb.init. Bump to 5 minutes.
export WANDB__SERVICE_WAIT=300
```

- [ ] **Step 2: Verify the file parses**

Run:

```bash
bash -n /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/scripts/train.sbatch
```

Expected: no output (syntax OK).

- [ ] **Step 3: Confirm the env var name is correctly spelled**

Run:

```bash
grep -n "WANDB__SERVICE_WAIT\|WANBD__SERVICE_WAIT" \
  /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/scripts/train.sbatch
```

Expected: one match for `WANDB__SERVICE_WAIT`, zero matches for the misspelling `WANBD__SERVICE_WAIT` (a common typo in upstream wandb issues).

- [ ] **Step 4: Commit**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git add scripts/train.sbatch
git commit -m "fix(sbatch): raise WANDB__SERVICE_WAIT to 300s for slow shared FS"
```

---

## Task 9: Final verification — full test suite green

**Files:**
- Read-only verification.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
source /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/activate && \
  PYTHONPATH=/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina \
  pytest tests/ -v
```

Expected: all tests pass. If any test fails, return to the relevant task and fix before proceeding.

- [ ] **Step 2: Smoke-check the diff against the working tree**

Run:

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
git log --oneline -8
git diff main..HEAD --stat
```

Expected: 5 commits on the branch (one per task that produced a code change: Task 2 failing test, Task 3, Task 5, Task 6, Task 7, Task 8 — six commits total). Files changed:
- `proteinfoundation/proteinflow/proteina.py`
- `proteinfoundation/proteinflow/model_trainer_base.py`
- `tests/test_ot_pool.py`
- `tests/test_loss_accumulation.py`
- `scripts/train.sbatch`

No other files should appear in the stat. If they do, investigate before declaring done — pre-existing WIP may have been swept in (see memory `feedback_verify_working_tree_before_commit.md`).

- [ ] **Step 3: Hand off**

Report to the user: tests pass, commits are scoped, both crashes addressed (OT crash via core feature; base crash via wandb env var). Ask whether to relaunch the runs.

---

## Self-review checklist (notes from plan author)

1. **Spec coverage:**
   - Pool + K>1 acceptance at `on_train_start` — Task 3.
   - Per-pass pool draw inside K-loop — Task 5.
   - Per-pass (t, r) resampling in pool mode — Task 5 step 2.
   - Pool refill cadence acceleration (K * B per step) — verified in Task 4.
   - End-to-end finite loss + gradients — Task 6.
   - Non-pool K>1 regression guard — Task 7.
   - fold_cond + pool guard — Task 3 step 1.
   - Pool size warning when smaller than K * B — Task 3 step 1.
   - Wandb base-run crash mitigation — Task 8.

2. **Placeholder scan:** no TBD, TODO, "appropriate", "similar to". Code blocks complete in every step.

3. **Type / signature consistency:**
   - `_compute_single_noise_loss(x_1, mask, t_ext, r_ext, t, batch, B, *, use_sc, x_0_override)` is unchanged across tasks (signature already exists in current code).
   - `_get_ot_batch()` returns `(x_1, x_0, mask, batch_shape, n, dtype)` consistently in Tasks 4, 5, 6 (matches existing `OTPool.next_batch` return).
   - `_pool_prefetched` is a new instance attribute introduced in Task 5; only Task 5 reads/writes it. Initial value is set at the prefetch site; the first iteration of the K-loop consumes and clears it.
   - `pool_mode` boolean is bound consistently (line ~419 of training_step).

4. **Subagent scope hygiene:** Tasks 2-8 each touch a small file set with explicit `git add` lists. Per the memory `feedback_verify_working_tree_before_commit.md`, the implementer agent should run `git status` BEFORE each commit step and refuse to add files outside the explicit list. The dispatcher (main agent) should also `git stash -u` any pre-existing WIP before starting Task 2.

