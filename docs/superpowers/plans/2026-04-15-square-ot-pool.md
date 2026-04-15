# Square OT Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the biased rectangular OT noise-oversampling with a proper square OT pool that samples extra proteins from the dataset, computes a full OT assignment, and randomly selects B pairs for GPU training.

**Architecture:** Add a `_build_ot_pool` method to `Proteina` that constructs the OT pool on CPU. The method loads extra proteins from the training dataset, pads all samples to uniform length, samples noise, computes a square cost matrix, runs Hungarian assignment, randomly selects B pairs, and moves them to GPU. The `training_step` in `ModelTrainerBase` is modified to call this method when `noise_samples` is configured.

**Tech Stack:** PyTorch, scipy (Hungarian algorithm), existing `dense_padded_from_data_list` collation utility, PyTorch Lightning hooks.

---

### Task 1: Add Dataset Reference Hook in Proteina

**Files:**
- Modify: `proteinfoundation/proteinflow/proteina.py:59-78`

- [ ] **Step 1: Add `on_train_start` hook to store dataset reference**

Add after the `__init__` method (after line 124), before `align_wrapper`:

```python
def on_train_start(self):
    """Store reference to training dataset for OT pool sampling."""
    ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
    noise_samples = ot_cfg.get("noise_samples", None)
    if noise_samples is not None and self.ot_sampler is not None:
        dm = self.trainer.datamodule
        if dm.train_ds is None:
            dm.setup("fit")
        self._ot_dataset = dm.train_ds
        assert noise_samples >= self.trainer.datamodule.batch_size, (
            f"noise_samples ({noise_samples}) must be >= batch_size "
            f"({self.trainer.datamodule.batch_size})"
        )
```

- [ ] **Step 2: Verify hook is recognized by Lightning**

Run: `cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina && python -c "from proteinfoundation.proteinflow.proteina import Proteina; print('import OK')"`
Expected: `import OK` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add proteinfoundation/proteinflow/proteina.py
git commit -m "Add on_train_start hook to store dataset reference for OT pool"
```

---

### Task 2: Add `_build_ot_pool` Method in Proteina

**Files:**
- Modify: `proteinfoundation/proteinflow/proteina.py`

This is the core new method. It builds a square OT pool of `noise_samples` data/noise pairs on CPU, runs Hungarian assignment, randomly selects B pairs, and returns them ready for GPU training.

- [ ] **Step 1: Add the `_build_ot_pool` method**

Add after the `on_train_start` method, before `align_wrapper`:

```python
def _build_ot_pool(self, batch):
    """Build a square OT pool on CPU and return B selected pairs on GPU.

    Samples extra proteins from the training dataset to build a pool of
    `noise_samples` data/noise pairs, computes square OT assignment via
    Hungarian algorithm, then randomly selects B pairs for training.

    Args:
        batch: Dataloader batch (used to extract the initial B proteins).

    Returns:
        Tuple (x_1, x_0, mask, batch_shape, n, dtype) with B selected
        pairs on self.device (GPU).
    """
    import scipy.optimize
    from proteinfoundation.utils.dense_padding_data_loader import (
        dense_padded_from_data_list,
    )

    ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
    K = ot_cfg["noise_samples"]  # total pool size

    # --- 1. Extract clean data from the dataloader batch (on CPU) ---
    x_1_batch, mask_batch, batch_shape_orig, n_batch, dtype = (
        self.extract_clean_sample(batch)
    )
    x_1_batch = self.fm._mask_and_zero_com(x_1_batch, mask_batch)
    B = batch_shape_orig[0]

    # Ensure tensors are on CPU for OT computation
    x_1_batch = x_1_batch.detach().cpu()
    mask_batch = mask_batch.cpu()

    # --- 2. Sample extra proteins to fill the pool ---
    n_extra = K - B

    if n_extra > 0:
        extra_indices = torch.randint(0, len(self._ot_dataset), (n_extra,))
        extra_data_list = [self._ot_dataset[int(idx)] for idx in extra_indices]
        extra_batch = dense_padded_from_data_list(extra_data_list)

        # Process extra proteins identically to the main batch
        x_1_extra, mask_extra, _, _, _ = self.extract_clean_sample(extra_batch)
        x_1_extra = self.fm._mask_and_zero_com(x_1_extra, mask_extra)
        x_1_extra = x_1_extra.detach().cpu()
        mask_extra = mask_extra.cpu()

        # --- 3. Pad both to the same sequence length ---
        N_batch = x_1_batch.shape[1]
        N_extra = x_1_extra.shape[1]
        N_pool = max(N_batch, N_extra)

        if N_batch < N_pool:
            pad_size = N_pool - N_batch
            x_1_batch = torch.cat(
                [x_1_batch, torch.zeros(B, pad_size, 3, dtype=dtype)], dim=1
            )
            mask_batch = torch.cat(
                [mask_batch, torch.zeros(B, pad_size, dtype=torch.bool)], dim=1
            )

        if N_extra < N_pool:
            pad_size = N_pool - N_extra
            x_1_extra = torch.cat(
                [x_1_extra, torch.zeros(n_extra, pad_size, 3, dtype=dtype)],
                dim=1,
            )
            mask_extra = torch.cat(
                [mask_extra, torch.zeros(n_extra, pad_size, dtype=torch.bool)],
                dim=1,
            )

        x_1_pool = torch.cat([x_1_batch, x_1_extra], dim=0)  # [K, N_pool, 3]
        mask_pool = torch.cat([mask_batch, mask_extra], dim=0)  # [K, N_pool]
    else:
        # K == B: no extra samples needed
        x_1_pool = x_1_batch
        mask_pool = mask_batch
        N_pool = x_1_batch.shape[1]

    # --- 4. Sample noise on CPU ---
    x_0_pool = (
        torch.randn(K, N_pool, self.fm.dim, dtype=dtype) * self.fm.scale_ref
    )
    x_0_pool = self.fm._mask_and_zero_com(x_0_pool, mask_pool)

    # --- 5. Build cost matrix [K, K] ---
    mask_3d = mask_pool[..., None]  # [K, N_pool, 1] bool
    x_1_flat = (x_1_pool * mask_3d).reshape(K, -1)  # [K, N_pool*3]
    x_0_flat = (x_0_pool * mask_3d).reshape(K, -1)  # [K, N_pool*3]
    M = torch.cdist(x_1_flat, x_0_flat) ** 2  # [K, K]

    # --- 6. Hungarian algorithm for optimal assignment ---
    _, sigma = scipy.optimize.linear_sum_assignment(M.numpy())

    # --- 7. Randomly select B pairs from the OT plan ---
    sel = torch.randperm(K)[:B].numpy()
    x_1_sel = x_1_pool[sel]  # [B, N_pool, 3]
    x_0_sel = x_0_pool[sigma[sel]]  # [B, N_pool, 3]
    mask_sel = mask_pool[sel]  # [B, N_pool]

    # --- 8. Trim to max actual length of selected proteins ---
    max_len = int(mask_sel.sum(dim=1).max().item())
    if max_len == 0:
        max_len = N_pool  # safety fallback
    x_1_sel = x_1_sel[:, :max_len, :]
    x_0_sel = x_0_sel[:, :max_len, :]
    mask_sel = mask_sel[:, :max_len]

    # Re-apply mask and zero COM after trimming
    x_1_sel = self.fm._mask_and_zero_com(x_1_sel, mask_sel)
    x_0_sel = self.fm._mask_and_zero_com(x_0_sel, mask_sel)

    # --- 9. Move to GPU ---
    x_1_out = x_1_sel.to(device=self.device)
    x_0_out = x_0_sel.to(device=self.device)
    mask_out = mask_sel.to(device=self.device)

    # Clean up large CPU tensors
    del x_1_pool, x_0_pool, mask_pool, M, x_1_flat, x_0_flat

    return x_1_out, x_0_out, mask_out, torch.Size([B]), max_len, dtype
```

- [ ] **Step 2: Verify import succeeds**

Run: `cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina && python -c "from proteinfoundation.proteinflow.proteina import Proteina; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add proteinfoundation/proteinflow/proteina.py
git commit -m "Add _build_ot_pool method for square OT coupling"
```

---

### Task 3: Modify training_step to Use OT Pool

**Files:**
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py:274-327`

Replace the existing data extraction + OT coupling block with a branching structure that uses `_build_ot_pool` when `noise_samples` is configured.

- [ ] **Step 1: Replace the OT coupling block in training_step**

Replace lines 274-327 (from `# Extract inputs from batch` through the end of the `else` block for OT) with:

```python
        # Extract inputs and handle OT coupling
        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        noise_samples = ot_cfg.get("noise_samples", None)

        if noise_samples is not None and self.ot_sampler is not None:
            # Square OT pool: build large pool on CPU, select B pairs
            x_1, x_0, mask, batch_shape, n, dtype = self._build_ot_pool(batch)
        else:
            # Standard path: extract from batch, sample noise
            x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, mask)
            x_0 = self.fm.sample_reference(
                n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
            )
            # Square-batch OT coupling (when enabled without noise_samples)
            if self.ot_sampler is not None:
                ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(
                    x_1, x_0, mask
                )
                x_0 = x_0[ot_noise_idx]

        # Sample (t, r) -- PAPER CONVENTION: t is noise-side, r is data-side, t >= r
        t, r = self.fm.sample_two_timesteps(
            batch_shape, self.device,
            ratio=self.meanflow_ratio,
            P_mean_t=self.meanflow_P_mean_t,
            P_std_t=self.meanflow_P_std_t,
            P_mean_r=self.meanflow_P_mean_r,
            P_std_r=self.meanflow_P_std_r,
        )
```

This replaces everything from line 274 (`# Extract inputs from batch`) through line 327 (end of the noise/OT block). The timestep sampling block (lines 280-288) is moved after the OT decision since it only depends on `batch_shape`, not on data content.

The lines starting at 328 (`# --- PAPER CONVENTION: z_t = ...`) remain unchanged.

- [ ] **Step 2: Remove the now-unused `scipy.optimize` import if no longer needed in this file**

Check if `scipy.optimize` is still needed. After this change, the rectangular OT code that used `scipy.optimize.linear_sum_assignment` in `training_step` is removed. The square-batch OT path uses `self.ot_sampler.sample_plan_with_scipy` which has its own import. So remove line 20-21:

```python
import scipy.optimize
```

Only remove this if no other code in `model_trainer_base.py` uses `scipy.optimize`. Search the file first.

- [ ] **Step 3: Verify import succeeds**

Run: `cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina && python -c "from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase; print('import OK')"`
Expected: `import OK`

- [ ] **Step 4: Commit**

```bash
git add proteinfoundation/proteinflow/model_trainer_base.py
git commit -m "Replace rectangular OT with square OT pool in training_step"
```

---

### Task 4: Update Debug Training Script

**Files:**
- Modify: `scripts/train_debug.sbatch:83`

- [ ] **Step 1: Enable OT coupling in the sbatch script**

Change line 83 from:
```bash
OT_COUPLING_ENABLED=False
```
to:
```bash
OT_COUPLING_ENABLED=True
```

The `NOISE_SAMPLES=128` on line 87 is already correct for the new semantics (total OT pool size).

- [ ] **Step 2: Commit**

```bash
git add scripts/train_debug.sbatch
git commit -m "Enable OT coupling in debug training script"
```

---

### Task 5: End-to-End Verification

- [ ] **Step 1: Run debug training locally to verify the pipeline works**

Run a quick smoke test (a few steps, not full training):

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export DATA_PATH="$(pwd)/data"

python proteinfoundation/train.py \
    --config_name training_ca_debug \
    --single \
    --show_prog_bar \
    --exp_overrides \
        run_name_=ot-pool-test \
        opt.max_epochs=3 \
        opt.accumulate_grad_batches=1 \
        training.ot_coupling.enabled=True \
        training.ot_coupling.noise_samples=128 \
        log.log_wandb=False \
        log.checkpoint=False \
    --data_overrides \
        datamodule.repeat=2 \
        datamodule.batch_size=2
```

Expected: Training runs for 3 epochs without errors. Loss values are logged. No OOM errors (OT pool runs on CPU).

- [ ] **Step 2: Verify OT pool shapes by adding temporary debug logging**

If step 1 fails or you want to verify shapes, add a temporary print inside `_build_ot_pool` before the return:

```python
print(f"OT pool: K={K}, B={B}, N_pool={N_pool}, max_len={max_len}")
print(f"  x_1_out: {x_1_out.shape}, x_0_out: {x_0_out.shape}, mask_out: {mask_out.shape}")
```

Expected output for debug config (B=2, K=128, 1ubq with 76 residues):
```
OT pool: K=128, B=2, N_pool=76, max_len=76
  x_1_out: torch.Size([2, 76, 3]), x_0_out: torch.Size([2, 76, 3]), mask_out: torch.Size([2, 76])
```

- [ ] **Step 3: Remove temporary debug logging and commit**

Remove any temporary print statements added in step 2.

```bash
git add -A
git commit -m "Verify square OT pool works with debug pipeline"
```
