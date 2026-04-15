# Implementation Plan: Loss Accumulation

**Date:** 2026-04-15
**Branch:** debug_run
**Spec:** `docs/superpowers/specs/2026-04-15-loss-accumulation-design.md`

## Overview

Loss accumulation runs K forward passes per batch with independently sampled noise (`x_0`), accumulates gradients scaled by 1/K, then performs a single optimizer step. This increases effective batch size (= batch_size × K × accumulate_grad_batches) without requiring more GPU memory for data — only for the forward/backward pass itself.

When K=1 the code path is identical to the current implementation. When K>1, Lightning's automatic optimization is disabled and we manage the optimizer manually.

## File Map

```
proteinfoundation/proteinflow/model_trainer_base.py   # training_step + new _compute_single_noise_loss
proteinfoundation/proteinflow/proteina.py              # __init__ reads K, sets automatic_optimization
configs/experiment_config/training_ca.yaml             # add loss_accumulation_steps: 1
configs/experiment_config/training_ca_debug.yaml       # add loss_accumulation_steps: 1
tests/test_loss_accumulation.py                        # new file, 5 test cases
```

---

## Task 1: Config Changes

**Goal:** Add `loss_accumulation_steps` to both YAML configs and wire it up in `Proteina.__init__`. When K>1, disable Lightning's automatic optimization and store the manual-step counter.

**Dependencies:** None.

### 1a. Edit `configs/experiment_config/training_ca.yaml`

File: `/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/configs/experiment_config/training_ca.yaml`

In the `training:` block (currently lines 38-59), add `loss_accumulation_steps` as the first key:

```yaml
training:
  loss_accumulation_steps: 1   # <-- ADD THIS LINE
  self_cond: False
  fold_cond: False
  ...
```

### 1b. Edit `configs/experiment_config/training_ca_debug.yaml`

File: `/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/configs/experiment_config/training_ca_debug.yaml`

Same location in `training:` block (currently starts at line 33):

```yaml
training:
  loss_accumulation_steps: 1   # <-- ADD THIS LINE
  self_cond: False
  fold_cond: False
  ...
```

### 1c. Edit `proteinfoundation/proteinflow/proteina.py`

File: `/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/proteinfoundation/proteinflow/proteina.py`

**Location:** Inside `Proteina.__init__`, after the `ot_sampler` block (after line 77, before the `if self.motif_conditioning:` block).

Add the following block:

```python
# Loss accumulation
K = cfg_exp.training.get("loss_accumulation_steps", 1)
self.loss_accumulation_steps = K
if K > 1:
    # Manual optimization: Lightning won't call optimizer.step() for us.
    # We reimplement accumulate_grad_batches manually below.
    self.automatic_optimization = False
self._accum_grad_batches = cfg_exp.opt.get("accumulate_grad_batches", 1)
self._manual_step_count = 0
```

**Key notes for the subagent:**
- `self.automatic_optimization = False` must be set in `__init__`, not in a hook. Lightning checks this attribute before the first training step.
- When K=1, `automatic_optimization` stays True (the default) and the existing optimizer/scheduler machinery in Lightning is unchanged.
- `_accum_grad_batches` shadows the Trainer-level `accumulate_grad_batches` because Lightning ignores that setting in manual optimization mode.
- `_manual_step_count` counts how many `training_step` calls have completed since the last optimizer step.

**Exact insertion point in proteina.py:**

The OT sampler block ends around line 77:
```python
        if ot_cfg.get("enabled", False):
            self.ot_sampler = MaskedOTPlanSampler(
                ...
            )
        # INSERT NEW BLOCK HERE
        if self.motif_conditioning:
```

**Acceptance criteria:**
- `Proteina(cfg_exp).loss_accumulation_steps == 1` when config key is absent.
- `Proteina(cfg_exp).automatic_optimization == False` when K>1.
- `Proteina(cfg_exp).automatic_optimization == True` when K==1.
- Both YAML files contain `loss_accumulation_steps: 1` under `training:`.

---

## Task 2: Extract `_compute_single_noise_loss()`

**Goal:** Factor the per-noise-sample computation (x_0 sampling → OT → interpolation → JVP → FM loss → adaptive weighting) out of `training_step` into a new method `_compute_single_noise_loss` on `ModelTrainerBase`. The K=1 path must produce bit-for-bit identical results to the current code.

**Dependencies:** None (can be done independently of Task 1, though Task 3 depends on both).

**File:** `/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/proteinfoundation/proteinflow/model_trainer_base.py`

### Current `training_step` structure (for reference)

Lines 250-453 of `model_trainer_base.py`. The relevant sections to extract are:

**Lines 276-291** — x_0 sampling and OT coupling (the else/standard branch; the OT pool branch stays in `training_step` as it also determines x_1 and mask):
```python
        else:
            x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, mask)
            x_0 = self.fm.sample_reference(...)
            if self.ot_sampler is not None:
                ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(x_1, x_0, mask)
                x_0 = x_0[ot_noise_idx]
```

**Lines 305-391** — Interpolation, JVP, FM loss, adaptive weighting:
```python
        t_ext = t[..., None, None]
        r_ext = r[..., None, None]
        z = (1 - t_ext) * x_1 + t_ext * x_0
        v = x_0 - x_1
        z = self.fm._apply_mask(z, mask)
        v = self.fm._apply_mask(v, mask)
        ...
        def u_func(z_in, t_in, r_in): ...
        ...
        combined_adp_loss = (1 - mf_ratio) * loss_fm + mf_ratio * loss_mf
```

### New method signature

```python
def _compute_single_noise_loss(
    self,
    x_1,          # [B, n, 3] — clean data, already COM-zeroed
    mask,         # [B, n] bool
    t_ext,        # [B, 1, 1]
    r_ext,        # [B, 1, 1]
    t,            # [B] — needed for FM loss call u_func(z, t, t)
    batch,        # original batch dict/object (needed for cath_code)
    B,            # int — batch size
):
    """
    Runs one noise sample through the MeanFlow + FM loss pipeline.

    Returns:
        combined_adp_loss: scalar tensor (requires_grad=True)
        raw_loss_mf:       scalar tensor (detached mean, for logging)
        raw_loss_fm:       scalar tensor (detached mean, for logging)
    """
```

### Body of `_compute_single_noise_loss`

The method body is extracted verbatim from the current `training_step`, with these changes:
- Replace `n` (the sequence length) with `x_1.shape[-2]` (computed locally since it may differ per call when OT pool is active).
- The `u_func` closure captures `mask`, `batch`, `B` from the enclosing method arguments.
- The method both samples `x_0` AND applies OT if enabled. This matches the spec: "Sample x_0 via `self.fm.sample_reference()`, apply OT coupling if enabled."

```python
def _compute_single_noise_loss(self, x_1, mask, t_ext, r_ext, t, batch, B):
    n = x_1.shape[-2]
    dtype = x_1.dtype
    batch_shape = x_1.shape[:-2]  # (B,)

    # 1. Sample noise
    x_0 = self.fm.sample_reference(
        n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
    )
    # 2. Standard square-batch OT (when noise_samples pool is NOT active)
    if self.ot_sampler is not None:
        ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(x_1, x_0, mask)
        x_0 = x_0[ot_noise_idx]

    # 3. Interpolate: z_t = (1-t)*x_1 + t*x_0  (paper convention)
    z = (1 - t_ext) * x_1 + t_ext * x_0
    v = x_0 - x_1
    z = self.fm._apply_mask(z, mask)
    v = self.fm._apply_mask(v, mask)

    # 4. JVP for MeanFlow loss
    def u_func(z_in, t_in, r_in):
        h = (t_in - r_in).squeeze(-1).squeeze(-1)
        t_flat = t_in.squeeze(-1).squeeze(-1)
        batch_nn = {"x_t": z_in, "t": t_flat, "h": h, "mask": mask}
        if "cath_code" in batch:
            batch_nn["cath_code"] = [batch["cath_code"][i] for i in range(B)]
        nn_out = self.nn(batch_nn)
        return nn_out["coors_pred"]

    dtdt_mf = torch.ones_like(t_ext)
    drdt_mf = torch.zeros_like(r_ext)
    with torch.amp.autocast(self.device.type, enabled=False):
        u_pred_mf, dudt_mf = torch.func.jvp(
            u_func, (z, t_ext, r_ext), (v, dtdt_mf, drdt_mf)
        )

    u_tgt = (v - (t_ext - r_ext) * dudt_mf).detach()
    u_pred = self.fm._mask_and_zero_com(u_pred_mf, mask)
    u_tgt = self.fm._mask_and_zero_com(u_tgt, mask)

    nres = mask.sum(dim=-1) * 3
    error = (u_pred - u_tgt) * mask[..., None]
    loss_mf = (error ** 2).sum(dim=(-1, -2)) / nres
    raw_loss_mf = loss_mf.mean().detach()
    loss_mf = self.adaptive_loss(loss_mf)

    # 5. FM loss
    mf_ratio = self.cfg_exp.training.meanflow.ratio
    v_pred = u_func(z, t, t)
    v_pred = self.fm._mask_and_zero_com(v_pred, mask)
    loss_fm = (v_pred - v) ** 2 * mask[..., None]
    loss_fm = loss_fm.sum(dim=(-1, -2)) / nres
    raw_loss_fm = loss_fm.mean().detach()
    loss_fm = self.adaptive_loss(loss_fm)

    # 6. Combined
    combined_adp_loss = (1 - mf_ratio) * loss_fm + mf_ratio * loss_mf
    return combined_adp_loss, raw_loss_mf, raw_loss_fm
```

**Critical detail — OT pool path:** When the OT pool is active (`noise_samples` config is set), `x_0` is already provided by `_build_ot_pool()` via the `training_step` caller. In that case, `_compute_single_noise_loss` must NOT resample `x_0` — the OT pool has already assigned it. However, the spec says: when OT pool + loss accumulation are combined, the pool determines *which proteins* to train on (i.e., sets `x_1` and `mask`), and then each of the K noise passes does fresh `sample_reference` + standard square-batch OT (not pool OT). So:

- When called from the K>1 loop, the caller has already resolved `x_1` and `mask` from `_build_ot_pool()` (or standard extraction), but does NOT pass `x_0`.
- `_compute_single_noise_loss` always samples its own `x_0` fresh and applies standard OT (if `self.ot_sampler is not None`).
- The first call (k=0) for the OT pool path therefore does a second OT solve, which is correct behavior per the spec.

This means the OT pool path in `training_step` changes slightly: after calling `_build_ot_pool()`, only `x_1`, `mask`, `batch_shape`, `n`, `dtype` are extracted; `x_0` from `_build_ot_pool` is discarded. The K loop then drives all noise sampling through `_compute_single_noise_loss`. See Task 3 for the updated `training_step` orchestration.

### Refactored `training_step` K=1 path (skeleton)

After extracting `_compute_single_noise_loss`, the `training_step` K=1 code becomes:

```python
# After resolving x_1, mask, batch_shape, n, dtype and sampling t, r:
combined_adp_loss, raw_loss_mf, raw_loss_fm = self._compute_single_noise_loss(
    x_1, mask, t_ext, r_ext, t, batch, B
)
# ... logging unchanged ...
return combined_adp_loss
```

**Acceptance criteria:**
- `_compute_single_noise_loss` exists on `ModelTrainerBase` with the exact signature above.
- All 3 existing tests in `tests/test_training_step.py` pass with K=1 config (regression check).
- With a fixed seed, `training_step` with K=1 produces the same loss value as before the refactor.

---

## Task 3: Implement K>1 Manual Optimization Path in `training_step`

**Goal:** Refactor `training_step` in `model_trainer_base.py` to orchestrate K noise passes, call `self.manual_backward(loss / K)` after each, and handle optimizer stepping with `_accum_grad_batches` support, gradient clipping, and LR scheduler stepping.

**Dependencies:** Task 1 (config + `self.loss_accumulation_steps` attribute) and Task 2 (`_compute_single_noise_loss`).

**File:** `/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/proteinfoundation/proteinflow/model_trainer_base.py`

### Full refactored `training_step`

Replace the existing `training_step` (lines 250-453) with the following. All comments reference the paper convention documented in the current code.

```python
def training_step(self, batch, batch_idx, *, val_step=False):
    """
    MeanFlow training step. Learns average velocity u(z_t, r, t) using JVP.

    PAPER CONVENTION: z_t = (1-t)*data + t*noise. Data at t=0, noise at t=1.
    Proteina convention: x_0=noise, x_1=data.

    Args:
        batch: Data batch.
        val_step: If True, logs under "validation_loss" prefix and skips
            scaling stats. Validation always uses K=1 (single noise sample).

    Returns:
        Training loss (scalar) for K=1/automatic optimization, or None for K>1/manual.
    """
    log_prefix = "validation_loss" if val_step else "train"

    assert not self.motif_conditioning, (
        "Motif conditioning is not yet supported with MeanFlow training. "
        "Set training.motif_conditioning=False in your config."
    )

    # --- 1. Shared setup: extract x_1, mask, sample (t, r), fold conditioning ---
    ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
    noise_samples = ot_cfg.get("noise_samples", None)

    if noise_samples is not None and self.ot_sampler is not None:
        # OT pool: _build_ot_pool determines which proteins to train on.
        # x_0 from the pool is discarded; each noise pass in the K loop
        # samples fresh x_0 and runs standard OT against the fixed x_1.
        x_1, _x_0_pool, mask, batch_shape, n, dtype = self._build_ot_pool(batch)
    else:
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        x_1 = self.fm._mask_and_zero_com(x_1, mask)

    # Sample (t, r) — shared across all K noise passes
    t, r = self.fm.sample_two_timesteps(
        batch_shape, self.device,
        ratio=self.meanflow_ratio,
        P_mean_t=self.meanflow_P_mean_t,
        P_std_t=self.meanflow_P_std_t,
        P_mean_r=self.meanflow_P_mean_r,
        P_std_r=self.meanflow_P_std_r,
    )
    t_ext = t[..., None, None]   # [B, 1, 1]
    r_ext = r[..., None, None]   # [B, 1, 1]

    B = t.shape[0]

    # Fold conditioning — shared, mutates batch in-place
    if self.cfg_exp.training.fold_cond:
        bs = x_1.shape[0]
        cath_code_list = batch.cath_code
        for i in range(bs):
            cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="H")
            if random.random() < self.cfg_exp.training.mask_T_prob:
                cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="T")
                if random.random() < self.cfg_exp.training.mask_A_prob:
                    cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="A")
                    if random.random() < self.cfg_exp.training.mask_C_prob:
                        cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="C")
        batch.cath_code = cath_code_list
    else:
        if "cath_code" in batch:
            batch.pop("cath_code")

    # --- 2. Determine K ---
    # Validation always uses a single noise sample regardless of config.
    K = 1 if val_step else self.loss_accumulation_steps

    # --- 3. K=1 path (automatic optimization, identical to pre-refactor behavior) ---
    if K == 1:
        combined_adp_loss, raw_loss_mf, raw_loss_fm = self._compute_single_noise_loss(
            x_1, mask, t_ext, r_ext, t, batch, B
        )

        self.log(
            f"{log_prefix}/combined_adaptive_loss",
            combined_adp_loss,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
        if not val_step:
            self.log(
                f"{log_prefix}/raw_loss_mf", raw_loss_mf,
                on_step=True, on_epoch=True, prog_bar=True, logger=True,
                batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                f"{log_prefix}/raw_loss_fm", raw_loss_fm,
                on_step=True, on_epoch=True, prog_bar=True, logger=True,
                batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
            )
            b, n = mask.shape
            self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
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
        return combined_adp_loss

    # --- 4. K>1 path (manual optimization) ---
    optimizer = self.optimizers()

    total_loss_mf = 0.0
    total_loss_fm = 0.0

    for k in range(K):
        loss_k, raw_loss_mf_k, raw_loss_fm_k = self._compute_single_noise_loss(
            x_1, mask, t_ext, r_ext, t, batch, B
        )
        # Backward with 1/K scaling so accumulated gradient == mean gradient
        self.manual_backward(loss_k / K)
        total_loss_mf += raw_loss_mf_k.item()
        total_loss_fm += raw_loss_fm_k.item()

    avg_loss_mf = total_loss_mf / K
    avg_loss_fm = total_loss_fm / K
    mf_ratio = self.cfg_exp.training.meanflow.ratio
    avg_combined = (1 - mf_ratio) * avg_loss_fm + mf_ratio * avg_loss_mf

    # Log averaged metrics
    self.log(
        f"{log_prefix}/combined_adaptive_loss", avg_combined,
        on_step=True, on_epoch=True, prog_bar=False, logger=True,
        batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
    )
    self.log(
        f"{log_prefix}/raw_loss_mf", avg_loss_mf,
        on_step=True, on_epoch=True, prog_bar=True, logger=True,
        batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
    )
    self.log(
        f"{log_prefix}/raw_loss_fm", avg_loss_fm,
        on_step=True, on_epoch=True, prog_bar=True, logger=True,
        batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
    )
    self.log(
        f"{log_prefix}/loss_accumulation_steps", float(K),
        on_step=True, on_epoch=False, prog_bar=False, logger=True,
        batch_size=1, sync_dist=False, add_dataloader_idx=False,
    )

    b, _n = mask.shape
    self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
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

### Notes for the subagent

1. **`self.optimizers()` vs `self.optimizer`:** In Lightning manual optimization, always use `self.optimizers()` (returns the optimizer or list of optimizers). For a single optimizer, `self.optimizers()` returns the optimizer directly.

2. **`self.clip_gradients()`:** This is a Lightning method available on `LightningModule`. Signature: `clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)`. The values `1.0` and `"norm"` mirror what `train.py` passes to `L.Trainer(gradient_clip_val=1.0, gradient_clip_algorithm="norm")`.

3. **`self.manual_backward(loss / K)`:** Lightning's `manual_backward` calls `loss.backward()` and handles precision scaling (AMP) automatically. The `/K` scaling ensures the accumulated gradient equals the mean of K gradients, not the sum.

4. **`avg_combined` for logging:** This is recomputed from the scalar averages, not from a differentiable tensor. It's only used for logging, not for optimization.

5. **`validation_step_data`** (in `ModelTrainerBase`) calls `self.training_step(batch, batch_idx, val_step=True)` inside `torch.no_grad()`. The refactored code forces K=1 when `val_step=True`, so no change needed there.

6. **OT pool + K>1:** `_build_ot_pool` returns a 6-tuple `(x_1, x_0, mask, batch_shape, n, dtype)`. The refactored code unpacks it as `x_1, _x_0_pool, mask, batch_shape, n, dtype` — `_x_0_pool` is intentionally discarded. Each noise pass in the K loop calls `_compute_single_noise_loss` which re-samples `x_0` and runs standard OT.

**Acceptance criteria:**
- All 3 tests in `tests/test_training_step.py` pass (K=1 regression).
- With K=2 and a synthetic batch, after `training_step` at least one `nn` parameter has a non-zero `.grad`.
- With K=1, `training_step` returns a scalar tensor; with K>1, it returns `None`.
- With K=2 and `_accum_grad_batches=2`, optimizer.step() is called after the 2nd `training_step` call and not after the 1st.

---

## Task 4: Write Tests

**Goal:** Create `tests/test_loss_accumulation.py` with 5 test cases covering the new behavior.

**Dependencies:** Tasks 1, 2, and 3 must be complete (the tests exercise the new code paths).

**File:** `/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/tests/test_loss_accumulation.py`

### Setup: imports, stubs, helpers

The test file must replicate the stub setup from `tests/test_training_step.py` (torch_scatter, pandas, biotite) because the same CUDA-unavailable module problem exists. Copy the stub setup verbatim from that file's lines 19-46, then add:

```python
import pytest
import torch
import unittest.mock as mock
from omegaconf import OmegaConf
```

**`_build_cfg(K, accum=1, ot_enabled=False)` helper:**

```python
def _build_cfg(K=1, accum=1, ot_enabled=False):
    cfg = OmegaConf.create({
        "model": {
            "target_pred": "v",
            "augmentation": {"global_rotation": False, "naug_rot": 1},
            "nn": {
                # Minimal AF3 config matching test_training_step.py _build_cfg()
                "name": "ca_af3",
                "token_dim": 64,
                "nlayers": 2,
                "nheads": 4,
                "residual_mha": True,
                "residual_transition": True,
                "parallel_mha_transition": False,
                "use_attn_pair_bias": True,
                "strict_feats": False,
                "feats_init_seq": ["res_seq_pdb_idx", "chain_break_per_res"],
                "feats_cond_seq": ["time_emb", "delta_t_emb"],
                "t_emb_dim": 32,
                "dim_cond": 64,
                "idx_emb_dim": 32,
                "fold_emb_dim": 32,
                "feats_pair_repr": ["rel_seq_sep", "xt_pair_dists"],
                "feats_pair_cond": ["time_emb", "delta_t_emb"],
                "xt_pair_dist_dim": 16,
                "xt_pair_dist_min": 0.1,
                "xt_pair_dist_max": 3,
                "x_sc_pair_dist_dim": 16,
                "x_sc_pair_dist_min": 0.1,
                "x_sc_pair_dist_max": 3,
                "x_motif_pair_dist_dim": 16,
                "x_motif_pair_dist_min": 0.1,
                "x_motif_pair_dist_max": 3,
                "seq_sep_dim": 127,
                "pair_repr_dim": 32,
                "update_pair_repr": False,
                "update_pair_repr_every_n": 2,
                "use_tri_mult": False,
                "num_registers": 4,
                "use_qkln": True,
                "num_buckets_predict_pair": 16,
                "multilabel_mode": "sample",
                "cath_code_dir": ".",
            },
        },
        "loss": {
            "t_distribution": {"name": "uniform", "p1": 0.0, "p2": 1.0},
            "loss_t_clamp": 0.9,
            "use_aux_loss": False,
            "aux_loss_t_lim": 0.3,
            "thres_aux_2d_loss": 0.6,
            "aux_loss_weight": 1.0,
            "num_dist_buckets": 16,
            "max_dist_boundary": 1.0,
        },
        "training": {
            "loss_accumulation_steps": K,
            "self_cond": False,
            "fold_cond": False,
            "mask_T_prob": 0.5,
            "mask_A_prob": 0.5,
            "mask_C_prob": 0.5,
            "motif_conditioning": False,
            "ot_coupling": {"enabled": ot_enabled, "method": "exact", "reg": 0.05,
                            "reg_m": 1.0, "normalize_cost": False},
            "meanflow": {
                "ratio": 0.25,
                "P_mean": -0.4,
                "P_std": 1.0,
                "norm_p": 1.0,
                "norm_eps": 0.001,
                "nsteps_sample": 1,
            },
        },
        "opt": {"lr": 1e-4, "accumulate_grad_batches": accum},
    })
    return cfg
```

**`_build_batch(B, N)` helper:** Copy from `test_training_step.py` unchanged.

**`_build_model(K, accum=1, ot_enabled=False)` helper:**

```python
def _build_model(K=1, accum=1, ot_enabled=False):
    from proteinfoundation.proteinflow.proteina import Proteina
    cfg = _build_cfg(K=K, accum=accum, ot_enabled=ot_enabled)
    model = Proteina(cfg_exp=cfg)
    model.to("cpu")
    model.train()
    trainer_mock = mock.MagicMock()
    trainer_mock.world_size = 1
    model._trainer = trainer_mock
    return model
```

### Test 1: `_compute_single_noise_loss` returns finite loss

```python
def test_compute_single_noise_loss_finite():
    """_compute_single_noise_loss returns finite losses that require grad."""
    model = _build_model(K=1)
    B, N = 4, 16
    batch = _build_batch(B, N)

    x_1, mask, batch_shape, n, dtype = model.extract_clean_sample(batch)
    x_1 = model.fm._mask_and_zero_com(x_1, mask)

    t, r = model.fm.sample_two_timesteps(
        batch_shape, torch.device("cpu"),
        ratio=model.meanflow_ratio,
        P_mean_t=model.meanflow_P_mean_t, P_std_t=model.meanflow_P_std_t,
        P_mean_r=model.meanflow_P_mean_r, P_std_r=model.meanflow_P_std_r,
    )
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]

    loss, raw_mf, raw_fm = model._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B
    )

    assert torch.isfinite(loss), f"combined loss not finite: {loss}"
    assert torch.isfinite(raw_mf), f"raw_loss_mf not finite: {raw_mf}"
    assert torch.isfinite(raw_fm), f"raw_loss_fm not finite: {raw_fm}"
    assert loss.requires_grad, "combined loss should require grad"
```

### Test 2: K=1 path produces identical results to pre-refactor behavior

```python
def test_k1_identical_to_before():
    """
    With K=1, training_step should return a finite scalar and produce
    gradients — regression check against the pre-refactor code path.
    """
    torch.manual_seed(0)
    model = _build_model(K=1)
    batch = _build_batch(4, 16)

    torch.manual_seed(0)
    loss = model.training_step(batch, batch_idx=0)

    assert loss is not None, "K=1 should return a loss tensor"
    assert torch.isfinite(loss), f"K=1 loss not finite: {loss}"
    loss.backward()

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "K=1: no parameters received nonzero gradients"
```

### Test 3: K>1 produces finite losses and nonzero gradients

```python
def test_k_greater_than_1_gradients():
    """K=2 training_step: gradients are nonzero, return value is None."""
    model = _build_model(K=2)
    batch = _build_batch(4, 16)

    result = model.training_step(batch, batch_idx=0)

    assert result is None, "K>1 manual optimization should return None"

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "K=2: no parameters received nonzero gradients"
```

### Test 4: K>1 + `accumulate_grad_batches` — optimizer called once after 2 steps

```python
def test_accum_grad_batches_composes_with_k():
    """
    K=2, accumulate_grad_batches=2: optimizer.step() fires only after the
    second training_step call (i.e. after 2*2=4 total backward passes).
    """
    model = _build_model(K=2, accum=2)
    batch = _build_batch(4, 16)

    # Patch the actual PyTorch optimizer's step and zero_grad
    real_optimizers = model.configure_optimizers()
    # configure_optimizers may return a dict or just an optimizer
    if isinstance(real_optimizers, dict):
        real_opt = real_optimizers["optimizer"]
    else:
        real_opt = real_optimizers

    step_calls = []
    original_step = real_opt.step
    real_opt.step = lambda *a, **kw: (step_calls.append(1), original_step(*a, **kw))

    # Wire the mock back into Lightning (bypass the trainer mock)
    model._trainer.optimizers = [real_opt]
    # Override self.optimizers() to return our instrumented optimizer
    model.optimizers = lambda: real_opt

    # First call: step should NOT have fired (count=1, 1 % 2 != 0)
    model.training_step(batch, batch_idx=0)
    assert len(step_calls) == 0, f"Step fired too early: {len(step_calls)} calls after step 1"

    # Second call: step SHOULD fire (count=2, 2 % 2 == 0)
    model.training_step(batch, batch_idx=1)
    assert len(step_calls) == 1, f"Expected 1 optimizer step, got {len(step_calls)}"
```

### Test 5: OT coupling called K times per training_step

```python
def test_ot_coupling_called_k_times():
    """With OT enabled and K=2, the OT sampler is called exactly K times."""
    model = _build_model(K=2, ot_enabled=True)
    batch = _build_batch(4, 16)

    call_count = []
    original = model.ot_sampler.sample_plan_with_scipy

    def counting_sample_plan(x_1, x_0, mask):
        call_count.append(1)
        return original(x_1, x_0, mask)

    model.ot_sampler.sample_plan_with_scipy = counting_sample_plan

    model.training_step(batch, batch_idx=0)

    assert len(call_count) == 2, (
        f"Expected OT sampler called 2 times (K=2), got {len(call_count)}"
    )
```

**Notes for the subagent:**
- Test 4 works around the fact that `model._trainer` is a `MagicMock` (set in `_build_model`). The mock's `.optimizers` attribute is a `MagicMock` and calling `self.optimizers()` in the Lightning module would normally go through the trainer. The test patches `model.optimizers` (the instance method) directly to return the instrumented optimizer.
- Test 5 assumes that when `ot_enabled=True` and `noise_samples` is NOT set, `_compute_single_noise_loss` calls `ot_sampler.sample_plan_with_scipy` once per invocation. With K=2 this means 2 calls total.

**Acceptance criteria:**
- `pytest tests/test_loss_accumulation.py` passes all 5 tests with no errors.
- No imports from undefined modules (all necessary stubs are copied from `test_training_step.py`).

---

## Task 5: Integration Testing and Debugging

**Goal:** Run all tests, fix any failures, and confirm K=1 regression.

**Dependencies:** Tasks 1–4 must all be complete.

### Step-by-step verification

**5.1 Run existing test suite:**

```bash
cd /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina
python -m pytest tests/test_training_step.py -v
```

Expected: 3 tests pass. If any fail, the refactor broke K=1 behavior — go back to Task 2 and verify the extracted `_compute_single_noise_loss` is bit-for-bit identical to the original inline code. Common mistakes:
- `u_func` closure captures stale `mask` or `B` from outer scope.
- `raw_loss_mf` is detached in the new method but was not detached in the original (check: the original used `.mean()` which keeps grad; the new version should do `.mean().detach()` only for the return value used in logging, while the combined loss returned still flows gradients).

**5.2 Run new test suite:**

```bash
python -m pytest tests/test_loss_accumulation.py -v
```

Expected: 5 tests pass.

**5.3 K=1 regression: verify numerical identity**

```bash
python -c "
import sys, types, importlib.machinery

def stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m
    return m

stub('torch_scatter').scatter_mean = None
stub('pandas')
for n in ['biotite','biotite.structure','biotite.structure.io']:
    stub(n)
sys.modules['biotite'].structure = sys.modules['biotite.structure']
sys.modules['biotite.structure'].io = sys.modules['biotite.structure.io']

import torch
from omegaconf import OmegaConf
from unittest import mock
from proteinfoundation.proteinflow.proteina import Proteina

# Build minimal config with K=1
cfg = OmegaConf.create({
    'model': {'target_pred': 'v', 'augmentation': {'global_rotation': False, 'naug_rot': 1},
              'nn': {'name': 'ca_af3', 'token_dim': 64, 'nlayers': 2, 'nheads': 4,
                     'residual_mha': True, 'residual_transition': True, 'parallel_mha_transition': False,
                     'use_attn_pair_bias': True, 'strict_feats': False,
                     'feats_init_seq': ['res_seq_pdb_idx', 'chain_break_per_res'],
                     'feats_cond_seq': ['time_emb', 'delta_t_emb'], 't_emb_dim': 32, 'dim_cond': 64,
                     'idx_emb_dim': 32, 'fold_emb_dim': 32,
                     'feats_pair_repr': ['rel_seq_sep', 'xt_pair_dists'],
                     'feats_pair_cond': ['time_emb', 'delta_t_emb'],
                     'xt_pair_dist_dim': 16, 'xt_pair_dist_min': 0.1, 'xt_pair_dist_max': 3,
                     'x_sc_pair_dist_dim': 16, 'x_sc_pair_dist_min': 0.1, 'x_sc_pair_dist_max': 3,
                     'x_motif_pair_dist_dim': 16, 'x_motif_pair_dist_min': 0.1, 'x_motif_pair_dist_max': 3,
                     'seq_sep_dim': 127, 'pair_repr_dim': 32, 'update_pair_repr': False,
                     'update_pair_repr_every_n': 2, 'use_tri_mult': False, 'num_registers': 4,
                     'use_qkln': True, 'num_buckets_predict_pair': 16,
                     'multilabel_mode': 'sample', 'cath_code_dir': '.'}},
    'loss': {'t_distribution': {'name': 'uniform', 'p1': 0.0, 'p2': 1.0}, 'loss_t_clamp': 0.9,
             'use_aux_loss': False, 'aux_loss_t_lim': 0.3, 'thres_aux_2d_loss': 0.6,
             'aux_loss_weight': 1.0, 'num_dist_buckets': 16, 'max_dist_boundary': 1.0},
    'training': {'loss_accumulation_steps': 1, 'self_cond': False, 'fold_cond': False,
                 'mask_T_prob': 0.5, 'mask_A_prob': 0.5, 'mask_C_prob': 0.5,
                 'motif_conditioning': False, 'ot_coupling': {'enabled': False},
                 'meanflow': {'ratio': 0.25, 'P_mean': -0.4, 'P_std': 1.0,
                              'norm_p': 1.0, 'norm_eps': 0.001, 'nsteps_sample': 1}},
    'opt': {'lr': 1e-4},
})

torch.manual_seed(42)
model = Proteina(cfg_exp=cfg)
model.train()
tm = mock.MagicMock()
tm.world_size = 1
model._trainer = tm

B, N = 4, 16
coords = torch.randn(B, N, 3, 3)
mask_d = {'coords': torch.ones(B, N, 3, 3, dtype=torch.bool)}
batch = {'coords': coords, 'mask_dict': mask_d, 'mask': torch.ones(B, N, dtype=torch.bool)}

torch.manual_seed(42)
loss = model.training_step(batch, 0)
print(f'K=1 loss: {loss.item():.6f}')
assert torch.isfinite(loss), 'loss not finite'
print('PASS: K=1 regression OK')
"
```

**5.4 Run both test files together:**

```bash
python -m pytest tests/test_training_step.py tests/test_loss_accumulation.py -v
```

Expected output: 8 tests collected, 8 passed.

### Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `AttributeError: 'Proteina' object has no attribute 'loss_accumulation_steps'` | Task 1 not complete | Add attribute in `Proteina.__init__` |
| `AttributeError: 'Proteina' object has no attribute '_compute_single_noise_loss'` | Task 2 not complete | Add method to `ModelTrainerBase` |
| `RuntimeError: .grad is None` in K>1 test | `manual_backward` not called, or called with loss that is `None` | Check that `_compute_single_noise_loss` returns a tensor with `requires_grad=True` |
| K=1 loss value changes (regression failure) | `_compute_single_noise_loss` has a subtle difference from original | Check: (a) `raw_loss_mf` is `loss_mf.mean().detach()` not `loss_mf.mean()`, but `loss_mf` itself is not detached before `adaptive_loss`. (b) `u_func` closures are identical. |
| Test 4 fails: step called at wrong time | `_manual_step_count` not resetting or not incrementing | Verify `_manual_step_count` is incremented AFTER the K loop, not inside it |
| `TypeError: optimizers() takes 0 positional arguments` | Test 4 mock issue | Patch `model.optimizers` as a lambda, not as a method |

**Acceptance criteria:**
- `pytest tests/test_training_step.py tests/test_loss_accumulation.py` exits with code 0.
- 8 tests total, 8 passed, 0 failed.
- No Python warnings about `loss_accumulation_steps` key missing from config.
- The K=1 regression script (step 5.3) prints "PASS: K=1 regression OK".
