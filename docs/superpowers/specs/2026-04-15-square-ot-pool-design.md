# Square OT Pool for Small-Batch Flow Matching

## Problem

The current rectangular OT implementation oversamples noise (B×B*K cost matrix) and assigns each of B data samples to its best noise match from B*K candidates. This induces bias: the selected noise distribution is systematically shifted toward the data distribution rather than being a proper OT coupling from a larger joint plan.

With only 2-4 proteins fitting on GPU, the square-batch OT (B×B, Branch 2) is too small to be meaningful — a 2×2 or 4×4 assignment is nearly trivial.

## Solution

Build a large square OT pool (e.g., 128×128) entirely on CPU by sampling extra proteins from the dataset, compute a proper square OT assignment, then randomly select B pairs to use for the GPU training step.

## Data Flow

When `ot_coupling.enabled=True` and `noise_samples` is set (e.g., 128):

1. Dataloader yields batch of B=2 proteins (CPU).
2. Extract CA coords + masks from the batch: `x_1 [B, N_batch, 3]`, `mask [B, N_batch]`.
3. Sample `noise_samples - B` extra proteins from the dataset via random indices. Extract their CA coords + masks.
4. Pad all `noise_samples` proteins to `N_pool = max length in pool`. Build `x_1_pool [noise_samples, N_pool, 3]` and `mask_pool [noise_samples, N_pool]`.
5. Sample `noise_samples` noise vectors on CPU: `x_0_pool [noise_samples, N_pool, 3]`.
6. Apply mask + zero COM to both pools.
7. Build cost matrix `M [noise_samples, noise_samples]` where `M[i,j] = ||x_1_pool[i] - x_0_pool[j]||^2` (masked).
8. Run Hungarian algorithm (scipy) on M → permutation `sigma`.
9. Randomly pick B indices `k` from `{0..noise_samples-1}` → pairs `(x_1_pool[k], x_0_pool[sigma[k]])`.
10. Re-pad selected B pairs to their own max length, move to GPU.
11. Continue training with selected `x_1`, `x_0`, `mask` — downstream loss computation unchanged.

**Debug case:** Step 3 draws random indices from the dataset (which only contains repeated 1ubq), so the pool is filled with copies of the same protein. OT assignments are valid but degenerate — acceptable for pipeline testing.

**General case:** Step 3 draws random indices from the full training dataset, loading diverse proteins to fill the pool.

## Code Architecture

### New method: `Proteina._build_ot_pool(batch, noise_samples)`

Located in `proteina.py`. Encapsulates steps 2-10 above. Returns `x_1 [B, N, 3]`, `x_0 [B, N, 3]`, `mask [B, N]`, `batch_shape`, `n`, `dtype` — everything `training_step` needs to proceed. All tensors are on `self.device` (GPU).

**Responsibilities:**
- Extract CA coords from the dataloader batch
- Sample extra proteins from `self._ot_dataset` using random indices
- Pad all proteins to uniform length with masks
- Sample noise, apply masking and zero COM
- Compute cost matrix and run Hungarian
- Randomly select B pairs
- Re-pad to selected max length, move to device

### Dataset reference: `self._ot_dataset`

Set in `Proteina.on_train_start()`:
```python
self._ot_dataset = self.trainer.datamodule.train_dataset
```

This gives direct access to individual protein samples for the OT pool without going through the dataloader.

### Changes to `model_trainer_base.py` training_step

Replace the current two-branch OT logic (lines 290-327) with:
- If `noise_samples` is set and `ot_sampler` is not None → call `self._build_ot_pool(batch, noise_samples)` to get `x_1, x_0, mask` and derived `batch_shape, n, dtype`. All downstream code uses these.
- If `noise_samples` is None and `ot_sampler` is not None → existing square-batch OT (Branch 2), unchanged.
- If `ot_sampler` is None → standard random noise, unchanged.

### Config interface

The existing `noise_samples` parameter changes semantics:
```yaml
training:
  ot_coupling:
    enabled: True
    method: exact
    noise_samples: 128    # total OT pool size (data AND noise)
```

No new parameters. The old rectangular-OT behavior (oversample noise only) is removed.

### Validation

Assert at init: `noise_samples >= batch_size` when both are set. If `noise_samples == batch_size`, degenerates to standard square-batch OT.

## Edge Cases and Constraints

- **Padding:** OT pool pads to longest protein in pool. Individual protein lengths are tracked via `mask_pool`. After B-pair selection, the selected masks determine the actual max length `N_sel <= N_pool`, and tensors are sliced to `[:, :N_sel, :]` before moving to GPU.
- **zero_com:** Applied per-sample with masks before OT cost computation, and again to final pairs after re-padding.
- **motif_conditioning:** Existing assert (`not motif_conditioning` for MeanFlow) stays. OT pool path is MeanFlow-only.
- **Gradient accumulation:** Each micro-step independently builds its own OT pool. No state across steps.
- **Performance:** 128×128 Hungarian is ~1ms. Extra protein loads are CPU-only; `in_memory=True` makes them tensor copies, disk-backed datasets rely on OS page cache.

## Files Modified

1. `proteinfoundation/proteinflow/proteina.py` — add `on_train_start` hook, `_build_ot_pool` method
2. `proteinfoundation/proteinflow/model_trainer_base.py` — replace OT branches in `training_step`
3. `configs/experiment_config/training_ca_debug.yaml` — ensure `noise_samples: 128` with new semantics
4. `configs/experiment_config/training_ca.yaml` — add `noise_samples` if desired for production runs
