# Loss Accumulation for Increased Effective Batch Size

## Summary

Implement loss accumulation by running K forward passes per batch with different noise samples (`x_0`), averaging the losses, and performing a single gradient step. This increases the effective batch size without requiring more proteins in a batch.

**Effective batch size** = `batch_size x loss_accumulation_steps x accumulate_grad_batches`

## Configuration

New config key: `training.loss_accumulation_steps` (integer, default `1`).

```yaml
training:
  loss_accumulation_steps: 4  # K forward passes with different noise per batch
  ot_coupling:
    enabled: True
    method: exact
```

- `K=1`: disabled, training behaves exactly as today (automatic optimization).
- `K>1`: enables loss accumulation with manual optimization.

## Data Flow

For a single `training_step` call with `loss_accumulation_steps = K`:

```
batch (from dataloader)
  -> extract_clean_sample()          # x_1 [B, n, 3], mask, shared
  -> sample_two_timesteps()          # (t, r), shared across all K passes
  -> fold conditioning               # cath_code masking, shared
  -> FOR k = 1..K:
       -> sample_reference()         # x_0^(k) ~ N(0, I), unique per k
       -> OT coupling (if enabled)   # Hungarian assignment for x_0^(k), unique per k
       -> z^(k) = (1-t)*x_1 + t*x_0^(k)
       -> v^(k) = x_0^(k) - x_1
       -> JVP forward pass           # MeanFlow loss
       -> FM forward pass            # flow matching loss
       -> combined_loss_k
       -> manual_backward(combined_loss_k / K)
  -> handle optimizer step (every accumulate_grad_batches calls)
```

**Shared** across K noise passes: `x_1`, `mask`, `t`, `r`, `t_ext`, `r_ext`, `cath_code`, fold conditioning.

**Unique** per noise pass: `x_0`, OT permutation, `z`, `v`, all losses.

## Architecture

### Refactoring `training_step`

Extract the per-noise-sample computation into a new method `_compute_single_noise_loss()`.

#### `_compute_single_noise_loss(self, x_1, mask, t_ext, r_ext, batch, B, n, dtype, batch_shape, log_prefix, val_step)`

Encapsulates:
1. Sample `x_0` via `self.fm.sample_reference()`
2. Apply OT coupling if enabled:
   - If `noise_samples` config is set AND `self.ot_sampler` is not None: use `_build_ot_pool()` approach
   - If only `self.ot_sampler` is not None: standard square-batch OT via `sample_plan_with_scipy()`
3. Interpolate: `z = (1-t_ext)*x_1 + t_ext*x_0`, `v = x_0 - x_1`
4. Apply mask and zero COM to `z` and `v`
5. Compute JVP (MeanFlow loss via `torch.func.jvp`)
6. Compute FM loss (plain forward with `r=t`)
7. Apply adaptive loss weighting
8. Return `(combined_adaptive_loss, raw_loss_mf, raw_loss_fm)`

Note: the `u_func` closure is defined inside this method, capturing `mask`, `batch`, and `B`.

#### Refactored `training_step(self, batch, batch_idx, *, val_step=False)`

Orchestrator that:
1. Extracts clean samples, samples timesteps, handles fold conditioning (all shared).
2. Reads `K = self.loss_accumulation_steps`.
3. **K=1 path**: calls `_compute_single_noise_loss()` once, logs, returns the loss. Automatic optimization handles the rest (identical to current behavior).
4. **K>1 path**:
   - Loops K times, calling `_compute_single_noise_loss()` each iteration.
   - Calls `self.manual_backward(loss_k / K)` after each iteration (frees the computation graph, 1x memory).
   - Accumulates raw losses for logging: averages `raw_loss_mf` and `raw_loss_fm` over K.
   - Logs the averaged metrics.
   - Handles optimizer stepping (see below).
   - Returns `None` (manual optimization, Lightning doesn't need a loss).

### Manual Optimizer Management (K>1)

When `loss_accumulation_steps > 1`, `self.automatic_optimization = False`.

Lightning's `accumulate_grad_batches` is NOT used (it's ignored in manual mode). Instead, we reimplement it:

```python
# In __init__:
self._accum_grad_batches = cfg_exp.opt.get("accumulate_grad_batches", 1)
self._manual_step_count = 0

# In training_step (K>1 path), after the K-noise loop:
self._manual_step_count += 1
if self._manual_step_count % self._accum_grad_batches == 0:
    # Gradient clipping (train.py uses gradient_clip_val=1.0, algorithm="norm")
    self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
    optimizer.step()
    optimizer.zero_grad()
    # Step LR scheduler
    sch = self.lr_schedulers()
    if sch is not None:
        sch.step()
```

### OT Coupling per Noise Sample

When OT is enabled and K>1, each of the K noise samples gets its own OT assignment:

- **Standard square-batch OT** (`ot_sampler` enabled, no `noise_samples`): For each k, sample `x_0^(k)`, then run `ot_sampler.sample_plan_with_scipy(x_1, x_0^(k), mask)` to get the permuted `x_0^(k)`.
- **OT pool** (`noise_samples` configured): The `_build_ot_pool()` method returns paired `(x_1, x_0)` by sampling extra proteins and running Hungarian assignment. When combined with loss accumulation, the pool is built **once** per training_step to determine the data batch `(x_1, mask)`. Then the K noise passes each sample fresh `x_0^(k)` and OT-couple it to the fixed `x_1` using standard square-batch OT. The pool determines *which* proteins to train on; loss accumulation adds multiple noise draws per protein.

## Logging

For K>1, metrics are the **average** across K noise samples:

| Metric | Description |
|--------|-------------|
| `train/combined_adaptive_loss` | Mean of K combined adaptive losses |
| `train/raw_loss_mf` | Mean of K raw MeanFlow losses |
| `train/raw_loss_fm` | Mean of K raw FM losses |
| `train/loss_accumulation_steps` | K (for monitoring dashboards) |

Scaling stats (`nsamples_processed`, `nparams`) count actual data samples, not noise resamples.

## Validation

During validation (`val_step=True`), loss accumulation is **disabled** — always K=1. Validation uses `torch.no_grad()` and automatic optimization is not involved. The validation loss should reflect a single noise sample for consistency.

## Testing

New test file: `tests/test_loss_accumulation.py`

### Test 1: `_compute_single_noise_loss` returns finite loss
- Build model with standard config
- Call `_compute_single_noise_loss()` with synthetic batch
- Assert loss is finite, raw losses are finite
- Assert loss requires grad

### Test 2: K=1 path is identical to current behavior
- Build model with `loss_accumulation_steps=1`
- Seed RNG, run training_step, capture loss
- Compare against the old training_step (regression test)

### Test 3: K>1 produces finite loss and gradients
- Build model with `loss_accumulation_steps=2`
- Run training_step with synthetic batch
- Assert gradients are nonzero on model parameters

### Test 4: K>1 gradient accumulation composes with accumulate_grad_batches
- Build model with `loss_accumulation_steps=2`, `accumulate_grad_batches=2`
- Run 2 training_step calls
- Assert optimizer.step() is called exactly once (after 2nd call)

### Test 5: OT coupling applied per noise sample
- Build model with OT enabled and `loss_accumulation_steps=2`
- Mock `ot_sampler.sample_plan_with_scipy` to track call count
- Run training_step
- Assert OT sampler was called K=2 times

## Files Changed

| File | Change |
|------|--------|
| `proteinfoundation/proteinflow/model_trainer_base.py` | Extract `_compute_single_noise_loss()`, refactor `training_step()` with K=1/K>1 paths, add manual optimization logic |
| `proteinfoundation/proteinflow/proteina.py` | Read `loss_accumulation_steps` from config in `__init__`, set `self.automatic_optimization = False` when K>1, store `_accum_grad_batches` |
| `configs/experiment_config/training_ca.yaml` | Add `loss_accumulation_steps: 1` default |
| `configs/experiment_config/training_ca_debug.yaml` | Add `loss_accumulation_steps: 1` default |
| `tests/test_loss_accumulation.py` | New test file with 5 test cases |

## Constraints

- `torch.func.jvp` is incompatible with `torch.utils.checkpoint`, so no activation checkpointing. Memory per forward pass is full.
- Each noise sample requires 2 forward passes (1 JVP + 1 FM). K noise samples = 2K forward passes per training step.
- OT coupling per noise sample adds K Hungarian assignments (O(B^3) each). For large B this can be significant.
