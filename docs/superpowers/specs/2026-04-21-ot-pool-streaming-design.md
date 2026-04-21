# Streaming OT Pool for Flow Matching

## Problem

The current OT pool (see `2026-04-15-square-ot-pool-design.md`) is rebuilt
every training step: sample `K` proteins + `K` noise vectors on CPU, run a
Hungarian algorithm over a `K×K` cost matrix, randomly pick `B` pairs from
the assignment, and discard the remaining `K-B` pairs.

This wastes `(K-B)/K` of the work each step. At the typical setting
`K=128, B=4`, **97% of the Hungarian-coupled pairs are thrown away** before
the next step rebuilds a fresh pool with entirely new random data/noise.

## Solution

Compute the pool once, then **stream `K/B` consecutive training batches out
of it** (drawn at random without replacement) before refilling. Only 1 in
every `K/B` steps pays the Hungarian + dataset-sampling cost.

The pool lives on CPU. Only the B pairs that a given step will train on are
transferred to GPU, and they are trimmed to their own actual max length
before transfer — never padded to the pool's `N_pool`.

## Data Flow (steady state)

Let `K` = `ot_pool_size`, `B` = `batch_size`. `K` is floored to the nearest
multiple of `B` at `on_train_start` (warning logged if truncation happens).

```
step t        → pool.empty? yes → refill: sample K random proteins + K noise
                                         → pad to N_pool, mask, zero-COM
                                         → cost matrix [K,K], Hungarian
                                         → x_0 reordered by sigma so (x_1[i],x_0[i]) is the OT pair
                                         → random permutation over {0..K-1}
                                         → cursor = 0
             → next_batch(device):
                  idx = perm[cursor:cursor+B]; cursor += B
                  slice x_1, x_0, mask to those B rows
                  trim to max actual length among selected
                  mask + zero-COM after trim
                  .to(device) → ship to GPU

step t+1     → pool.empty? no (cursor = B) → nop
             → next_batch → pops next B
...
step t+K/B-1 → next_batch → pops last B, cursor = K → pool now "empty"
step t+K/B   → pool.empty? yes → refill ...
```

**Random without replacement.** A single `torch.randperm(K)` is drawn at
refill time; `cursor` walks through it `B` indices at a time. Every protein
in a pool is used exactly once before the pool is rebuilt.

**Refill is lazy.** It happens on the step that finds `cursor == K`, i.e.
*after* the final batch of the prior pool has been served. The serving step
itself does not refill — no step ever pays both the refill cost and the
training cost.

**Dataloader batch is ignored.** The dataloader still advances (Lightning
needs it for epoch accounting), but its contents are discarded in the pool
path. Pool `x_1` is drawn entirely from random dataset indices.

## Components

### New: `OTPool` class (`proteinfoundation/flow_matching/ot_pool.py`)

Owns pool state and lifecycle. All stored tensors are on CPU.

```python
class OTPool:
    def __init__(
        self,
        pool_size: int,
        batch_size: int,
        dim: int,
        scale_ref: float,
    ):
        # Floor pool_size to nearest multiple of batch_size; warn if truncated.
        effective = (pool_size // batch_size) * batch_size
        if effective != pool_size:
            logger.warning(
                "ot_pool_size=%d not divisible by batch_size=%d; flooring to %d",
                pool_size, batch_size, effective,
            )
        assert effective >= batch_size, "pool_size must be >= batch_size"
        self.pool_size = effective
        self.batch_size = batch_size
        self.dim = dim
        self.scale_ref = scale_ref
        # Empty state → empty == True on first call.
        self._x1 = None
        self._x0 = None
        self._mask = None
        self._perm = None
        self._cursor = self.pool_size

    @property
    def empty(self) -> bool:
        return self._cursor >= self.pool_size

    def refill(
        self,
        dataset,
        fm,                      # for _mask_and_zero_com
        extract_clean_sample,    # callable: batch -> (x_1, mask, bshape, n, dtype)
    ) -> None:
        # dtype is determined from the proteins loaded during refill
        # (via extract_clean_sample), not passed in.
        # 1. Sample K random dataset indices, load proteins.
        # 2. Extract CA coords + masks, mask_and_zero_com, move to CPU.
        # 3. Pad across the pool to N_pool (max over selected proteins).
        # 4. Sample x_0_pool [K, N_pool, 3] on CPU, mask_and_zero_com.
        # 5. Build cost matrix M [K,K] via torch.cdist on flat masked coords.
        # 6. Run scipy.optimize.linear_sum_assignment → sigma.
        # 7. Re-order x_0_pool by sigma so x_0_pool[i] pairs with x_1_pool[i].
        # 8. Store self._x1, self._x0, self._mask (all CPU).
        # 9. self._perm = torch.randperm(self.pool_size); self._cursor = 0.
        # 10. `del` cost matrix and flattened intermediates before returning.

    def next_batch(self, device):
        # Returns (x_1, x_0, mask, batch_shape, n, dtype) on `device`.
        assert not self.empty, "OTPool is empty — call refill() first."
        idx = self._perm[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        x_1_sel = self._x1[idx]      # [B, N_pool, 3] CPU
        x_0_sel = self._x0[idx]      # [B, N_pool, 3] CPU
        mask_sel = self._mask[idx]   # [B, N_pool]    CPU
        # Trim to actual max length among selected.
        n_sel = int(mask_sel.sum(dim=1).max().item())
        if n_sel == 0:
            n_sel = x_1_sel.shape[1]
        x_1_sel = x_1_sel[:, :n_sel, :]
        x_0_sel = x_0_sel[:, :n_sel, :]
        mask_sel = mask_sel[:, :n_sel]
        # Re-apply mask + zero COM after trim (trim can change COM via padding removal).
        x_1_sel = fm._mask_and_zero_com(x_1_sel, mask_sel)  # inject fm via ctor or param
        x_0_sel = fm._mask_and_zero_com(x_0_sel, mask_sel)
        return (
            x_1_sel.to(device),
            x_0_sel.to(device),
            mask_sel.to(device),
            torch.Size([self.batch_size]),
            n_sel,
            self._x1.dtype,
        )
```

Implementation note: `fm._mask_and_zero_com` can either be passed to
`next_batch` or stored on the pool at construction time. The plan will
decide. It does not affect the data flow.

### Modified: `Proteina` (`proteinfoundation/proteinflow/proteina.py`)

- **`on_train_start`**: construct `self._ot_pool = OTPool(pool_size,
  batch_size, self.fm.dim, self.fm.scale_ref)` when
  `ot_coupling.ot_pool_size` is set and `self.ot_sampler is not None`.
  Keep storing `self._ot_dataset` as today. Assert
  `self.loss_accumulation_steps == 1` in pool mode (fail fast with a
  clear message — pool + K>1 is unsupported by design).
- **Remove `_build_ot_pool`.** Its refill logic moves into `OTPool.refill()`.
- **Add `_get_ot_batch()`** (small helper):
  ```python
  def _get_ot_batch(self):
      if self._ot_pool.empty:
          self._ot_pool.refill(
              self._ot_dataset, self.fm, self.extract_clean_sample,
          )
      return self._ot_pool.next_batch(self.device)
  ```

### Modified: `model_trainer_base.py` training_step

Replace the `_build_ot_pool(batch)` call at line ~484 with a call to the
new getter. The dataloader `batch` is ignored on this path (pool `x_1` is
dataset-sampled inside `refill`).

```python
# Before:
# x_1, _x_0_pool, mask, batch_shape, n, dtype = self._build_ot_pool(batch)

# After: the pool's x_0 IS the training x_0 — no fresh re-sampling
# inside _compute_single_noise_loss when pool mode is active.
x_1, x_0_pool, mask, batch_shape, n, dtype = self._get_ot_batch()
```

Pool mode owns the (x_1, x_0) pairing end-to-end. To honor this,
`_compute_single_noise_loss` gains an optional `x_0_override` argument:

- **If provided:** skip the internal `self.fm.sample_reference(...)` and
  the internal `self.ot_sampler.sample_plan_with_scipy(...)` call; use
  `x_0_override` as-is. This is the pool-mode path.
- **If None (default):** current behavior — sample fresh x_0 and run B×B
  OT, exactly as today. This preserves the non-pool code path.

In `training_step`, pool mode passes `x_0_override=x_0_pool` on the K=1
branch. Because the pool supplies exactly one OT-paired x_0 per served
batch, **pool mode requires `loss_accumulation_steps == 1`** — asserted
at `on_train_start`. K>1 with the pool is explicitly unsupported (the
intended efficiency model pays one Hungarian per K/B dataloader steps;
running K extra noise passes per micro-step reintroduces the
fresh-noise-plus-B×B-OT the pool is meant to replace).

`_get_ot_batch` takes no dtype argument; the pool determines its own dtype
during `refill` from the first protein it loads (mirrors how
`_build_ot_pool` does it today via `extract_clean_sample`).

The inner K-loop over fresh noise samples (the old rectangular-OT vestige)
is untouched by this refactor — it continues to run as today for
`ot_coupling.enabled=True` with no `ot_pool_size`. Only the pool branch
changes.

### Config rename (hard): `noise_samples` → `ot_pool_size`

```yaml
# configs/experiment_config/training_ca.yaml
# configs/experiment_config/training_ca_debug.yaml
training:
  ot_coupling:
    enabled: True
    method: exact
    ot_pool_size: 128   # was: noise_samples
```

Any remaining reference to `noise_samples` is updated in the same change:

- `proteinfoundation/proteinflow/proteina.py`: reads `ot_cfg["ot_pool_size"]`
- `proteinfoundation/proteinflow/model_trainer_base.py`: reads
  `ot_cfg.get("ot_pool_size", None)`
- Any sbatch scripts in this branch's worktree that pass
  `training.ot_coupling.noise_samples=...` — rename the override key.

Configs containing `noise_samples` are rejected by Hydra's schema (the key
no longer exists). No soft-deprecation path.

## CPU/GPU Discipline (OOM prevention)

1. **Pool tensors (`_x1`, `_x0`, `_mask`) live exclusively on CPU.**
   `refill` never calls `.to(device)`. They are allocated once per pool
   cycle and released when the next `refill` overwrites them.
2. **Cost matrix `M [K,K]` and flattened `[K, N_pool*3]` views are CPU-only
   and local to `refill()`.** They are `del`-ed before `refill` returns so
   Python can release memory before the first `next_batch` runs.
3. **`next_batch` is the only device transfer.** It ships `[B, N_sel, 3]`
   tensors (trimmed to selected rows' max actual length, not `N_pool`).
4. **No retained refs across steps.** `OTPool` keeps only `_x1, _x0, _mask,
   _perm, _cursor`. All other intermediates are local variables.
5. **Refill boundary.** Because refill is lazy (happens on the step after
   the last-batch serve), a single step never holds both the previous
   pool's tensors and the next pool's tensors simultaneously — the
   reassignment in `refill` drops refs to the old CPU tensors before the
   new ones are constructed (done via local variables, then assignment).

## Edge Cases

- **`pool_size % batch_size != 0`.** Log a WARN and floor at
  `on_train_start`. Effective pool has an integer number of batches; no
  stragglers, no partial-batch step.
- **`pool_size == batch_size`.** Legal. Degenerates to a refill every
  step — equivalent to today's `_build_ot_pool`, minus the discard of
  `K - B` extra pairs (there are none).
- **Debug regime (1 protein in dataset).** `refill` samples K copies of
  the same protein. Hungarian is degenerate but valid. Same behavior as
  today's pool.
- **Validation step.** Unchanged. `training_step(val_step=True)` does not
  go through the pool (the pool only wraps training-time OT).
- **Gradient accumulation.** `accumulate_grad_batches=N` (Lightning-level
  accumulation) means `N` pool draws per optimizer step. No interaction
  with pool cycle length — they are independent dimensions.
- **`loss_accumulation_steps` (manual K-loop).** Pool mode requires
  `loss_accumulation_steps == 1`, enforced at `on_train_start`. Pool +
  K>1 is explicitly unsupported.
- **DDP / multi-GPU.** Each rank holds its own pool. Pool content differs
  across ranks (random dataset indices drawn independently). This matches
  today's behavior — `_build_ot_pool` also samples per-rank.

## Testing

New file `tests/test_ot_pool.py`:

1. **Floor + warn.** `OTPool(pool_size=130, batch_size=4)` → `pool_size
   == 128`; `caplog` contains the truncation warning.
2. **Refill produces valid OT.** With `K=4` and a tiny fake dataset,
   verify the stored `(x_1[i], x_0[i])` assignment equals the output of
   `scipy.optimize.linear_sum_assignment` on the same cost matrix (i.e.,
   pool's internal reordering is correct).
3. **Exhaustion cycle.** `pool_size=8, batch_size=4`:
   - First `next_batch` → pool not empty after (`cursor=4`).
   - Second `next_batch` → pool empty after (`cursor=8`).
   - Third call without refill → assertion error.
   - After manual `refill()` → two more distinct batches.
4. **No-replacement within a cycle.** Union of consecutive `next_batch`
   index sets within one cycle equals `{0..K-1}` exactly (each protein
   used once).
5. **CPU/GPU separation.**
   - After `refill()`: `pool._x1.device.type == 'cpu'`, same for `_x0`,
     `_mask`.
   - After `next_batch('cpu')`: returned tensors on CPU. If CUDA
     available, `next_batch('cuda')` returns tensors on CUDA *while pool
     internals remain on CPU*.
6. **Trim correctness.** Build a pool with mixed lengths (e.g., three
   proteins of length 20, one of length 80 → `N_pool=80`). Draw a batch
   whose selected rows have max actual length 20 → returned `x_1.shape[1]
   == 20`, not 80.
7. **End-to-end smoke.** One `training_step` in pool mode with `K=4,
   B=2` against a small real-ish fake dataset. Loss is finite, gradients
   flow into `self.nn.parameters()`, shapes match expectations.

Existing tests in `tests/test_training_step.py` and related must keep
passing after the rename. Any test that references `noise_samples`
directly is updated.

## Files Changed

- **New:** `proteinfoundation/flow_matching/ot_pool.py`
- **New:** `tests/test_ot_pool.py`
- **Modified:** `proteinfoundation/proteinflow/proteina.py` — remove
  `_build_ot_pool`, add `_get_ot_batch`, construct `OTPool` in
  `on_train_start`, read `ot_pool_size`.
- **Modified:** `proteinfoundation/proteinflow/model_trainer_base.py` —
  read `ot_pool_size` (not `noise_samples`), call `_get_ot_batch()`,
  thread `x_0_override` through `_compute_single_noise_loss` to bypass
  fresh-noise sampling and B×B OT when pool mode is active.
- **Modified:** `configs/experiment_config/training_ca.yaml` — add
  `ot_pool_size`.
- **Modified:** `configs/experiment_config/training_ca_debug.yaml` —
  rename `noise_samples` → `ot_pool_size`.
- **Modified:** any sbatch script in the current (train_1000) worktree
  that passes `training.ot_coupling.noise_samples=...` — rename the
  override. (Scripts in other worktrees stay pinned to their own code
  version and are out of scope.)

## Behavior changes vs. current pool

- **Pool x_0 is no longer discarded.** Previously the K×K Hungarian
  picked an OT pairing, the x_0 half was thrown away, and
  `_compute_single_noise_loss` resampled fresh x_0 + ran a separate B×B
  OT inside the K-loop. Now the pool's x_0 flows straight through to the
  JVP/FM loss — the K×K Hungarian is the *actual* OT coupling used.
- **Pool mode forbids `loss_accumulation_steps > 1`** (asserted). This
  aligns with the production sbatch (`LOSS_ACCUMULATION_STEPS=1`) and
  with the "one OT pair per training sample" contract.

## Non-goals

- **OT algorithm change.** Hungarian on flat `[K, N*3]` cost is kept
  as-is. This change is streaming + x_0-propagation, not a numerical OT
  change.
- **Sinkhorn / approximate OT.** Out of scope.
- **Modifying the non-pool path** (`ot_coupling.enabled=True` with no
  `ot_pool_size`). Untouched — still samples fresh x_0 and runs B×B OT
  in `_compute_single_noise_loss`.
- **Soft deprecation** of `noise_samples`. We rename hard — any lingering
  reference is fixed in the same PR.
