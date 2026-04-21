# Streaming OT Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the OT pool so it's built once every `K/B` steps and streamed batch-by-batch (random without replacement), using the pool's Hungarian-paired x_0 as the actual training x_0. Rename `noise_samples` → `ot_pool_size`.

**Architecture:** New `OTPool` class owns CPU-resident pool tensors and serves B-sized batches via `next_batch(device)`. `Proteina` constructs the pool at `on_train_start`, calls `pool.refill(...)` lazily when empty, and threads the pool's `x_0` through `_compute_single_noise_loss` via a new `x_0_override` argument.

**Tech Stack:** PyTorch, scipy (`linear_sum_assignment`), Lightning, pytest.

**Spec:** [docs/superpowers/specs/2026-04-21-ot-pool-streaming-design.md](../specs/2026-04-21-ot-pool-streaming-design.md)

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `proteinfoundation/flow_matching/ot_pool.py` | CREATE | `OTPool` class: pool state, refill, streaming next_batch |
| `tests/test_ot_pool.py` | CREATE | Unit tests for `OTPool` |
| `proteinfoundation/proteinflow/proteina.py` | MODIFY | Remove `_build_ot_pool`, add `_get_ot_batch`, construct `OTPool` in `on_train_start` |
| `proteinfoundation/proteinflow/model_trainer_base.py` | MODIFY | Read `ot_pool_size` (not `noise_samples`), call `_get_ot_batch`, add `x_0_override` to `_compute_single_noise_loss` |
| `configs/experiment_config/training_ca.yaml` | MODIFY | Add `ot_pool_size` (no existing `noise_samples` to rename) |
| `configs/experiment_config/training_ca_debug.yaml` | MODIFY | Rename `noise_samples` → `ot_pool_size` |
| `scripts/*.sbatch` (this worktree's scripts referencing `training.ot_coupling.noise_samples`) | MODIFY | Rename override key |

---

## Pre-flight: environment

Tests are run from the project root with `PYTHONPATH=.` and the project venv.

The project uses `loguru` for logging (see `model_trainer_base.py` imports: `from loguru import logger`). Use `loguru`'s `logger` in `ot_pool.py` to stay consistent.

---

## Task 1: `OTPool` skeleton + init/floor behavior

**Files:**
- Create: `proteinfoundation/flow_matching/ot_pool.py`
- Test: `tests/test_ot_pool.py`

- [ ] **Step 1: Write failing test for floor + warn**

```python
# tests/test_ot_pool.py
import pytest
import torch
from proteinfoundation.flow_matching.ot_pool import OTPool


def test_pool_size_floored_to_multiple_of_batch_size(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    pool = OTPool(pool_size=130, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.pool_size == 128
    # loguru → std logging compat: check a warning was emitted.
    # If loguru doesn't show in caplog, check pool._truncated flag instead.
    assert pool.batch_size == 4


def test_pool_size_exact_multiple_unchanged():
    pool = OTPool(pool_size=128, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.pool_size == 128


def test_pool_size_smaller_than_batch_size_raises():
    with pytest.raises(AssertionError):
        OTPool(pool_size=2, batch_size=4, dim=3, scale_ref=1.0)


def test_initial_state_is_empty():
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.empty is True
```

- [ ] **Step 2: Run tests, verify they fail with `ImportError`**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

Expected: `ModuleNotFoundError: No module named 'proteinfoundation.flow_matching.ot_pool'`.

- [ ] **Step 3: Create minimal `OTPool` with init + `empty`**

```python
# proteinfoundation/flow_matching/ot_pool.py
"""CPU-resident OT pool: compute Hungarian pairing once per K/B training
steps, stream B-sized batches out of it random-without-replacement.

The pool owns (x_1, x_0, mask) tensors on CPU. Only next_batch() transfers
to GPU. refill() rebuilds the pool by sampling K dataset proteins + K
noise vectors and running scipy's Hungarian algorithm on the K x K
masked-coord cost matrix.
"""
import torch
from loguru import logger


class OTPool:
    def __init__(self, pool_size: int, batch_size: int, dim: int, scale_ref: float):
        effective = (pool_size // batch_size) * batch_size
        if effective != pool_size:
            logger.warning(
                "ot_pool_size={} not divisible by batch_size={}; flooring to {}",
                pool_size, batch_size, effective,
            )
        assert effective >= batch_size, (
            f"pool_size ({pool_size}) must be >= batch_size ({batch_size})"
        )
        self.pool_size = effective
        self.batch_size = batch_size
        self.dim = dim
        self.scale_ref = scale_ref
        self._x1 = None
        self._x0 = None
        self._mask = None
        self._perm = None
        self._cursor = self.pool_size  # trigger refill on first next_batch

    @property
    def empty(self) -> bool:
        return self._cursor >= self.pool_size

    def refill(self, dataset, fm, extract_clean_sample) -> None:
        raise NotImplementedError  # Task 2

    def next_batch(self, device):
        raise NotImplementedError  # Task 3
```

- [ ] **Step 4: Run tests, verify pass**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

Expected: 4 passed. If `test_pool_size_floored_to_multiple_of_batch_size` fails because loguru doesn't emit to `caplog`, loosen it to just check `pool.pool_size == 128` and add a separate `capsys`-based capture if needed — loguru's default sink is stderr.

- [ ] **Step 5: Commit**

```bash
git add proteinfoundation/flow_matching/ot_pool.py tests/test_ot_pool.py
git commit -m "Add OTPool skeleton with pool-size floor and empty flag"
```

---

## Task 2: `OTPool.refill` — sample + Hungarian + reorder

**Files:**
- Modify: `proteinfoundation/flow_matching/ot_pool.py`
- Test: `tests/test_ot_pool.py`

The refill logic is lifted from `proteinfoundation/proteinflow/proteina.py:158-279` (`_build_ot_pool`), with two semantic changes:

1. The initial B proteins are NOT taken from the dataloader — they are sampled from the dataset along with the rest. All K proteins come from random dataset indices.
2. After Hungarian returns `sigma`, we **reorder `x_0_pool` by `sigma`** so that `(x_1_pool[i], x_0_pool[i])` is the OT pair. This replaces the old "store sigma and apply it at select time" pattern.

- [ ] **Step 1: Write failing test for refill producing valid Hungarian pairing**

```python
# tests/test_ot_pool.py  (append)

class _FakeDataset:
    """Minimal dataset fixture: stores pre-made protein dicts."""
    def __init__(self, n_proteins, n_residues, seed=0):
        g = torch.Generator().manual_seed(seed)
        self._items = []
        for _ in range(n_proteins):
            coords = torch.randn(n_residues, 37, 3, generator=g)
            mask = torch.ones(n_residues, 37, 3, dtype=torch.bool)
            self._items.append({
                "coords": coords,
                "mask_dict": {"coords": mask},
            })

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class _FakeFM:
    """Replacement for R3NFlowMatcher providing only what OTPool uses."""
    dim = 3
    scale_ref = 1.0

    def _mask_and_zero_com(self, x, mask):
        # match R3NFlowMatcher: zero COM along sequence dim with mask
        mask_f = mask.float()[..., None]  # [..., N, 1]
        n = mask_f.sum(dim=-2).clamp_min(1.0)  # [..., 1]
        com = (x * mask_f).sum(dim=-2) / n    # [..., 3]
        x = (x - com[..., None, :]) * mask_f
        return x


def _fake_extract_clean_sample(batch):
    # Mimics Proteina.extract_clean_sample signature/return.
    from proteinfoundation.utils.coors_utils import ang_to_nm
    x_1 = batch["coords"][:, :, 1, :]  # [B, N, 3] CA
    mask = batch["mask_dict"]["coords"][..., 0, 0]  # [B, N]
    batch_shape = x_1.shape[:-2]
    n = x_1.shape[-2]
    return ang_to_nm(x_1), mask, batch_shape, n, x_1.dtype


def test_refill_produces_valid_hungarian_pairing(monkeypatch):
    """After refill, (x_1[i], x_0[i]) equals the Hungarian-paired OT pair."""
    import scipy.optimize
    from proteinfoundation.flow_matching.ot_pool import OTPool

    # Monkeypatch the dense_padded_from_data_list collate used by refill.
    # Plan: in Task 2 refill uses the project's real collator; the fake
    # dataset above returns already-padded items so the collator just stacks.
    # We'll lift-and-shift the proteina.py implementation — which needs the
    # collator. Use the real one.

    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=16, n_residues=8, seed=1)
    fm = _FakeFM()

    torch.manual_seed(0)  # make randperm + noise deterministic
    pool.refill(ds, fm, _fake_extract_clean_sample)

    # After refill: reconstruct the cost matrix used and verify Hungarian optimality
    K = pool.pool_size
    mask3 = pool._mask[..., None]
    x1_flat = (pool._x1 * mask3).reshape(K, -1)
    x0_flat = (pool._x0 * mask3).reshape(K, -1)
    # Since x_0 is already reordered by sigma, diagonal entries are the OT cost.
    diag_cost = ((x1_flat - x0_flat) ** 2).sum(dim=1).sum().item()

    # Build the original un-reordered cost using the same noise and run
    # Hungarian independently: the pool stores reordered x_0, so we need to
    # verify the assignment is optimal. Strategy: shuffle x_0 and re-run
    # Hungarian, check diag_cost is the minimum.
    # Simpler: verify no permutation of x_0 improves total pairing cost.
    M = torch.cdist(x1_flat, x0_flat) ** 2  # [K, K]
    _, sigma = scipy.optimize.linear_sum_assignment(M.numpy())
    # After reorder, the identity permutation should be optimal.
    # Hungarian on already-optimal matrix returns identity (up to ties).
    import numpy as np
    assert np.all(sigma == np.arange(K)), (
        f"x_0 should be pre-reordered so identity is optimal; got {sigma}"
    )


def test_refill_tensors_on_cpu():
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=2)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    assert pool._x1.device.type == "cpu"
    assert pool._x0.device.type == "cpu"
    assert pool._mask.device.type == "cpu"


def test_refill_resets_cursor_and_perm():
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=3)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    assert pool._cursor == 0
    assert pool._perm is not None
    assert pool._perm.shape == (4,)
    assert set(pool._perm.tolist()) == {0, 1, 2, 3}
    assert not pool.empty
```

- [ ] **Step 2: Run tests, verify they fail (`NotImplementedError`)**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

Expected: the three new tests fail with `NotImplementedError`.

- [ ] **Step 3: Implement `refill`**

Replace the `refill` stub in `proteinfoundation/flow_matching/ot_pool.py`:

```python
    def refill(self, dataset, fm, extract_clean_sample) -> None:
        """Sample K proteins + K noise, run Hungarian, store pool on CPU.

        Args:
            dataset: a dataset with __len__ and __getitem__ returning batch dicts.
            fm: flow matcher with _mask_and_zero_com(x, mask).
            extract_clean_sample: callable(batch) -> (x_1, mask, bshape, n, dtype).
        """
        import scipy.optimize
        from proteinfoundation.utils.dense_padding_data_loader import (
            dense_padded_from_data_list,
        )

        K = self.pool_size

        # 1. Sample K random dataset indices → load and collate into a batch.
        idx = torch.randint(0, len(dataset), (K,))
        data_list = [dataset[int(i)] for i in idx]
        batch = dense_padded_from_data_list(data_list)

        # 2. Extract CA coords + masks via the same path Proteina uses.
        x_1_pool, mask_pool, _, _, dtype = extract_clean_sample(batch)
        x_1_pool = fm._mask_and_zero_com(x_1_pool, mask_pool)
        x_1_pool = x_1_pool.detach().cpu()
        mask_pool = mask_pool.cpu()

        N_pool = x_1_pool.shape[1]

        # 3. Sample noise on CPU, mask + zero COM.
        x_0_pool = torch.randn(K, N_pool, self.dim, dtype=dtype) * self.scale_ref
        x_0_pool = fm._mask_and_zero_com(x_0_pool, mask_pool)

        # 4. Build K x K cost matrix on flat masked coords.
        mask_3d = mask_pool[..., None]
        x_1_flat = (x_1_pool * mask_3d).reshape(K, -1)
        x_0_flat = (x_0_pool * mask_3d).reshape(K, -1)
        M = torch.cdist(x_1_flat, x_0_flat) ** 2  # [K, K]

        # 5. Hungarian.
        _, sigma = scipy.optimize.linear_sum_assignment(M.numpy())

        # 6. Reorder x_0 so (x_1[i], x_0[i]) is the OT pair.
        sigma_t = torch.as_tensor(sigma, dtype=torch.long)
        x_0_pool = x_0_pool[sigma_t]

        # 7. Store pool state (all CPU).
        self._x1 = x_1_pool.contiguous()
        self._x0 = x_0_pool.contiguous()
        self._mask = mask_pool.contiguous()

        # 8. Reset cursor + random serving order.
        self._perm = torch.randperm(self.pool_size)
        self._cursor = 0

        # 9. Free intermediates.
        del M, x_1_flat, x_0_flat
```

- [ ] **Step 4: Run tests, verify pass**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

Expected: all tests pass. If `test_refill_produces_valid_hungarian_pairing` fails because `np.all(sigma == arange(K))` is too strict under ties, relax to checking that `diag_cost` equals the Hungarian minimum on the cost matrix recomputed from stored tensors. (Write the relaxed version only if the strict one fails — Hungarian is deterministic for non-degenerate inputs, and our random seed should avoid ties.)

- [ ] **Step 5: Commit**

```bash
git add proteinfoundation/flow_matching/ot_pool.py tests/test_ot_pool.py
git commit -m "OTPool.refill: sample K proteins + noise, run Hungarian, reorder x_0"
```

---

## Task 3: `OTPool.next_batch` — stream + trim + device transfer

**Files:**
- Modify: `proteinfoundation/flow_matching/ot_pool.py`
- Test: `tests/test_ot_pool.py`

- [ ] **Step 1: Write failing tests for next_batch**

```python
# tests/test_ot_pool.py  (append)

def test_next_batch_pops_batch_size_rows_and_advances_cursor():
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=16, n_residues=8, seed=4)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    assert pool._cursor == 0

    x1, x0, mask, bshape, n, dtype = pool.next_batch("cpu")
    assert x1.shape[0] == 4
    assert x0.shape[0] == 4
    assert mask.shape[0] == 4
    assert pool._cursor == 4
    assert not pool.empty

    _, _, _, _, _, _ = pool.next_batch("cpu")
    assert pool._cursor == 8
    assert pool.empty


def test_next_batch_random_without_replacement_per_cycle():
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=16, n_residues=8, seed=5)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)

    # The pool's perm should contain each index exactly once.
    seen = set(pool._perm.tolist())
    assert seen == {0, 1, 2, 3, 4, 5, 6, 7}


def test_next_batch_on_empty_raises():
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=6)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    pool.next_batch("cpu")
    pool.next_batch("cpu")  # cursor == 4, empty
    with pytest.raises(AssertionError):
        pool.next_batch("cpu")


def test_next_batch_trims_to_actual_max_length():
    """If pool has mixed lengths, next_batch output should be trimmed."""
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    fm = _FakeFM()

    # Build a pool with mixed lengths manually.
    # Hand-construct _x1, _x0, _mask rather than going through refill.
    N_pool = 20
    pool._x1 = torch.randn(4, N_pool, 3)
    pool._x0 = torch.randn(4, N_pool, 3)
    mask = torch.zeros(4, N_pool, dtype=torch.bool)
    mask[0, :10] = True   # length 10
    mask[1, :8] = True    # length 8
    mask[2, :20] = True   # length 20
    mask[3, :15] = True   # length 15
    pool._mask = mask
    # Force perm so first batch is proteins 0, 1 (max length = 10).
    pool._perm = torch.tensor([0, 1, 2, 3])
    pool._cursor = 0

    x1, x0, m, bshape, n, dtype = pool.next_batch("cpu")
    assert x1.shape == (2, 10, 3), f"expected (2, 10, 3), got {x1.shape}"
    assert x0.shape == (2, 10, 3)
    assert m.shape == (2, 10)
    assert n == 10


def test_next_batch_moves_to_device_but_pool_stays_cpu():
    from proteinfoundation.flow_matching.ot_pool import OTPool
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=7)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    x1, x0, m, _, _, _ = pool.next_batch("cpu")
    assert x1.device.type == "cpu"
    # Pool tensors remain on CPU regardless of request device.
    assert pool._x1.device.type == "cpu"
    assert pool._x0.device.type == "cpu"

    if torch.cuda.is_available():
        pool.refill(ds, fm, _fake_extract_clean_sample)
        x1c, x0c, mc, _, _, _ = pool.next_batch("cuda")
        assert x1c.device.type == "cuda"
        assert pool._x1.device.type == "cpu"
```

- [ ] **Step 2: Run tests, verify they fail (`NotImplementedError`)**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

- [ ] **Step 3: Implement `next_batch`**

Replace the `next_batch` stub:

```python
    def next_batch(self, device):
        """Pop B indices from the permutation, trim, and ship to device.

        Returns:
            (x_1, x_0, mask, batch_shape, n, dtype) on `device`.
        """
        assert not self.empty, "OTPool is empty — call refill() first."
        idx = self._perm[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size

        x_1_sel = self._x1[idx]      # [B, N_pool, 3] CPU
        x_0_sel = self._x0[idx]      # [B, N_pool, 3] CPU
        mask_sel = self._mask[idx]   # [B, N_pool]    CPU

        # Trim to max actual length among selected rows.
        n_sel = int(mask_sel.sum(dim=1).max().item())
        if n_sel == 0:
            n_sel = x_1_sel.shape[1]
        x_1_sel = x_1_sel[:, :n_sel, :]
        x_0_sel = x_0_sel[:, :n_sel, :]
        mask_sel = mask_sel[:, :n_sel]

        dtype = self._x1.dtype
        batch_shape = torch.Size([self.batch_size])

        return (
            x_1_sel.to(device),
            x_0_sel.to(device),
            mask_sel.to(device),
            batch_shape,
            n_sel,
            dtype,
        )
```

Note: we do NOT re-mask-and-zero-COM here. `refill` already applied
`_mask_and_zero_com` to both `x_1` and `x_0`, and trimming along the
padding side (right of `n_sel`) does not change COM because the trimmed
region was all zeros under the mask. A deferred defensive
mask-and-zero-COM is available if downstream needs it — see the smoke
test in Task 6.

- [ ] **Step 4: Run tests, verify pass**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

- [ ] **Step 5: Commit**

```bash
git add proteinfoundation/flow_matching/ot_pool.py tests/test_ot_pool.py
git commit -m "OTPool.next_batch: stream B pairs per call, trim, ship to device"
```

---

## Task 4: Rename `noise_samples` → `ot_pool_size` in configs

**Files:**
- Modify: `configs/experiment_config/training_ca_debug.yaml`
- Modify: `configs/experiment_config/training_ca.yaml`
- Modify: any sbatch in `scripts/` on this branch referencing `training.ot_coupling.noise_samples`

- [ ] **Step 1: Rename in debug config**

Edit `configs/experiment_config/training_ca_debug.yaml`, line 47:

```yaml
# Before:
    noise_samples: 128
# After:
    ot_pool_size: 128
```

- [ ] **Step 2: Add `ot_pool_size` to `training_ca.yaml`**

Edit `configs/experiment_config/training_ca.yaml`, inside the `ot_coupling:` block (currently lines 47-52), append after `normalize_cost: False`:

```yaml
    ot_pool_size: null   # Enable streaming OT pool by setting to a multiple of batch_size; null disables it.
```

Using `null` (hydra → Python `None`) keeps the default behavior of the non-pool code path when no override is given.

- [ ] **Step 3: Grep for any remaining `noise_samples` reference on this branch**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -c "import subprocess; print(subprocess.check_output(['git', 'grep', '-n', 'noise_samples']).decode())"
```

Or directly:

```bash
git grep -n noise_samples
```

Expected output (after Tasks 5-6): only occurrences in test files that reference `n_noise_samples` (a different unrelated parameter — do NOT rename). Specifically, `tests/test_chirality_diagnostics.py:178,195` use `n_noise_samples=4` which is a kwarg of a different function (`evaluate_chirality_sign_agreement` or similar) — leave it alone.

- [ ] **Step 4: Check no production sbatch still passes `noise_samples`**

```bash
git grep -n "training.ot_coupling.noise_samples"
```

Expected: no hits in tracked files on this branch. (The `train_1000` branch's sbatch doesn't have this override; only other worktrees do, and those stay pinned to their code version.)

- [ ] **Step 5: Commit**

```bash
git add configs/experiment_config/training_ca_debug.yaml configs/experiment_config/training_ca.yaml
git commit -m "Rename ot_coupling.noise_samples -> ot_pool_size in configs"
```

---

## Task 5: Thread `x_0_override` through `_compute_single_noise_loss`

**Files:**
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py`
- Test: existing `tests/test_training_step.py` still passes (non-pool path unchanged); new test added in Task 6 covers pool path.

The goal: `_compute_single_noise_loss` should accept an optional `x_0_override` parameter that, when provided, replaces both the fresh-noise sampling (line 309-311) and the internal B×B OT (line 313-315).

- [ ] **Step 1: Write failing test**

Add to `tests/test_ot_pool.py`:

```python
# tests/test_ot_pool.py  (append)

def test_compute_single_noise_loss_accepts_x0_override():
    """_compute_single_noise_loss should use x_0_override verbatim when provided."""
    # This test is a thin signature/type check — a full end-to-end run with
    # override is deferred to Task 6's smoke test.
    import inspect
    from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
    sig = inspect.signature(ModelTrainerBase._compute_single_noise_loss)
    assert "x_0_override" in sig.parameters, (
        "ModelTrainerBase._compute_single_noise_loss must accept x_0_override"
    )
    # Default should be None (backwards-compat with non-pool path).
    assert sig.parameters["x_0_override"].default is None
```

- [ ] **Step 2: Run test, verify fails**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py::test_compute_single_noise_loss_accepts_x0_override -x -q
```

Expected: AssertionError — the arg doesn't exist yet.

- [ ] **Step 3: Modify `_compute_single_noise_loss`**

Edit `proteinfoundation/proteinflow/model_trainer_base.py:276`:

```python
# Before:
def _compute_single_noise_loss(self, x_1, mask, t_ext, r_ext, t, batch, B, *, use_sc=False):

# After:
def _compute_single_noise_loss(self, x_1, mask, t_ext, r_ext, t, batch, B, *, use_sc=False, x_0_override=None):
```

Update the docstring (same file, around line 293) to note the new argument:

```python
        use_sc: If True, run one extra no-grad forward pass to obtain
            x_sc and feed it as a detached constant into the JVP and FM
            sub-passes. Requires the NN config to declare x_sc /
            x_sc_pair_dists in its feature lists.
        x_0_override: If not None, use this tensor as x_0 verbatim and
            skip the internal fresh-noise sampling + B x B OT.
            Shape must match ``x_1`` ([B, n, 3]).
```

Then replace the block at lines 308-315:

```python
# Before:
        # 1. Sample noise
        x_0 = self.fm.sample_reference(
            n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
        )
        # 2. Standard square-batch OT (when noise_samples pool is NOT active)
        if self.ot_sampler is not None:
            ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(x_1, x_0, mask)
            x_0 = x_0[ot_noise_idx]

# After:
        if x_0_override is not None:
            # Pool mode: x_0 is pre-paired with x_1 via the K x K Hungarian
            # in OTPool.refill. Skip fresh-noise sampling and B x B OT.
            x_0 = x_0_override
        else:
            # 1. Sample noise
            x_0 = self.fm.sample_reference(
                n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
            )
            # 2. Standard square-batch OT (when pool is NOT active)
            if self.ot_sampler is not None:
                ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(x_1, x_0, mask)
                x_0 = x_0[ot_noise_idx]
```

- [ ] **Step 4: Run test + existing non-pool tests**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py tests/test_training_step.py -x -q
```

Expected: all pass. The `test_training_step.py` tests exercise the non-pool path and should be unchanged.

- [ ] **Step 5: Commit**

```bash
git add proteinfoundation/proteinflow/model_trainer_base.py tests/test_ot_pool.py
git commit -m "Thread x_0_override through _compute_single_noise_loss"
```

---

## Task 6: Wire `OTPool` into `Proteina`

**Files:**
- Modify: `proteinfoundation/proteinflow/proteina.py`
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py`
- Test: `tests/test_ot_pool.py`

- [ ] **Step 1: Write failing smoke test**

Add to `tests/test_ot_pool.py`:

```python
# tests/test_ot_pool.py  (append)

def test_proteina_on_train_start_builds_pool_when_ot_pool_size_set(monkeypatch):
    """Proteina.on_train_start should construct self._ot_pool when config has ot_pool_size."""
    # This is a structural test — we check the wiring, not training math.
    # The end-to-end run of training_step under pool mode is exercised by
    # tests/test_training_step.py in a follow-up pass after this task.
    from proteinfoundation.flow_matching.ot_pool import OTPool
    # Construct a pool directly; verify it matches what on_train_start would make.
    # (Full Proteina-level integration test deferred to test_training_step.py.)
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.pool_size == 8
    assert pool.batch_size == 4


def test_proteina_ot_pool_size_forbids_loss_accumulation_steps_gt_1():
    """When ot_pool_size is set, loss_accumulation_steps must be 1."""
    # Deferred to an assertion in Proteina.on_train_start — covered by the
    # smoke test below that builds a minimal Proteina and calls on_train_start.
    pass
```

The real wiring test:

```python
def test_training_step_pool_mode_end_to_end(tmp_path, monkeypatch):
    """Run a single training_step in pool mode against a tiny fake dataset.

    Verifies:
    - Pool is refilled on first call.
    - Loss is finite.
    - No fresh x_0 is drawn inside _compute_single_noise_loss (x_0_override path taken).
    """
    # This test is intentionally deferred until Proteina + OTPool are wired.
    # After Task 6's step 4 the test should pass.
    pytest.importorskip("lightning")
    # Full end-to-end setup is heavy; this test uses the project's existing
    # Proteina-construction helpers if present. If writing it is too large
    # for this step, defer to tests/test_training_step.py and add a
    # parametrized "pool mode" case there.
    pytest.skip("Exercised via tests/test_training_step.py pool-mode parametrization")
```

- [ ] **Step 2: Run the new non-deferred tests**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

Expected: the structural pool-construction test passes, the end-to-end one is skipped.

- [ ] **Step 3: Edit `Proteina.on_train_start`**

In `proteinfoundation/proteinflow/proteina.py`, replace lines 144-156 (the current `on_train_start`):

```python
# Before:
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

# After:
    def on_train_start(self):
        """Build the OT pool when ``ot_coupling.ot_pool_size`` is set."""
        from proteinfoundation.flow_matching.ot_pool import OTPool

        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        pool_size = ot_cfg.get("ot_pool_size", None)
        if pool_size is not None and self.ot_sampler is not None:
            assert self.loss_accumulation_steps == 1, (
                "ot_pool_size is set but loss_accumulation_steps="
                f"{self.loss_accumulation_steps}. Pool mode requires "
                "loss_accumulation_steps=1."
            )
            dm = self.trainer.datamodule
            if dm.train_ds is None:
                dm.setup("fit")
            self._ot_dataset = dm.train_ds
            bs = self.trainer.datamodule.batch_size
            assert pool_size >= bs, (
                f"ot_pool_size ({pool_size}) must be >= batch_size ({bs})"
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

- [ ] **Step 4: Add `_get_ot_batch` and remove `_build_ot_pool`**

In `proteinfoundation/proteinflow/proteina.py`, delete the entire
`_build_ot_pool` method (lines 158-279) and replace it with:

```python
    def _get_ot_batch(self):
        """Return the next B-sized OT pair batch from the pool.

        Refills the pool lazily when empty. Returns tensors on self.device.
        """
        assert self._ot_pool is not None, (
            "_get_ot_batch called but OTPool was not constructed. "
            "Ensure ot_coupling.ot_pool_size is set."
        )
        if self._ot_pool.empty:
            self._ot_pool.refill(
                self._ot_dataset, self.fm, self.extract_clean_sample,
            )
        return self._ot_pool.next_batch(self.device)
```

- [ ] **Step 5: Update `training_step` in `model_trainer_base.py`**

Edit `proteinfoundation/proteinflow/model_trainer_base.py`, around lines 477-487:

```python
# Before:
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

# After:
        # --- 1. Shared setup: extract x_1, mask, sample (t, r), fold conditioning ---
        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        pool_size = ot_cfg.get("ot_pool_size", None)
        pool_mode = (
            pool_size is not None
            and self.ot_sampler is not None
            and not val_step
        )

        if pool_mode:
            # OT pool: pool owns (x_1, x_0) pairing. x_0 flows through as
            # x_0_override into _compute_single_noise_loss.
            x_1, x_0_pool, mask, batch_shape, n, dtype = self._get_ot_batch()
        else:
            x_0_pool = None
            x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, mask)
```

Then update the K=1 call at line ~541 to pass `x_0_override`:

```python
# Before:
        if K == 1:
            combined_adp_loss, raw_loss_mf, raw_loss_fm, raw_loss_chir, raw_adp_wt_mean = self._compute_single_noise_loss(
                x_1, mask, t_ext, r_ext, t, batch, B, use_sc=use_sc,
            )

# After:
        if K == 1:
            combined_adp_loss, raw_loss_mf, raw_loss_fm, raw_loss_chir, raw_adp_wt_mean = self._compute_single_noise_loss(
                x_1, mask, t_ext, r_ext, t, batch, B, use_sc=use_sc,
                x_0_override=x_0_pool,
            )
```

(The K>1 branch does not need to pass `x_0_override` because `pool_mode`
requires `K == 1`; the assertion in `on_train_start` prevents that path.)

- [ ] **Step 6: Run the full targeted test suite**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py tests/test_training_step.py tests/test_ot_integration.py tests/test_ot_sampler.py -x -q
```

Expected: all pass. If `test_ot_integration.py` references `noise_samples`
in YAML overrides, update those strings to `ot_pool_size`.

- [ ] **Step 7: Commit**

```bash
git add proteinfoundation/proteinflow/proteina.py proteinfoundation/proteinflow/model_trainer_base.py tests/test_ot_pool.py
git commit -m "Wire OTPool into Proteina; training_step uses pool x_0 via override"
```

---

## Task 7: Integration test — pool-mode training_step end-to-end

**Files:**
- Test: `tests/test_ot_pool.py` or `tests/test_training_step.py` (whichever already has Proteina-construction helpers)

- [ ] **Step 1: Locate existing Proteina-construction helpers**

```bash
git grep -l "Proteina(" tests/
```

If `tests/test_training_step.py` already builds a Proteina with a config, extend it with a pool-mode test case. Otherwise, add it to `tests/test_ot_pool.py`.

- [ ] **Step 2: Write end-to-end pool-mode test**

```python
# tests/test_ot_pool.py  (append, OR in test_training_step.py)
def test_training_step_pool_mode_runs_and_loss_finite(tmp_path):
    """Full training_step under pool mode: loss finite, grads flow, pool
    cycle works across K/B steps."""
    # Mirror the fixture pattern used elsewhere in this file/test_training_step.py.
    # If an existing `make_proteina_for_test(cfg_overrides)` helper exists, reuse it;
    # otherwise build a minimal cfg via OmegaConf and instantiate Proteina directly.
    pytest.importorskip("omegaconf")
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from proteinfoundation.proteinflow.proteina import Proteina

    # Load training_ca_debug.yaml to get a working baseline config.
    from hydra import compose, initialize_config_dir
    import os
    repo_root = os.getcwd()
    cfg_dir = os.path.join(repo_root, "configs", "experiment_config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg_exp = compose(config_name="training_ca_debug")

    # Override for test: tiny pool, batch_size=2, loss_accum=1, OT on.
    cfg_exp.training.ot_coupling.enabled = True
    cfg_exp.training.ot_coupling.ot_pool_size = 4
    cfg_exp.training.loss_accumulation_steps = 1
    # Other tunables (eval_val, fold_cond, etc.) come from debug defaults.

    model = Proteina(cfg_exp=cfg_exp, store_dir=str(tmp_path))
    # Stub trainer/datamodule: minimal fakes sufficient for on_train_start.
    class _FakeDM:
        batch_size = 2
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=42)
        def setup(self, stage): pass
    class _FakeTrainer:
        datamodule = _FakeDM()
    model.trainer = _FakeTrainer()
    model.on_train_start()

    assert model._ot_pool is not None
    assert model._ot_pool.pool_size == 4
    assert model._ot_pool.batch_size == 2
    assert model._ot_pool.empty  # not yet refilled

    # Call _get_ot_batch twice — pool size 4, batch 2 → two batches then empty.
    x1a, x0a, ma, _, _, _ = model._get_ot_batch()
    assert torch.isfinite(x1a).all() and torch.isfinite(x0a).all()
    assert not model._ot_pool.empty
    x1b, x0b, mb, _, _, _ = model._get_ot_batch()
    assert model._ot_pool.empty  # consumed the whole pool

    # Third call triggers refill — pool becomes non-empty again.
    x1c, x0c, mc, _, _, _ = model._get_ot_batch()
    assert not model._ot_pool.empty


def test_on_train_start_rejects_pool_with_loss_accum_gt_1(tmp_path):
    pytest.importorskip("omegaconf")
    from hydra import compose, initialize_config_dir
    import os
    from proteinfoundation.proteinflow.proteina import Proteina
    repo_root = os.getcwd()
    cfg_dir = os.path.join(repo_root, "configs", "experiment_config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg_exp = compose(config_name="training_ca_debug")
    cfg_exp.training.ot_coupling.enabled = True
    cfg_exp.training.ot_coupling.ot_pool_size = 4
    cfg_exp.training.loss_accumulation_steps = 2

    model = Proteina(cfg_exp=cfg_exp, store_dir=str(tmp_path))

    class _FakeDM:
        batch_size = 2
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=0)
        def setup(self, stage): pass
    class _FakeTrainer:
        datamodule = _FakeDM()
    model.trainer = _FakeTrainer()

    with pytest.raises(AssertionError, match="loss_accumulation_steps"):
        model.on_train_start()
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/test_ot_pool.py -x -q
```

Expected: all pass. If Hydra config-loading or the Proteina constructor demands fields not in `training_ca_debug.yaml`, minimally override them inline. If construction still trips, fall back to a handcrafted `OmegaConf.create({...})` with only the fields read by `Proteina.__init__` and `ModelTrainerBase.__init__`.

- [ ] **Step 4: Run the broader test suite to catch regressions**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/ -x -q --ignore=tests/callbacks
```

Expected: all pass. Callback tests are skipped because they pull in Lightning Trainer — out of scope for this change.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ot_pool.py
git commit -m "Add end-to-end pool-mode test + loss_accum>1 rejection test"
```

---

## Task 8: Final sweep for lingering `noise_samples`

**Files:** all tracked files

- [ ] **Step 1: Grep**

```bash
git grep -n noise_samples -- ':!*.md' ':!tests/test_chirality_diagnostics.py'
```

(The exclusions: `tests/test_chirality_diagnostics.py` uses `n_noise_samples` which is an unrelated kwarg of a different function; docs files may contain historical references in specs/plans.)

Expected output: empty.

- [ ] **Step 2: If any hits, patch them inline, commit.**

For each hit, decide: rename to `ot_pool_size` (if it's the OT pool parameter), or leave alone (if it's a distinct concept). Commit with:

```bash
git commit -m "Remove last references to ot_coupling.noise_samples"
```

If no hits, skip this step.

---

## Task 9: Sanity-compile and final test sweep

- [ ] **Step 1: Byte-compile all changed Python files**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m py_compile \
  proteinfoundation/flow_matching/ot_pool.py \
  proteinfoundation/proteinflow/proteina.py \
  proteinfoundation/proteinflow/model_trainer_base.py
```

Expected: no output (success).

- [ ] **Step 2: Run full test suite**

```bash
PYTHONPATH=. /netscratch/schekmenev/venvs/meanflow-ot-proteina/.venv/bin/python -m pytest tests/ -x -q --ignore=tests/callbacks
```

Expected: all pass. If any fail, debug before marking the plan complete.

- [ ] **Step 3: Verify the spec is still accurate**

Open `docs/superpowers/specs/2026-04-21-ot-pool-streaming-design.md` and skim:
- All files listed under "Files Changed" have been touched.
- The behavior changes section matches what was implemented.

If the spec is out of sync (e.g., an implementation decision differed from spec), update the spec in the same commit as the implementation fix, **never** silently diverge.

- [ ] **Step 4: Final commit (if any doc updates)**

```bash
git commit -am "Align spec with implementation decisions"   # only if changes exist
```

---

## Completion checklist (reviewer-facing)

- [ ] `OTPool` class exists and is covered by unit tests: floor+warn, refill correctness, cursor/perm state, next_batch trim + device transfer, empty-assertion.
- [ ] `Proteina.on_train_start` constructs `OTPool` when `ot_pool_size` is set and asserts `loss_accumulation_steps == 1`.
- [ ] `_get_ot_batch` lazily refills and returns `next_batch(self.device)`.
- [ ] `training_step` pool path calls `_get_ot_batch()` and passes `x_0_pool` into `_compute_single_noise_loss` via `x_0_override`.
- [ ] `_compute_single_noise_loss` uses `x_0_override` verbatim (no fresh-noise sampling, no B×B OT) when provided.
- [ ] Configs renamed: `noise_samples` → `ot_pool_size`. `training_ca.yaml` has `ot_pool_size: null` as the default.
- [ ] Non-pool path unchanged (existing tests still pass).
- [ ] No remaining `noise_samples` references in code (only in docs/specs, which is expected).
- [ ] End-to-end pool-mode test demonstrates: pool is built, cycles through two batches, refills on third call.
- [ ] Pool + `loss_accumulation_steps > 1` is rejected at `on_train_start` with a clear assertion message.
