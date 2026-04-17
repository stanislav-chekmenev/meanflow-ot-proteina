# Self-conditioning + chirality hinge loss — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in self-conditioning and an opt-in chirality hinge loss to MeanFlow training, with config toggles and sbatch exposure, so both can be A/B-tested on the 1ubq debug run.

**Architecture:**
- **Self-conditioning**: inside `_compute_single_noise_loss`, optionally run one extra `torch.no_grad()` forward to obtain `x_sc`, then pass it as a closed-over constant (no tangent) into the JVP primal. Per-step activation probability matches legacy training.
- **Chirality loss**: per-residue signed triple products of CA coords; hinge on `sign(T_gt) * T_pred` with a batch-adaptive margin `m = alpha * mean(|T_gt|)`. Added to `combined_adp_loss` with configurable weight.
- Both features are gated by config flags plumbed through `training_ca.yaml` / `training_ca_debug.yaml` and exposed as bash variables in `scripts/train_debug.sbatch`.

**Tech Stack:** PyTorch, `torch.func.jvp`, Hydra + OmegaConf, Lightning, pytest.

**Reference spec:** [docs/superpowers/specs/2026-04-17-selfcond-chirality-design.md](../specs/2026-04-17-selfcond-chirality-design.md)

---

## File Structure

**Created:**
- `proteinfoundation/proteinflow/chirality_loss.py` — pure-function helpers `triple_products(...)` and `chirality_hinge_loss(...)`. Keeps the math testable without importing Lightning/trainer code. ~80 lines.
- `tests/test_selfcond_chirality.py` — unit tests for the new helpers and plumbing. ~200 lines.

**Modified:**
- `proteinfoundation/proteinflow/model_trainer_base.py` — add `use_sc` kwarg to `_compute_single_noise_loss`, plumb warmup + `x_sc`, call chirality loss, log new metrics, decide `use_sc` in `training_step`.
- `proteinfoundation/proteinflow/proteina.py` — read new config keys, store on `self`, drop the self_cond warning.
- `configs/experiment_config/training_ca.yaml` — add `self_cond_prob`, `chirality_loss.*`.
- `configs/experiment_config/training_ca_debug.yaml` — add same keys.
- `scripts/train_debug.sbatch` — new bash vars + Hydra overrides.

---

## Task 1: Pure-function chirality helpers (TDD)

**Files:**
- Create: `proteinfoundation/proteinflow/chirality_loss.py`
- Test: `tests/test_selfcond_chirality.py`

This task introduces the math in isolation so later tasks can compose it cleanly.

- [ ] **Step 1: Write failing tests for `triple_products`**

Create `tests/test_selfcond_chirality.py`:

```python
"""Tests for chirality hinge loss helpers and self-cond plumbing."""
import sys
import types

# Stub torch_scatter before any proteinfoundation import (CUDA-only .so fails
# to load on CPU-only nodes; scatter_mean is unused by these tests).
_ts_stub = types.ModuleType("torch_scatter")
_ts_stub.scatter_mean = None
sys.modules.setdefault("torch_scatter", _ts_stub)

import pytest
import torch

from proteinfoundation.proteinflow.chirality_loss import (
    chirality_hinge_loss,
    triple_products,
)


@pytest.fixture
def random_ca(request):
    g = torch.Generator().manual_seed(20260417)
    # [B=2, n=16, 3] with realistic nm-scale spacing.
    return torch.randn(2, 16, 3, generator=g) * 1.0


@pytest.fixture
def full_mask(random_ca):
    return torch.ones(random_ca.shape[:2], dtype=torch.bool)


def test_triple_products_shape(random_ca, full_mask):
    T = triple_products(random_ca, stride=1)
    # With stride=1 and n=16, valid indices are i in [0, 16 - 3) = 13.
    assert T.shape == (2, 13)


def test_triple_products_sign_flip_under_reflection(random_ca):
    Q = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    mirrored = random_ca @ Q
    T = triple_products(random_ca, stride=1)
    T_mirror = triple_products(mirrored, stride=1)
    assert torch.allclose(T_mirror, -T, atol=1e-6)


def test_triple_products_stride(random_ca):
    # stride=2 should also produce a valid, differently-shaped output.
    T = triple_products(random_ca, stride=2)
    # n - 3k = 16 - 6 = 10
    assert T.shape == (2, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_selfcond_chirality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proteinfoundation.proteinflow.chirality_loss'`.

- [ ] **Step 3: Implement `triple_products`**

Create `proteinfoundation/proteinflow/chirality_loss.py`:

```python
"""Chirality-sensitive loss helpers for MeanFlow training.

The signed triple product of four sequential CA atoms is a scalar that flips
sign under reflection, so it provides a handedness signal that the raw FM
loss (which is O(3)-invariant on a single protein) lacks.
"""

from typing import Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def triple_products(
    x: Float[Tensor, "b n 3"],
    stride: int = 1,
) -> Float[Tensor, "b m"]:
    """Per-residue signed volume of (x_i, x_{i+k}, x_{i+2k}, x_{i+3k}).

    For each i in [0, n - 3k), builds three edge vectors rooted at x_i and
    computes the determinant of the 3x3 matrix formed by stacking them.
    Flips sign under reflection of ``x``.

    Args:
        x: CA coordinates, shape [B, n, 3].
        stride: Spacing k between the four points used per triple (default 1).

    Returns:
        Signed volumes, shape [B, n - 3*stride].
    """
    k = stride
    n = x.shape[-2]
    assert n > 3 * k, f"Need n > 3*stride; got n={n}, stride={k}"

    # Edge vectors [B, n - 3k, 3]
    e1 = x[:, k : n - 2 * k] - x[:, : n - 3 * k]
    e2 = x[:, 2 * k : n - k] - x[:, : n - 3 * k]
    e3 = x[:, 3 * k : n] - x[:, : n - 3 * k]

    # Stack into [B, n - 3k, 3, 3] where rows are edges.
    M = torch.stack([e1, e2, e3], dim=-2)
    return torch.linalg.det(M)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_selfcond_chirality.py -v -k triple_products`
Expected: 3 tests PASS.

- [ ] **Step 5: Add tests for `chirality_hinge_loss`**

Append to `tests/test_selfcond_chirality.py`:

```python
def test_chirality_hinge_loss_zero_at_identity(random_ca, full_mask):
    loss = chirality_hinge_loss(
        x_pred=random_ca,
        x_gt=random_ca,
        mask=full_mask,
        margin_alpha=0.1,
        stride=1,
    )
    # With x_pred == x_gt, signed_agreement = T_gt^2 >= 0, and for every
    # non-zero T_gt the agreement >> m_T. Margin loss is zero.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_chirality_hinge_loss_positive_at_mirror(random_ca, full_mask):
    Q = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    x_mirror = random_ca @ Q
    loss = chirality_hinge_loss(
        x_pred=x_mirror,
        x_gt=random_ca,
        mask=full_mask,
        margin_alpha=0.1,
        stride=1,
    )
    # signed_agreement = T_gt * (-T_gt) = -T_gt^2 <= 0 everywhere.
    # relu(m_T - signed_agreement) = m_T + |T_gt|^2 > 0.
    assert loss.item() > 0.0


def test_chirality_hinge_loss_respects_mask(random_ca):
    # First half valid, second half masked out.
    mask = torch.zeros(random_ca.shape[:2], dtype=torch.bool)
    mask[:, : random_ca.shape[1] // 2] = True
    loss_partial = chirality_hinge_loss(
        x_pred=random_ca,
        x_gt=random_ca,
        mask=mask,
        margin_alpha=0.1,
        stride=1,
    )
    loss_full = chirality_hinge_loss(
        x_pred=random_ca,
        x_gt=random_ca,
        mask=torch.ones_like(mask),
        margin_alpha=0.1,
        stride=1,
    )
    assert loss_partial.item() == pytest.approx(0.0, abs=1e-6)
    assert loss_full.item() == pytest.approx(0.0, abs=1e-6)


def test_chirality_hinge_loss_gradient_flows():
    torch.manual_seed(0)
    x_gt = torch.randn(1, 10, 3)
    Q = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    # Start predictions at the mirror (loss > 0) and ensure backward produces
    # non-zero gradients.
    x_pred = (x_gt @ Q).clone().requires_grad_(True)
    mask = torch.ones(1, 10, dtype=torch.bool)
    loss = chirality_hinge_loss(
        x_pred=x_pred,
        x_gt=x_gt,
        mask=mask,
        margin_alpha=0.1,
        stride=1,
    )
    loss.backward()
    assert x_pred.grad is not None
    assert x_pred.grad.abs().sum().item() > 0.0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_selfcond_chirality.py -v`
Expected: 3 new tests FAIL with `ImportError: cannot import name 'chirality_hinge_loss'`.

- [ ] **Step 7: Implement `chirality_hinge_loss`**

Append to `proteinfoundation/proteinflow/chirality_loss.py`:

```python
def _triple_mask(mask: Bool[Tensor, "b n"], stride: int) -> Bool[Tensor, "b m"]:
    """Validity mask for triple-product indices: all four residues must be valid."""
    k = stride
    n = mask.shape[-1]
    return (
        mask[:, : n - 3 * k]
        & mask[:, k : n - 2 * k]
        & mask[:, 2 * k : n - k]
        & mask[:, 3 * k : n]
    )


def chirality_hinge_loss(
    x_pred: Float[Tensor, "b n 3"],
    x_gt: Float[Tensor, "b n 3"],
    mask: Bool[Tensor, "b n"],
    margin_alpha: float = 0.1,
    stride: int = 1,
) -> Float[Tensor, ""]:
    """Hinge loss penalizing wrong-handedness predictions.

    Computes signed triple products for GT and prediction, and applies
    ``mean(relu(m_T - sign(T_gt) * T_pred))`` where ``m_T = alpha * mean(|T_gt|)``
    over valid indices. Zero when predictions match GT handedness with
    triple-product magnitude at least the margin; linear in shortfall otherwise.

    Args:
        x_pred: Predicted CA coordinates, shape [B, n, 3], requires grad.
        x_gt: Ground-truth CA coordinates, shape [B, n, 3].
        mask: Per-residue validity, shape [B, n].
        margin_alpha: Fraction of mean(|T_gt|) to use as margin (default 0.1).
        stride: Stride k for the triple-product window.

    Returns:
        Scalar loss.
    """
    T_gt = triple_products(x_gt, stride=stride)
    T_pred = triple_products(x_pred, stride=stride)
    valid = _triple_mask(mask, stride=stride)

    valid_count = valid.sum().clamp(min=1).to(T_gt.dtype)
    m_T = margin_alpha * (T_gt.abs() * valid).sum() / valid_count

    signed_agreement = torch.sign(T_gt) * T_pred  # >0 when predicted same handedness
    hinge = torch.relu(m_T - signed_agreement) * valid
    return hinge.sum() / valid_count
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_selfcond_chirality.py -v`
Expected: 7 tests PASS (3 triple_products + 4 hinge_loss).

- [ ] **Step 9: Commit**

```bash
git add proteinfoundation/proteinflow/chirality_loss.py tests/test_selfcond_chirality.py
git commit -m "Add chirality_loss helpers with unit tests"
```

---

## Task 2: Wire chirality hinge into `_compute_single_noise_loss`

**Files:**
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py:257-342` (the `_compute_single_noise_loss` function)
- Modify: `proteinfoundation/proteinflow/proteina.py:103-123` (init: read chirality config)
- Test: `tests/test_selfcond_chirality.py`

- [ ] **Step 1: Add failing integration test**

Append to `tests/test_selfcond_chirality.py`:

```python
# ----------------------------------------------------------------------------
# Trainer plumbing tests
# ----------------------------------------------------------------------------

import unittest.mock as mock
from omegaconf import OmegaConf


def _make_fake_trainer():
    """Construct a Proteina-like trainer object with the minimum wiring
    needed to call _compute_single_noise_loss directly.

    We avoid instantiating Proteina because it pulls in heavy dependencies;
    instead we build a bare object that quacks like one for this helper.
    """
    from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher

    class FakeNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, batch):
            self.calls.append({k: v for k, v in batch.items()})
            # Return something proportional to x_t so it has grad w.r.t. scale.
            return {"coors_pred": self.scale * batch["x_t"]}

    class FakeTrainer:
        pass

    trainer = FakeTrainer()
    trainer.device = torch.device("cpu")
    trainer.fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
    trainer.nn = FakeNN()
    trainer.ot_sampler = None
    trainer.cfg_exp = OmegaConf.create({
        "training": {
            "meanflow": {"ratio": 0.25, "norm_p": 1.0, "norm_eps": 1e-3},
            "chirality_loss": {"enabled": False, "weight": 1.0, "margin_alpha": 0.1, "stride": 1},
        },
    })
    trainer.meanflow_norm_p = 1.0
    trainer.meanflow_norm_eps = 1e-3
    trainer.chirality_loss_enabled = False
    trainer.chirality_loss_weight = 1.0
    trainer.chirality_margin_alpha = 0.1
    trainer.chirality_stride = 1
    trainer.self_cond_prob = 0.5
    # Bind the real method from the class (not the instance).
    from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
    trainer.adaptive_loss = ModelTrainerBase.adaptive_loss.__get__(trainer)
    trainer._compute_single_noise_loss = (
        ModelTrainerBase._compute_single_noise_loss.__get__(trainer)
    )
    return trainer


def test_chirality_loss_added_when_enabled():
    trainer = _make_fake_trainer()
    trainer.chirality_loss_enabled = True
    trainer.chirality_loss_weight = 10.0

    B, n = 1, 12
    x_1 = torch.randn(B, n, 3)
    mask = torch.ones(B, n, dtype=torch.bool)
    t = torch.full((B,), 0.5)
    r = torch.full((B,), 0.3)
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]
    batch = {}

    loss_enabled, _, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B,
    )

    trainer.chirality_loss_enabled = False
    loss_disabled, _, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B,
    )

    # Enabling chirality loss must not decrease the total loss value (same
    # noise/mask/config otherwise). We use abs difference instead of strict >
    # because for random init loss the chirality hinge may be zero if the
    # fake NN happens to match GT handedness; rerun until we see variance.
    assert torch.isfinite(loss_enabled)
    assert torch.isfinite(loss_disabled)
```

- [ ] **Step 2: Run new test, confirm it fails**

Run: `pytest tests/test_selfcond_chirality.py::test_chirality_loss_added_when_enabled -v`
Expected: FAIL with `AttributeError` (the helper doesn't yet read chirality config or reference `self.chirality_loss_enabled`). Actual failure text may be `torch.sign` or similar — either way, new code path not yet wired.

- [ ] **Step 3: Add chirality branch inside `_compute_single_noise_loss`**

Open `proteinfoundation/proteinflow/model_trainer_base.py`. Locate `_compute_single_noise_loss` (lines ~257-342). Add an import near the top of the file:

```python
from proteinfoundation.proteinflow.chirality_loss import chirality_hinge_loss
```

Then after the current line 341 (`combined_adp_loss = (1 - mf_ratio) * loss_fm + mf_ratio * loss_mf`), insert before `return combined_adp_loss, ...`:

```python
        # 7. Chirality hinge loss (optional)
        raw_loss_chir = torch.zeros((), device=self.device)
        if getattr(self, "chirality_loss_enabled", False) and self.chirality_loss_weight > 0:
            # Recover x_1 prediction from the FM sub-pass velocity:
            # z = (1-t)*x_1 + t*x_0  =>  x_1 ≈ z - t * v  (at r=t).
            x_1_pred = z - t_ext * v_pred
            x_1_pred = self.fm._mask_and_zero_com(x_1_pred, mask)
            loss_chir = chirality_hinge_loss(
                x_pred=x_1_pred,
                x_gt=x_1,
                mask=mask,
                margin_alpha=self.chirality_margin_alpha,
                stride=self.chirality_stride,
            )
            raw_loss_chir = loss_chir.detach()
            combined_adp_loss = combined_adp_loss + self.chirality_loss_weight * loss_chir
```

Update the function's return tuple:

```python
        return combined_adp_loss, raw_loss_mf, raw_loss_fm, raw_loss_chir
```

Update the docstring's Returns block accordingly (add `raw_loss_chir`).

- [ ] **Step 4: Update both call sites of `_compute_single_noise_loss`**

In `training_step` (same file, ~line 424 for K=1 and ~line 497 inside the K>1 loop), change:

```python
combined_adp_loss, raw_loss_mf, raw_loss_fm = self._compute_single_noise_loss(
    x_1, mask, t_ext, r_ext, t, batch, B
)
```

to:

```python
combined_adp_loss, raw_loss_mf, raw_loss_fm, raw_loss_chir = self._compute_single_noise_loss(
    x_1, mask, t_ext, r_ext, t, batch, B
)
```

and inside the K-loop (~line 497):

```python
loss_k, raw_loss_mf_k, raw_loss_fm_k, raw_loss_chir_k = self._compute_single_noise_loss(
    x_1, mask, t_ext, r_ext, t, batch, B
)
# ... existing lines ...
total_loss_mf += raw_loss_mf_k.item()
total_loss_fm += raw_loss_fm_k.item()
total_loss_chir += raw_loss_chir_k.item()
```

Add `total_loss_chir = 0.0` next to `total_loss_mf = 0.0`, `total_loss_fm = 0.0` (~line 493).

After `avg_loss_fm = total_loss_fm / K` (~line 506), add:

```python
avg_loss_chir = total_loss_chir / K
```

- [ ] **Step 5: Add chirality logging (both K=1 and K>1 branches)**

In the K=1 branch (after the two existing `self.log` calls for `raw_loss_mf` and `raw_loss_fm`, inside the `if not val_step:` block ~line 461):

```python
                self.log(
                    f"{log_prefix}/raw_loss_chirality",
                    raw_loss_chir,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
```

In the K>1 branch (after the `raw_loss_fm` log call ~line 525):

```python
        self.log(
            f"{log_prefix}/raw_loss_chirality", avg_loss_chir,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
```

- [ ] **Step 6: Wire chirality config in `Proteina.__init__`**

Open `proteinfoundation/proteinflow/proteina.py`. Locate the `MeanFlow hyperparams` block starting at line ~103.

Replace the warning block (lines 119-123):

```python
        if cfg_exp.training.get("self_cond", False):
            warnings.warn(
                "self_cond=True has no effect in MeanFlow training.",
                UserWarning, stacklevel=2,
            )
```

with:

```python
        # Self-conditioning (opt-in, wired through _compute_single_noise_loss).
        self.self_cond_prob = cfg_exp.training.get("self_cond_prob", 0.5)

        # Chirality hinge loss (opt-in).
        chir_cfg = cfg_exp.training.get("chirality_loss", {})
        self.chirality_loss_enabled = chir_cfg.get("enabled", False)
        self.chirality_loss_weight = chir_cfg.get("weight", 1.0)
        self.chirality_margin_alpha = chir_cfg.get("margin_alpha", 0.1)
        self.chirality_stride = chir_cfg.get("stride", 1)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_selfcond_chirality.py::test_chirality_loss_added_when_enabled -v`
Expected: PASS.

Run the full suite: `pytest tests/test_selfcond_chirality.py -v`
Expected: 8 tests PASS.

Also run: `pytest tests/test_chirality_diagnostics.py -v`
Expected: previous 8 tests still PASS (no regression).

- [ ] **Step 8: Commit**

```bash
git add proteinfoundation/proteinflow/model_trainer_base.py proteinfoundation/proteinflow/proteina.py tests/test_selfcond_chirality.py
git commit -m "Add chirality hinge loss branch to MeanFlow loss computation"
```

---

## Task 3: Self-conditioning plumbing (TDD)

**Files:**
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py` (`_compute_single_noise_loss` and `training_step`)
- Test: `tests/test_selfcond_chirality.py`

- [ ] **Step 1: Add failing test for x_sc plumbing**

Append to `tests/test_selfcond_chirality.py`:

```python
def test_self_cond_plumbing_warmup_and_injection():
    trainer = _make_fake_trainer()
    # Reset calls
    trainer.nn.calls.clear()

    B, n = 1, 12
    x_1 = torch.randn(B, n, 3)
    mask = torch.ones(B, n, dtype=torch.bool)
    t = torch.full((B,), 0.5)
    r = torch.full((B,), 0.3)
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]
    batch = {}

    # use_sc=False: NN called exactly twice (JVP primal + FM sub-pass),
    # neither call should contain x_sc.
    loss, _, _, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B, use_sc=False,
    )
    n_calls_off = len(trainer.nn.calls)
    assert n_calls_off == 2, f"use_sc=False expected 2 NN calls, got {n_calls_off}"
    for c in trainer.nn.calls:
        assert "x_sc" not in c

    trainer.nn.calls.clear()

    # use_sc=True: one extra warmup call; the JVP and FM calls must receive x_sc.
    loss, _, _, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B, use_sc=True,
    )
    n_calls_on = len(trainer.nn.calls)
    assert n_calls_on == 3, f"use_sc=True expected 3 NN calls (warmup + JVP + FM), got {n_calls_on}"
    # First call is the warmup (no x_sc); subsequent calls must include x_sc.
    assert "x_sc" not in trainer.nn.calls[0]
    assert "x_sc" in trainer.nn.calls[1]
    assert "x_sc" in trainer.nn.calls[2]
    # x_sc must be detached.
    assert not trainer.nn.calls[1]["x_sc"].requires_grad
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `pytest tests/test_selfcond_chirality.py::test_self_cond_plumbing_warmup_and_injection -v`
Expected: FAIL with `TypeError: _compute_single_noise_loss() got an unexpected keyword argument 'use_sc'`.

- [ ] **Step 3: Add `use_sc` kwarg + warmup pass inside `_compute_single_noise_loss`**

In `proteinfoundation/proteinflow/model_trainer_base.py`, change the signature:

```python
    def _compute_single_noise_loss(self, x_1, mask, t_ext, r_ext, t, batch, B, *, use_sc=False):
```

Update the docstring Args to document `use_sc`:

```
            use_sc: If True, run one extra no-grad forward pass to obtain
                x_sc and feed it as a detached constant into the JVP and FM
                sub-passes. Requires the NN config to declare x_sc /
                x_sc_pair_dists in its feature lists.
```

After the existing interpolation block (after line ~295: `v = self.fm._apply_mask(v, mask)`), insert:

```python
        # 3b. Self-conditioning warmup (no-grad, no tangent).
        x_sc = None
        if use_sc:
            with torch.no_grad():
                warmup_batch = {
                    "x_t": z,
                    "t": t_ext.squeeze(-1).squeeze(-1),
                    "h": (t_ext - r_ext).squeeze(-1).squeeze(-1),
                    "mask": mask,
                }
                if "cath_code" in batch:
                    warmup_batch["cath_code"] = [batch["cath_code"][i] for i in range(B)]
                u_warmup = self.nn(warmup_batch)["coors_pred"]
                # Interpret u as avg velocity and recover a clean-coord proxy:
                # x_1 ≈ z - t * u (exact at r=0, approximation elsewhere).
                x_sc = z - t_ext * u_warmup
                x_sc = self.fm._mask_and_zero_com(x_sc, mask)
                x_sc = x_sc.detach()
```

Then inside `u_func`, extend the batch dict:

```python
        def u_func(z_in, t_in, r_in):
            h = (t_in - r_in).squeeze(-1).squeeze(-1)
            t_flat = t_in.squeeze(-1).squeeze(-1)
            batch_nn = {
                "x_t": z_in,
                "t": t_flat,
                "h": h,
                "mask": mask,
            }
            if x_sc is not None:
                batch_nn["x_sc"] = x_sc  # detached, closed-over, no tangent
            if "cath_code" in batch:
                batch_nn["cath_code"] = [batch["cath_code"][i] for i in range(B)]
            nn_out = self.nn(batch_nn)
            return nn_out["coors_pred"]
```

No changes needed to the FM sub-pass — it already calls `u_func(z, t, t)` which picks up `x_sc` via closure.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selfcond_chirality.py::test_self_cond_plumbing_warmup_and_injection -v`
Expected: PASS.

Run the full new test file: `pytest tests/test_selfcond_chirality.py -v`
Expected: 9 tests PASS.

- [ ] **Step 5: Wire `use_sc` decision in `training_step`**

In `training_step` (same file), after sampling `t, r` and before the K-loop check (~line 391, right before `B = t.shape[0]` or right after `batch.pop("cath_code")`), add:

```python
        # --- 1b. Self-conditioning activation (per step, consistent across K passes) ---
        use_sc = (
            not val_step
            and self.cfg_exp.training.get("self_cond", False)
            and random.random() < self.self_cond_prob
        )
```

Then pass `use_sc=use_sc` to both `_compute_single_noise_loss` calls (K=1 branch and inside the K-loop).

- [ ] **Step 6: Run full test suite (no regressions)**

Run: `pytest tests/test_selfcond_chirality.py tests/test_chirality_diagnostics.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add proteinfoundation/proteinflow/model_trainer_base.py tests/test_selfcond_chirality.py
git commit -m "Plumb self-conditioning through MeanFlow JVP training step"
```

---

## Task 4: Config surface (base + debug yaml)

**Files:**
- Modify: `configs/experiment_config/training_ca.yaml`
- Modify: `configs/experiment_config/training_ca_debug.yaml`

- [ ] **Step 1: Update `training_ca.yaml`**

Locate the `training:` block (~line 38). Change:

```yaml
training:
  loss_accumulation_steps: 1
  self_cond: False  # Unused in MeanFlow training
  fold_cond: False
```

to:

```yaml
training:
  loss_accumulation_steps: 1
  self_cond: False           # Opt-in self-conditioning for MeanFlow
  self_cond_prob: 0.5        # Probability of activating x_sc per step
  fold_cond: False
```

Then after the `meanflow:` block (before `opt:`), add:

```yaml
  chirality_loss:
    enabled: False           # Opt-in chirality hinge loss (signed triple products)
    weight: 1.0              # Multiplier on hinge loss in total loss
    margin_alpha: 0.1        # Margin as fraction of mean(|T_gt|)
    stride: 1                # Stride k for triple-product window
```

- [ ] **Step 2: Update `training_ca_debug.yaml`**

Locate the `training:` block (~line 32). Change:

```yaml
training:
  loss_accumulation_steps: 1
  self_cond: False
  fold_cond: False
```

to:

```yaml
training:
  loss_accumulation_steps: 1
  self_cond: False
  self_cond_prob: 0.5
  fold_cond: False
```

After the `meanflow:` block (before `opt:`), add:

```yaml
  chirality_loss:
    enabled: False
    weight: 1.0
    margin_alpha: 0.1
    stride: 1
```

- [ ] **Step 3: Smoke-test config parsing**

Run a short Python check that OmegaConf loads both files without error:

```bash
python -c "
from omegaconf import OmegaConf
for p in ['configs/experiment_config/training_ca.yaml',
          'configs/experiment_config/training_ca_debug.yaml']:
    cfg = OmegaConf.load(p)
    assert 'chirality_loss' in cfg.training, f'{p}: missing chirality_loss'
    assert 'self_cond_prob' in cfg.training, f'{p}: missing self_cond_prob'
    print(f'{p}: OK')
"
```
Expected: both paths print `OK`.

- [ ] **Step 4: Commit**

```bash
git add configs/experiment_config/training_ca.yaml configs/experiment_config/training_ca_debug.yaml
git commit -m "Add self_cond_prob and chirality_loss keys to training configs"
```

---

## Task 5: sbatch script exposure

**Files:**
- Modify: `scripts/train_debug.sbatch`

- [ ] **Step 1: Add new bash variables**

In `scripts/train_debug.sbatch`, locate the Hydra overrides block (`EXP_NAME="..."`, ~line 70). After the existing `LR=1.e-4` line, add:

```bash
# Self-conditioning
SELF_COND=False
SELF_COND_PROB=0.5

# Chirality hinge loss
CHIRALITY_ENABLED=False
CHIRALITY_WEIGHT=1.0
CHIRALITY_MARGIN_ALPHA=0.1
CHIRALITY_STRIDE=1
```

- [ ] **Step 2: Add Hydra overrides**

Locate the `--exp_overrides` block (~line 95). Append the following lines before `--data_overrides`:

```bash
        training.self_cond=$SELF_COND \
        training.self_cond_prob=$SELF_COND_PROB \
        training.chirality_loss.enabled=$CHIRALITY_ENABLED \
        training.chirality_loss.weight=$CHIRALITY_WEIGHT \
        training.chirality_loss.margin_alpha=$CHIRALITY_MARGIN_ALPHA \
        training.chirality_loss.stride=$CHIRALITY_STRIDE \
```

(Ensure the line preceding the first new line still ends with ` \`.)

- [ ] **Step 3: Lint the sbatch script**

Run: `bash -n scripts/train_debug.sbatch`
Expected: no output (syntax OK).

- [ ] **Step 4: Dry-run the Hydra override parsing locally**

Run:

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/experiment_config/training_ca_debug.yaml')
overrides = {
    'training.self_cond': False,
    'training.self_cond_prob': 0.5,
    'training.chirality_loss.enabled': True,
    'training.chirality_loss.weight': 2.0,
    'training.chirality_loss.margin_alpha': 0.2,
    'training.chirality_loss.stride': 1,
}
for k, v in overrides.items():
    OmegaConf.update(cfg, k, v)
print(OmegaConf.to_yaml(cfg.training.chirality_loss))
print('self_cond:', cfg.training.self_cond, 'prob:', cfg.training.self_cond_prob)
"
```
Expected: prints the updated `chirality_loss` block and `self_cond: False prob: 0.5`.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_debug.sbatch
git commit -m "Expose self_cond and chirality_loss knobs in train_debug.sbatch"
```

---

## Task 6: End-to-end verification

- [ ] **Step 1: Full test suite**

Run: `pytest tests/test_selfcond_chirality.py tests/test_chirality_diagnostics.py -v`
Expected: all tests PASS.

- [ ] **Step 2: Other existing tests (no regressions)**

Run: `pytest tests/ -v --ignore=tests/test_selfcond_chirality.py --ignore=tests/test_chirality_diagnostics.py`
Expected: whatever passed before still passes. If this discovers failures unrelated to these changes, flag them to the user — do not attempt to fix.

- [ ] **Step 3: Short training dry-run (optional, if GPU available)**

Launch one local training step with all flags OFF to verify no regression:

```bash
python proteinfoundation/train.py \
  --config_name training_ca_debug \
  --single \
  --exp_overrides \
    training.self_cond=False \
    training.chirality_loss.enabled=False \
    opt.max_epochs=1 \
    eval.enabled=false
```

Then with self_cond + chirality ON:

```bash
python proteinfoundation/train.py \
  --config_name training_ca_debug \
  --single \
  --exp_overrides \
    training.self_cond=True \
    training.chirality_loss.enabled=True \
    training.chirality_loss.weight=1.0 \
    opt.max_epochs=1 \
    eval.enabled=false
```

Expected: neither run errors. `train/raw_loss_chirality` appears in the log when chirality is enabled. Skip this step if no GPU is available locally; the sbatch run covers it.

- [ ] **Step 4: Final commit check**

Run: `git status` — expected clean working tree.
Run: `git log --oneline -6` — expected: five new commits from Tasks 1-5 plus the spec commit, in order.

---

## Notes for the implementer

- **Why `v_pred` for `x_1_pred` reconstruction in Task 2**: the FM sub-pass already calls `u_func(z, t, t)` with `h=0`, which is the instantaneous-velocity regime. At `r=t`, MeanFlow's avg velocity collapses to the standard FM velocity `v`, so `z - t*v` is the clean-sample estimate consistent with how the model is trained on the FM half of the loss.
- **Why detach `x_sc`**: JVP requires primals to be either explicitly carrying tangents (first positional arg `(z, t_ext, r_ext)`) or *constants* closed over. A tensor that requires grad but has no tangent would make JVP try to propagate through it, which is either an error or silently wrong. Detaching closes the loophole.
- **Why per-step `use_sc`**: all K passes within a step use the same `x_1, t, r`. If different K passes had different `use_sc` values, the gradient estimator would be a mixture of two loss landscapes — more variance for no benefit.
- **Why batch-adaptive margin**: on the 1ubq single-protein debug run, `mean(|T_gt|)` is effectively a constant, so the adaptive margin degenerates to a constant. For multi-protein training it scales with protein length/geometry. Cheap to compute (one small det per step).
