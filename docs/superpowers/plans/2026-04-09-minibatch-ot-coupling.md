# Mini-Batch Optimal Transport Coupling for Flow Matching

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the independent noise-data pairing in the flow matching training loop with mini-batch optimal transport (OT) coupling, so that each noise sample x_0 is paired with the data sample x_1 that minimizes transport cost within the batch.

**Architecture:** We adapt `OTPlanSampler` from the [torchcfm](https://github.com/atong01/conditional-flow-matching) library (Tong et al., ICLR 2024) into a thin wrapper that handles masked/variable-length protein data. The wrapper is called in the training step between noise sampling and interpolation. The dataloader is NOT modified (see assessment below). A config flag enables/disables OT coupling.

**Tech Stack:** PyTorch, [POT (Python Optimal Transport)](https://pythonot.github.io/) library, scipy.optimize.linear_sum_assignment

---

## Assessment: Where Should OT Coupling Live?

**The OT plan should be computed in the training step, NOT in the dataloader.** Here is why:

1. **Noise is sampled in the training step** ([model_trainer_base.py:246-248](proteinfoundation/proteinflow/model_trainer_base.py#L246-L248)), not in the dataloader. The dataloader only returns clean protein structures. Mini-batch OT requires both noise (x_0) and data (x_1) to be simultaneously available, which only happens in `training_step()`.

2. **OT is a per-batch operation.** The OT plan depends on which specific samples happen to land in the same batch and on freshly-sampled noise. It must be recomputed every training step. Precomputing it in the dataloader would require the dataloader to know about the noise distribution and sample noise itself — mixing data-loading concerns with training logic.

3. **Minimal code change.** The modification is a 3-line insertion in `training_step()` between the existing `sample_reference()` call and `interpolate()` call. No dataloader classes need to change.

4. **Performance.** Solving a B x B assignment problem (B = batch size, typically 32-128) takes <1ms on CPU via Hungarian algorithm. This is negligible compared to the forward/backward pass. No need to overlap it with data loading.

**What COULD be modified in the dataloader** (optional enhancement, NOT in this plan): length-bucketed batching to ensure proteins within a batch have similar lengths. This would make the OT cost matrix more meaningful since all samples would have comparable dimensionality. However, this is an optimization, not a requirement — the OT sampler handles variable-length proteins via masked cost computation.

---

## Reference: OTPlanSampler from torchcfm

The implementation builds on `OTPlanSampler` from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py. Key methods used:

- **`get_map(x0, x1)`** — Computes the OT plan as a [B, B] matrix using POT solvers. Internally reshapes to [B, -1], computes `torch.cdist(x0, x1)**2` as cost matrix, calls the selected POT solver.
- **`sample_map(pi, batch_size)`** — Draws (i, j) index pairs from the plan probabilistically.
- **`sample_plan(x0, x1)`** — Calls `get_map` then `sample_map`. Returns `(x0[i], x1[j])` — both tensors permuted.
- **`sample_plan_with_scipy(x0, x1)`** — Uses `scipy.optimize.linear_sum_assignment` for deterministic OT. Returns `(x0[j], x1)` — only noise permuted, data order preserved. Lower variance.

**Adaptation needed for proteins:** The original `OTPlanSampler` works on flat tensors. For variable-length proteins with masks, we must zero out padded positions before computing the cost matrix to prevent padding from affecting the OT plan.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `proteinfoundation/flow_matching/ot_sampler.py` | **Create** | `OTPlanSampler` from torchcfm + thin `MaskedOTPlanSampler` wrapper |
| `proteinfoundation/proteinflow/model_trainer_base.py` | **Modify** (lines 44-49 and 244-254) | Call OT sampler in training_step between noise sampling and interpolation |
| `proteinfoundation/proteinflow/proteina.py` | **Modify** (lines 58-65) | Instantiate OT sampler from config |
| `configs/experiment_config/training_ca.yaml` | **Modify** | Add `ot_coupling` config section |
| `tests/test_ot_sampler.py` | **Create** | Unit + integration tests |

---

## Task 1: Add OTPlanSampler and Masked Wrapper

**Files:**
- Create: `proteinfoundation/flow_matching/ot_sampler.py`
- Create: `tests/test_ot_sampler.py`

### Step 1.1: Write failing tests

- [ ] **Step 1.1a: Create test file**

```python
# tests/test_ot_sampler.py
import torch
import pytest
import numpy as np

from proteinfoundation.flow_matching.ot_sampler import OTPlanSampler, MaskedOTPlanSampler


class TestOTPlanSampler:
    """Tests for the base OTPlanSampler (from torchcfm)."""

    def test_get_map_shape(self):
        """OT plan should be [B, B]."""
        sampler = OTPlanSampler(method="exact")
        B = 8
        x0 = torch.randn(B, 30)
        x1 = torch.randn(B, 30)
        pi = sampler.get_map(x0, x1)
        assert pi.shape == (B, B)

    def test_get_map_is_valid_plan(self):
        """OT plan rows and columns should sum to 1/B (uniform marginals)."""
        sampler = OTPlanSampler(method="exact")
        B = 8
        x0 = torch.randn(B, 30)
        x1 = torch.randn(B, 30)
        pi = sampler.get_map(x0, x1)
        np.testing.assert_allclose(pi.sum(axis=0), np.ones(B) / B, atol=1e-5)
        np.testing.assert_allclose(pi.sum(axis=1), np.ones(B) / B, atol=1e-5)

    def test_sample_plan_output_shapes(self):
        """sample_plan should return tensors of same shape as input."""
        sampler = OTPlanSampler(method="exact")
        B = 8
        x0 = torch.randn(B, 30)
        x1 = torch.randn(B, 30)
        x0_ot, x1_ot = sampler.sample_plan(x0, x1)
        assert x0_ot.shape == x0.shape
        assert x1_ot.shape == x1.shape

    def test_sample_plan_with_3d_tensors(self):
        """sample_plan should work with [B, N, 3] shaped inputs."""
        sampler = OTPlanSampler(method="exact")
        B, N = 8, 20
        x0 = torch.randn(B, N, 3)
        x1 = torch.randn(B, N, 3)
        x0_ot, x1_ot = sampler.sample_plan(x0, x1)
        # Note: torchcfm flattens internally, output is [B, N*3]
        assert x0_ot.shape[0] == B
        assert x1_ot.shape[0] == B

    def test_sample_plan_with_scipy_preserves_x1_order(self):
        """sample_plan_with_scipy should not permute x1."""
        sampler = OTPlanSampler(method="exact")
        B = 8
        x0 = torch.randn(B, 30)
        x1 = torch.randn(B, 30)
        _, x1_ot = sampler.sample_plan_with_scipy(x0, x1)
        assert torch.equal(x1_ot, x1)


class TestMaskedOTPlanSampler:
    """Tests for the masked wrapper handling variable-length proteins."""

    def test_output_shape_preserved(self):
        """Output should be [B, N, 3] matching input shape."""
        sampler = MaskedOTPlanSampler(method="exact")
        B, N = 8, 20
        x_0 = torch.randn(B, N, 3)
        x_1 = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)
        x_0_ot, x_1_ot = sampler.sample_plan(x_0, x_1, mask)
        assert x_0_ot.shape == (B, N, 3)
        assert x_1_ot.shape == (B, N, 3)

    def test_x1_order_preserved(self):
        """Data samples x_1 should not be permuted."""
        sampler = MaskedOTPlanSampler(method="exact")
        B, N = 8, 20
        x_0 = torch.randn(B, N, 3)
        x_1 = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)
        _, x_1_ot = sampler.sample_plan(x_0, x_1, mask)
        assert torch.equal(x_1_ot, x_1)

    def test_output_is_permutation_of_input(self):
        """Each row of x_0_ot should be some row of x_0."""
        sampler = MaskedOTPlanSampler(method="exact")
        B, N = 4, 10
        x_0 = torch.randn(B, N, 3)
        x_1 = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)
        x_0_ot, _ = sampler.sample_plan(x_0, x_1, mask)
        for i in range(B):
            found = any(
                torch.allclose(x_0_ot[i], x_0[j], atol=1e-6) for j in range(B)
            )
            assert found, f"Row {i} of output not found in input"

    def test_ot_reduces_total_transport_cost(self):
        """OT pairing should have lower total cost than identity pairing."""
        sampler = MaskedOTPlanSampler(method="exact")
        B, N = 32, 30
        x_0 = torch.randn(B, N, 3)
        x_1 = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)

        identity_cost = ((x_0 - x_1) ** 2).sum(dim=(-1, -2)).sum()
        x_0_ot, _ = sampler.sample_plan(x_0, x_1, mask)
        ot_cost = ((x_0_ot - x_1) ** 2).sum(dim=(-1, -2)).sum()
        assert ot_cost <= identity_cost + 1e-4

    def test_masking_affects_assignment(self):
        """Different masks should produce different OT assignments."""
        sampler = MaskedOTPlanSampler(method="exact")
        B, N = 8, 30
        torch.manual_seed(42)
        x_0 = torch.randn(B, N, 3)
        x_1 = torch.randn(B, N, 3)

        mask_full = torch.ones(B, N, dtype=torch.bool)
        mask_partial = torch.ones(B, N, dtype=torch.bool)
        mask_partial[:, 15:] = False

        x_0_full, _ = sampler.sample_plan(x_0, x_1, mask_full)
        x_0_partial, _ = sampler.sample_plan(x_0, x_1, mask_partial)
        # With different masks, the cost matrix changes, so assignments differ
        # (not guaranteed for every seed, but very likely for B=8)
        assert not torch.equal(x_0_full, x_0_partial)

    def test_variable_length_proteins(self):
        """Should handle proteins of different lengths in same batch."""
        sampler = MaskedOTPlanSampler(method="exact")
        B, N = 6, 40
        x_0 = torch.randn(B, N, 3)
        x_1 = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)
        lengths = [15, 20, 25, 30, 35, 40]
        for i, length in enumerate(lengths):
            mask[i, length:] = False

        x_0_ot, x_1_ot = sampler.sample_plan(x_0, x_1, mask)
        assert x_0_ot.shape == (B, N, 3)
        assert x_1_ot.shape == (B, N, 3)
```

- [ ] **Step 1.1b: Run tests to verify they fail**

Run: `cd /home/schekmenev/code_projects/meanflow-ot-proteina && python -m pytest tests/test_ot_sampler.py -v 2>&1 | head -10`
Expected: FAIL with `ModuleNotFoundError: No module named 'proteinfoundation.flow_matching.ot_sampler'`

### Step 1.2: Implement OTPlanSampler and MaskedOTPlanSampler

- [ ] **Step 1.2a: Create ot_sampler.py with OTPlanSampler from torchcfm and MaskedOTPlanSampler wrapper**

```python
# proteinfoundation/flow_matching/ot_sampler.py
"""
Mini-batch optimal transport coupling for flow matching on protein structures.

OTPlanSampler is adapted from the torchcfm library by Tong et al.:
https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

Reference: Tong et al., "Improving and Generalizing Flow-Based Generative Models
with Minibatch Optimal Transport" (ICLR 2024).

MaskedOTPlanSampler is a thin wrapper that handles masked/variable-length protein
data by zeroing out padded positions before computing the OT cost matrix.
"""

import warnings
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import ot as pot
import torch
from torch import Tensor


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan
    (wrt squared Euclidean cost) with different implementations of the plan
    calculation.

    Adapted from torchcfm (https://github.com/atong01/conditional-flow-matching).
    """

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(
                pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m
            )
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source
        and a target minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi $(x,z) \sim \pi$.

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        batch_size : int
            number of samples to draw
        replace : bool
            sampling with or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays
            indices of source and target data samples from pi
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan and draw source and target samples from it.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        replace : bool
            sampling with or without replacement

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            source minibatch drawn from OT plan
        x1[j] : Tensor, shape (bs, *dim)
            target minibatch drawn from OT plan
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_scipy(self, x0, x1):
        r"""Compute deterministic OT assignment using scipy. Only permutes x0;
        preserves x1 order. Lower variance than sample_plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        x0[j] : Tensor, shape (bs, *dim)
            source minibatch permuted by OT assignment
        x1 : Tensor, shape (bs, *dim)
            target minibatch (unchanged)
        """
        import scipy

        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0.detach(), x1.detach()) ** 2
        if self.normalize_cost:
            M = M / M.max()
        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        pi_x0 = x0[j]
        pi_x1 = x1
        return pi_x0, pi_x1


class MaskedOTPlanSampler:
    """Wrapper around OTPlanSampler that handles masked/variable-length
    protein data.

    Before computing the OT cost matrix, padded positions (where mask=False)
    are zeroed out so they don't affect the transport cost. The original
    [B, N, 3] tensor shapes are preserved in the output (unlike base
    OTPlanSampler which flattens).

    Uses scipy's linear_sum_assignment for deterministic OT (lower variance).
    Only x_0 (noise) is permuted; x_1 (data) order is preserved.
    """

    def __init__(
        self,
        method: str = "exact",
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        self.ot_plan_sampler = OTPlanSampler(
            method=method,
            reg=reg,
            reg_m=reg_m,
            normalize_cost=normalize_cost,
            num_threads=num_threads,
            warn=warn,
        )

    def sample_plan(
        self,
        x_0: Tensor,
        x_1: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Re-pair noise with data via OT, handling variable-length masks.

        Zeroes out padded positions before computing OT cost, then applies
        deterministic Hungarian assignment. Returns permuted noise in original
        [B, N, 3] shape.

        Args:
            x_0: Noise samples, shape [B, N, 3]
            x_1: Data samples, shape [B, N, 3]
            mask: Boolean residue mask, shape [B, N]. True = valid position.

        Returns:
            Tuple of (x_0_permuted, x_1_unchanged), both shape [B, N, 3].
        """
        import scipy

        B, N, D = x_0.shape

        # Zero out padded positions so they don't affect OT cost
        mask_3d = mask[..., None]  # [B, N, 1]
        x_0_masked = (x_0 * mask_3d).reshape(B, -1)  # [B, N*3]
        x_1_masked = (x_1 * mask_3d).reshape(B, -1)  # [B, N*3]

        # Compute cost matrix
        M = torch.cdist(x_0_masked, x_1_masked) ** 2  # [B, B]
        if self.ot_plan_sampler.normalize_cost:
            M = M / M.max()

        # Solve assignment (Hungarian algorithm)
        _, j = scipy.optimize.linear_sum_assignment(M.detach().cpu().numpy())

        # Permute x_0 using original (non-flattened) tensor
        j_tensor = torch.tensor(j, dtype=torch.long, device=x_0.device)
        x_0_permuted = x_0[j_tensor]

        return x_0_permuted, x_1
```

- [ ] **Step 1.2b: Run tests to verify they pass**

Run: `cd /home/schekmenev/code_projects/meanflow-ot-proteina && python -m pytest tests/test_ot_sampler.py -v`
Expected: All tests PASS

- [ ] **Step 1.3: Commit**

```bash
git add proteinfoundation/flow_matching/ot_sampler.py tests/test_ot_sampler.py
git commit -m "feat: add OTPlanSampler and MaskedOTPlanSampler for mini-batch OT coupling"
```

---

## Task 2: Integrate MaskedOTPlanSampler into Training Step

**Files:**
- Modify: `proteinfoundation/proteinflow/model_trainer_base.py` (lines 44-49 and 244-254)
- Modify: `proteinfoundation/proteinflow/proteina.py` (lines 58-65)

### Step 2.1: Add OT sampler initialization in Proteina

- [ ] **Step 2.1a: Import and instantiate MaskedOTPlanSampler in Proteina.__init__**

In `proteinfoundation/proteinflow/proteina.py`, add the import at the top (after line 27, with the other flow_matching imports):

```python
from proteinfoundation.flow_matching.ot_sampler import MaskedOTPlanSampler
```

Then in `__init__` (after line 65, after `self.fm = R3NFlowMatcher(...)`), add:

```python
        # Optimal transport coupling
        ot_cfg = cfg_exp.training.get("ot_coupling", {})
        if ot_cfg.get("enabled", False):
            self.ot_sampler = MaskedOTPlanSampler(
                method=ot_cfg.get("method", "exact"),
                reg=ot_cfg.get("reg", 0.05),
                reg_m=ot_cfg.get("reg_m", 1.0),
                normalize_cost=ot_cfg.get("normalize_cost", False),
            )
        else:
            self.ot_sampler = None
```

- [ ] **Step 2.1b: Initialize ot_sampler attribute in ModelTrainerBase**

In `proteinfoundation/proteinflow/model_trainer_base.py`, add after line 49 (`self.fm = None`):

```python
        self.ot_sampler = None  # Overridden by subclasses if OT coupling enabled
```

### Step 2.2: Modify training_step to use OT coupling

- [ ] **Step 2.2a: Add OT re-pairing between noise sampling and interpolation**

In `proteinfoundation/proteinflow/model_trainer_base.py`, in `training_step()`, insert the following block between the `x_0 = self.fm.sample_reference(...)` call (line 248) and the motif conditioning block (line 250):

```python
        # Optimal transport coupling: re-pair noise with data
        if self.ot_sampler is not None:
            x_0, _ = self.ot_sampler.sample_plan(x_0, x_1, mask)
```

The resulting code in context should read:

```python
        # Sample time, reference and align reference to target
        t = self.sample_t(batch_shape)
        x_0 = self.fm.sample_reference(
            n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
        )

        # Optimal transport coupling: re-pair noise with data
        if self.ot_sampler is not None:
            x_0, _ = self.ot_sampler.sample_plan(x_0, x_1, mask)

        if self.motif_conditioning:
            batch.update(self.motif_factory(batch))
            x_1 = batch["x_1"]
        # Interpolation
        x_t = self.fm.interpolate(x_0, x_1, t)
```

- [ ] **Step 2.3: Commit**

```bash
git add proteinfoundation/proteinflow/proteina.py proteinfoundation/proteinflow/model_trainer_base.py
git commit -m "feat: integrate MaskedOTPlanSampler into training loop"
```

---

## Task 3: Add Configuration

**Files:**
- Modify: `configs/experiment_config/training_ca.yaml`

### Step 3.1: Add OT coupling config section

- [ ] **Step 3.1a: Add ot_coupling block to training config**

In `configs/experiment_config/training_ca.yaml`, add the following inside the `training:` section (after line 44, after `fold_label_sample_ratio`):

```yaml
  # Optimal transport coupling for noise-data pairing (Tong et al., ICLR 2024)
  # When enabled, uses mini-batch OT to pair noise samples with data samples
  # that minimize transport cost, reducing flow matching variance.
  ot_coupling:
    enabled: False  # Set to True to enable OT coupling
    method: exact   # "exact" (Hungarian/EMD), "sinkhorn", "unbalanced", or "partial"
    reg: 0.05       # Regularization for Sinkhorn-based solvers (ignored if method=exact)
    reg_m: 1.0      # Regularization weight for unbalanced solver
    normalize_cost: False  # Normalize cost matrix (not recommended for minibatches)
```

- [ ] **Step 3.2: Commit**

```bash
git add configs/experiment_config/training_ca.yaml
git commit -m "feat: add OT coupling configuration options"
```

---

## Task 4: Add Integration Tests

**Files:**
- Create: `tests/test_ot_integration.py`

### Step 4.1: Write integration tests with R3NFlowMatcher

- [ ] **Step 4.1a: Create integration test file**

```python
# tests/test_ot_integration.py
import torch
import pytest

from proteinfoundation.flow_matching.ot_sampler import MaskedOTPlanSampler
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher


class TestOTIntegrationWithFlowMatcher:
    """Test OT coupling works correctly within the flow matching pipeline."""

    def test_ot_coupled_interpolation_same_shape(self):
        """OT-coupled interpolation should produce same shapes as standard."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 30
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)
        t = torch.rand(B)

        # Standard interpolation
        x_t_standard = fm.interpolate(x_0, x_1, t, mask)

        # OT-coupled interpolation
        x_0_ot, _ = ot_sampler.sample_plan(x_0, x_1, mask)
        x_t_ot = fm.interpolate(x_0_ot, x_1, t, mask)

        assert x_t_standard.shape == x_t_ot.shape == (B, N, 3)

    def test_ot_reduces_average_transport_distance(self):
        """OT-paired (x_0, x_1) should have lower avg distance than random."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 32, 50
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)

        random_dist = ((x_0 - x_1) ** 2).sum(dim=(-1, -2)).mean()
        x_0_ot, _ = ot_sampler.sample_plan(x_0, x_1, mask)
        ot_dist = ((x_0_ot - x_1) ** 2).sum(dim=(-1, -2)).mean()

        assert ot_dist <= random_dist

    def test_ot_with_variable_length_masks(self):
        """OT coupling should work with variable-length proteins."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 40
        mask = torch.ones(B, N, dtype=torch.bool)
        lengths = [15, 20, 25, 30, 35, 40, 18, 22]
        for i, length in enumerate(lengths):
            mask[i, length:] = False

        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)
        x_0_ot, x_1_ot = ot_sampler.sample_plan(x_0, x_1, mask)

        assert x_0_ot.shape == (B, N, 3)
        assert torch.equal(x_1_ot, x_1)

    def test_ot_preserves_zero_com(self):
        """OT coupling should preserve zero center-of-mass."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 30
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)

        x_0_ot, _ = ot_sampler.sample_plan(x_0, x_1, mask)
        # OT only permutes rows — each row was already zero-COM
        com = x_0_ot.mean(dim=1)  # [B, 3]
        assert torch.allclose(com, torch.zeros_like(com), atol=1e-5)

    def test_full_training_step_shapes(self):
        """Simulate the full training step flow with OT coupling."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 30
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        t = torch.rand(B)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)

        # OT coupling (the new step)
        x_0, _ = ot_sampler.sample_plan(x_0, x_1, mask)

        # Interpolation (existing step)
        x_t = fm.interpolate(x_0, x_1, t, mask)

        # Target (existing step)
        x_t_dot = fm.xt_dot(x_1, x_t, t, mask)

        assert x_t.shape == (B, N, 3)
        assert x_t_dot.shape == (B, N, 3)
```

- [ ] **Step 4.1b: Run all tests**

Run: `cd /home/schekmenev/code_projects/meanflow-ot-proteina && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4.2: Commit**

```bash
git add tests/test_ot_integration.py
git commit -m "test: add integration tests for OT coupling with flow matcher"
```

---

## Design Notes

### On Using OTPlanSampler from torchcfm

We vendor (copy) `OTPlanSampler` rather than adding `torchcfm` as a dependency because:
1. We only need one class from the library
2. We need a wrapper for masked protein data that the original doesn't support
3. Avoids potential version conflicts in the conda environment

The vendored code is kept faithful to the original with clear attribution.

### On MaskedOTPlanSampler vs Direct OTPlanSampler Usage

`MaskedOTPlanSampler` wraps the OT cost computation to:
1. Zero out padded positions before computing `torch.cdist` so padding doesn't pollute the cost
2. Preserve the original `[B, N, 3]` tensor shapes (base `OTPlanSampler.sample_plan_with_scipy` flattens to `[B, N*3]`)
3. Use deterministic Hungarian assignment (via scipy) for lower variance and reproducibility

### On Performance

- **Hungarian algorithm**: O(B^3) for batch size B. For B=128: ~2M ops, <1ms on CPU. Negligible vs forward/backward pass.
- **CPU roundtrip**: `scipy.optimize.linear_sum_assignment` requires `.cpu().numpy()`. For B <= 512 this is fast. For very large batches, the Sinkhorn method runs on CPU via POT but is iterative.
- **Cost matrix**: `torch.cdist` on GPU, O(B^2 * N * 3). Single kernel call.

### On Mask Handling After Permutation

After OT permutation, `x_0[perm[j]]` may have been generated with a different mask than `x_1[j]` (different protein lengths). This is handled correctly because:
1. The downstream `fm.interpolate(x_0, x_1, t)` calls `_mask_and_zero_com(x_0, mask)` which re-applies the target mask
2. The `fm.sample_reference()` produces centered Gaussian noise — permuting these samples gives another valid set of centered Gaussian samples
3. Positions that are "new" after permutation (valid in target but not in source) will have whatever values were in the padded region of the source — but `_mask_and_zero_com` zeros and re-centers these
