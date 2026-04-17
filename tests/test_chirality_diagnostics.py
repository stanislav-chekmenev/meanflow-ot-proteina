"""Tests for chirality / mirror-image diagnostics added to protein_eval.py.

Covers:
  - Pure-numpy geometric helpers (_ca_rmsd, _ca_rmsd_with_reflection,
    _chirality_sign, _kabsch_svd) under identity / translation / rotation /
    reflection transforms.
  - ProteinEvalCallback._fm_loss_mirror_diagnostic with a fake O(3)-equivariant
    network (should report zero loss gap) and a non-equivariant one (should
    report a meaningful gap).
"""

import sys
import types
import unittest.mock as mock

# Stub torch_scatter before any proteinfoundation import (CUDA-only .so fails
# to load on CPU-only nodes; scatter_mean is unused by these tests).
_ts_stub = types.ModuleType("torch_scatter")
_ts_stub.scatter_mean = None
sys.modules.setdefault("torch_scatter", _ts_stub)

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from proteinfoundation.callbacks.protein_eval import (
    ProteinEvalCallback,
    _ca_rmsd,
    _ca_rmsd_with_reflection,
    _chirality_sign,
    _kabsch_svd,
)


# ----------------------------------------------------------------------------
# Geometric helpers
# ----------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(20260417)


@pytest.fixture
def cloud(rng):
    # Realistic-ish scale so reflections produce clearly non-zero RMSD.
    return rng.standard_normal((20, 3)) * 5.0


def test_identity(cloud):
    assert _ca_rmsd(cloud, cloud) == pytest.approx(0.0, abs=1e-6)
    assert _ca_rmsd_with_reflection(cloud, cloud) == pytest.approx(0.0, abs=1e-6)
    assert _chirality_sign(cloud, cloud) == 1


def test_translation_invariance(cloud, rng):
    shift = rng.standard_normal(3) * 10.0
    shifted = cloud + shift
    assert _ca_rmsd(shifted, cloud) == pytest.approx(0.0, abs=1e-6)
    assert _ca_rmsd_with_reflection(shifted, cloud) == pytest.approx(0.0, abs=1e-6)
    assert _chirality_sign(shifted, cloud) == 1


def test_proper_rotation_invariance(cloud):
    R = Rotation.random(random_state=42).as_matrix()
    assert np.isclose(np.linalg.det(R), 1.0)
    rotated = cloud @ R.T
    assert _ca_rmsd(rotated, cloud) == pytest.approx(0.0, abs=1e-6)
    assert _chirality_sign(rotated, cloud) == 1


def test_mirror_detection(cloud):
    Q = np.diag([1.0, 1.0, -1.0])
    mirrored = cloud @ Q  # == Q @ cloud since Q is diagonal/symmetric
    assert _chirality_sign(mirrored, cloud) == -1
    assert _ca_rmsd_with_reflection(mirrored, cloud) == pytest.approx(0.0, abs=1e-6)
    assert _ca_rmsd(mirrored, cloud) > 0.1


def test_rotation_composed_with_reflection(cloud):
    Q = np.diag([1.0, 1.0, -1.0])
    R = Rotation.random(random_state=7).as_matrix()
    RQ = R @ Q  # det = -1
    assert np.isclose(np.linalg.det(RQ), -1.0)
    mirrored = cloud @ RQ.T
    assert _chirality_sign(mirrored, cloud) == -1
    assert _ca_rmsd_with_reflection(mirrored, cloud) == pytest.approx(0.0, abs=1e-6)


def test_two_random_clouds_sanity(rng):
    a = rng.standard_normal((20, 3)) * 5.0
    b = rng.standard_normal((20, 3)) * 5.0
    r = _ca_rmsd(a, b)
    r_refl = _ca_rmsd_with_reflection(a, b)
    assert np.isfinite(r) and r >= 0
    assert np.isfinite(r_refl) and r_refl >= 0
    assert r_refl <= r + 1e-6


# ----------------------------------------------------------------------------
# _fm_loss_mirror_diagnostic with fake pl_module
# ----------------------------------------------------------------------------

class _FakeFM:
    """Minimal stand-in for the FM module used by _fm_loss_mirror_diagnostic."""

    @staticmethod
    def _mask_and_zero_com(x, mask):
        # x: [B, n, 3], mask: [B, n] bool
        m = mask.unsqueeze(-1).to(x.dtype)
        com = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(min=1)
        return (x - com) * m

    @staticmethod
    def sample_reference(n, shape, device, dtype, mask):
        # Return [*shape, n, 3] isotropic Gaussian, zero-COM'd.
        full_shape = (*shape, n, 3)
        x = torch.randn(full_shape, device=device, dtype=dtype)
        return x - x.mean(dim=-2, keepdim=True)


class _EquivariantNN:
    """coors_pred = alpha * x_t — commutes with any orthogonal transform."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def __call__(self, batch_nn):
        return {"coors_pred": self.alpha * batch_nn["x_t"]}


class _NonEquivariantNN:
    """coors_pred = x_t @ M for a fixed non-orthogonal M that mixes axes.

    A diagonal scaling like diag(1,2,3) would *still* give equal squared-loss
    under Q=diag(1,1,-1) because the z-component flip is squared away. Mixing
    axes (and including a reflection-asymmetric off-diagonal term) breaks this.
    """

    def __call__(self, batch_nn):
        x = batch_nn["x_t"]
        M = torch.tensor(
            [[1.0, 0.5, 0.0],
             [0.0, 2.0, 0.7],
             [0.3, 0.0, 3.0]],
            device=x.device, dtype=x.dtype,
        )
        return {"coors_pred": x @ M}


class _FakePLModule:
    def __init__(self, nn_module):
        self.nn = nn_module
        self.fm = _FakeFM()
        self.device = torch.device("cpu")


def _make_callback_with_gt(gt_ca: np.ndarray) -> ProteinEvalCallback:
    cb = ProteinEvalCallback(
        eval_every_n_steps=1,
        n_residues=gt_ca.shape[0],
        run_name="test_chirality",
        ground_truth_pdb_path="/dev/null",  # non-empty; bypassed below
    )
    # Shadow the bound method with a callable attribute returning fixed coords.
    cb._get_gt_ca_coords = lambda: gt_ca
    return cb


def test_fm_loss_mirror_diagnostic_equivariant_is_zero():
    torch.manual_seed(0)
    gt_ca = np.random.default_rng(1).standard_normal((16, 3)).astype(np.float32) * 5.0
    cb = _make_callback_with_gt(gt_ca)
    pl_module = _FakePLModule(_EquivariantNN(alpha=0.3))

    loss_orig, loss_mirror, abs_diff = cb._fm_loss_mirror_diagnostic(
        pl_module, n_noise_samples=4
    )
    assert loss_orig is not None
    assert abs_diff < 1e-5, (
        f"Expected O(3)-equivariant nn to produce identical FM loss under "
        f"reflection, got orig={loss_orig}, mirror={loss_mirror}, "
        f"abs_diff={abs_diff}"
    )


def test_fm_loss_mirror_diagnostic_non_equivariant_is_nonzero():
    torch.manual_seed(0)
    gt_ca = np.random.default_rng(2).standard_normal((16, 3)).astype(np.float32) * 5.0
    cb = _make_callback_with_gt(gt_ca)
    pl_module = _FakePLModule(_NonEquivariantNN())

    loss_orig, loss_mirror, abs_diff = cb._fm_loss_mirror_diagnostic(
        pl_module, n_noise_samples=4
    )
    assert loss_orig is not None
    assert abs_diff > 0.01, (
        f"Expected non-equivariant nn to produce different FM loss under "
        f"reflection, got abs_diff={abs_diff}"
    )
