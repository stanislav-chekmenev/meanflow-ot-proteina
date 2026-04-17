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
    # Use a mirrored prediction so the full-mask loss is non-zero.
    Q = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    x_mirror = random_ca @ Q

    mask = torch.zeros(random_ca.shape[:2], dtype=torch.bool)
    mask[:, : random_ca.shape[1] // 2] = True  # first half valid only

    loss_partial = chirality_hinge_loss(
        x_pred=x_mirror,
        x_gt=random_ca,
        mask=mask,
        margin_alpha=0.1,
        stride=1,
    )
    loss_full = chirality_hinge_loss(
        x_pred=x_mirror,
        x_gt=random_ca,
        mask=torch.ones_like(mask),
        margin_alpha=0.1,
        stride=1,
    )
    # Both non-zero, but partial < full because fewer residues contribute.
    assert loss_full.item() > 0.0
    assert loss_partial.item() > 0.0
    assert loss_partial.item() < loss_full.item()


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
