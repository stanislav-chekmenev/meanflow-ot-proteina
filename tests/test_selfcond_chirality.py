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


# ----------------------------------------------------------------------------
# Trainer plumbing tests
# ----------------------------------------------------------------------------

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
            self.calls.append(dict(batch))
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
    # Default t-gate attributes: identity gate (backward-compat).
    trainer.chirality_t_gate_max = 1.0
    trainer.chirality_t_gate_mode = "hard"
    trainer.self_cond_prob = 0.5
    # Bind the real method from the class (not the instance).
    from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
    trainer.adaptive_loss = ModelTrainerBase.adaptive_loss.__get__(trainer)
    trainer._compute_adp_wt = ModelTrainerBase._compute_adp_wt.__get__(trainer)
    trainer._compute_single_noise_loss = (
        ModelTrainerBase._compute_single_noise_loss.__get__(trainer)
    )
    return trainer


def test_chirality_loss_added_when_enabled():
    """The chirality branch must actually fire when enabled: given a
    prediction whose implied x_1_pred is the mirror of x_1 (Q = diag(-1,1,1)),
    the hinge loss is strictly positive and the combined loss must exceed
    the disabled case (noise seeded identically in both runs).
    """
    trainer = _make_fake_trainer()
    trainer.chirality_loss_weight = 10.0

    B, n = 1, 12
    torch.manual_seed(0)
    x_1 = torch.randn(B, n, 3)
    # Zero COM so the trainer's mask_and_zero_com doesn't perturb our
    # mirror construction through the FakeNN.
    x_1 = x_1 - x_1.mean(dim=-2, keepdim=True)
    mask = torch.ones(B, n, dtype=torch.bool)
    t = torch.full((B,), 0.5)
    r = torch.full((B,), 0.3)
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]
    batch = {}

    # Replace the trainer's NN with one that, on the FM sub-pass (h=0, i.e.
    # r=t), returns v_pred = (x_t - Q @ x_1) / t so that the recovered
    # x_1_pred = x_t - t * v_pred equals Q @ x_1 (a mirrored copy). That
    # drives signed_agreement = -T_gt^2 <= 0 everywhere, hence a strictly
    # positive hinge loss with margin_alpha = 0.1.
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x_1_mirror = x_1 @ Q

    class MirrorNN(torch.nn.Module):
        def __init__(self, x_1_mirror, t_ext):
            super().__init__()
            # Trainable param so the module has grad, matching FakeNN.
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("x_1_mirror", x_1_mirror)
            self.register_buffer("t_ext", t_ext)

        def forward(self, batch):
            x_t = batch["x_t"]
            h = batch["h"]
            # FM sub-pass uses h == 0 (r == t). Return v_pred so that
            # x_1_pred = x_t - t * v_pred == Q @ x_1 exactly. Multiply by
            # self.scale (which starts at 1.0) so backward still has grad.
            if torch.allclose(h, torch.zeros_like(h)):
                v_pred = (x_t - self.x_1_mirror) / self.t_ext
                return {"coors_pred": self.scale * v_pred}
            # MeanFlow sub-pass (h != 0): return something arbitrary but
            # differentiable through the NN params.
            return {"coors_pred": self.scale * x_t}

    trainer.nn = MirrorNN(x_1_mirror, t_ext)

    # --- Enabled run ---
    trainer.chirality_loss_enabled = True
    torch.manual_seed(0)
    loss_enabled, _, _, raw_loss_chir, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B,
    )

    # --- Disabled run (same NN, same seed => same noise sample) ---
    trainer.chirality_loss_enabled = False
    torch.manual_seed(0)
    loss_disabled, _, _, raw_loss_chir_off, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B,
    )

    assert torch.isfinite(loss_enabled)
    assert torch.isfinite(loss_disabled)
    # Chirality branch must fire when enabled and be zero when disabled.
    assert raw_loss_chir.item() > 0.0
    assert raw_loss_chir_off.item() == 0.0
    # Adding a positive hinge term with weight 10.0 strictly increases the
    # combined loss relative to the disabled run.
    assert loss_enabled.item() > loss_disabled.item()


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
    loss, _, _, _, _ = trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B, use_sc=False,
    )
    n_calls_off = len(trainer.nn.calls)
    assert n_calls_off == 2, f"use_sc=False expected 2 NN calls, got {n_calls_off}"
    for c in trainer.nn.calls:
        assert "x_sc" not in c

    trainer.nn.calls.clear()

    # use_sc=True: one extra warmup call; the JVP and FM calls must receive x_sc.
    loss, _, _, _, _ = trainer._compute_single_noise_loss(
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
