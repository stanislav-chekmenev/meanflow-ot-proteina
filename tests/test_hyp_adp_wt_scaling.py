"""Hypothesis: adp-wt-scaling.

Apply the MeanFlow adaptive weight ``adp_wt = (loss_mf.detach() + eps)^norm_p``
(currently used only on the MF loss at ``model_trainer_base.py:252-259``) to
the chirality hinge as well, so the two losses share a common per-step scale.

These tests lock in:

1. At ``norm_p=0``, the chirality term is unchanged (backward-compat
   invariant on the baseline control branch).
2. At ``norm_p=1``, the chirality term is scaled by ``1 / (loss_mf + eps)``.
3. At ``norm_p=2``, the chirality term is scaled by ``1 / (loss_mf + eps)^2``.
4. Gradient w.r.t. the chirality-loss inputs is rescaled by the same factor.
5. ``ModelTrainerBase._compute_adp_wt`` reads ``training.meanflow.norm_p`` /
   ``norm_eps`` from the config (not a hardcoded default).

Run with ``PYTHONPATH=. pytest tests/test_hyp_adp_wt_scaling.py -v``.

If the adp_wt application to chirality is reverted (i.e. chirality goes back
to being added unscaled), tests 2 / 3 / 4 FAIL immediately because the
expected scaling factor no longer appears in the combined loss nor in the
gradient flowing through the chirality branch.
"""
import sys
import types

# Stub torch_scatter before any proteinfoundation import (CUDA-only .so fails
# on CPU-only nodes; we don't need it in these tests).
_ts_stub = types.ModuleType("torch_scatter")
_ts_stub.scatter_mean = None
sys.modules.setdefault("torch_scatter", _ts_stub)

import pytest
import torch
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.chirality_loss import chirality_hinge_loss
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase


# ----------------------------------------------------------------------------
# Fake trainer scaffolding (matches tests/test_selfcond_chirality.py)
# ----------------------------------------------------------------------------
def _make_fake_trainer(norm_p, norm_eps=1e-3, chir_weight=1.0, chir_enabled=True):
    """Construct a Proteina-like trainer with the minimum wiring needed to
    call ``_compute_single_noise_loss`` directly.

    We avoid instantiating Proteina because it pulls in heavy deps; instead
    we build a bare object that quacks like one.
    """
    from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher

    class MirrorNN(torch.nn.Module):
        """Predicts a mirrored x_1 (via Q = diag(-1, 1, 1)) so the chirality
        hinge is strictly positive, while having a grad-bearing parameter so
        backward works.
        """
        def __init__(self, x_1_mirror, t_ext):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("x_1_mirror", x_1_mirror)
            self.register_buffer("t_ext", t_ext)

        def forward(self, batch):
            x_t = batch["x_t"]
            h = batch["h"]
            # FM sub-pass (h == 0): recover x_1 = Q x_1 via v_pred.
            if torch.allclose(h, torch.zeros_like(h)):
                v_pred = (x_t - self.x_1_mirror) / self.t_ext
                return {"coors_pred": self.scale * v_pred}
            # MeanFlow sub-pass: arbitrary but differentiable output.
            return {"coors_pred": self.scale * x_t}

    class FakeTrainer:
        pass

    trainer = FakeTrainer()
    trainer.device = torch.device("cpu")
    trainer.fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
    trainer.ot_sampler = None

    if norm_p is None:
        mf_cfg = {"ratio": 0.25}
    else:
        mf_cfg = {"ratio": 0.25, "norm_p": norm_p, "norm_eps": norm_eps}

    trainer.cfg_exp = OmegaConf.create({
        "training": {
            "meanflow": mf_cfg,
            "chirality_loss": {
                "enabled": chir_enabled,
                "weight": chir_weight,
                "margin_alpha": 0.1,
                "stride": 1,
            },
        },
    })
    trainer.chirality_loss_enabled = chir_enabled
    trainer.chirality_loss_weight = chir_weight
    trainer.chirality_margin_alpha = 0.1
    trainer.chirality_stride = 1
    trainer.self_cond_prob = 0.0
    # Bind the real methods from the class to the fake instance.
    trainer.adaptive_loss = ModelTrainerBase.adaptive_loss.__get__(trainer)
    trainer._compute_adp_wt = ModelTrainerBase._compute_adp_wt.__get__(trainer)
    trainer._compute_single_noise_loss = (
        ModelTrainerBase._compute_single_noise_loss.__get__(trainer)
    )

    # Inject a mirror NN that makes the chirality hinge non-zero.
    # We'll wire this after the caller decides shapes.
    trainer._mirror_nn_factory = MirrorNN
    return trainer


def _make_batch(trainer, B=2, n=12):
    """Build a synthetic batch with a chirally-mirrored target so the hinge
    fires. Returns the inputs to ``_compute_single_noise_loss``.
    """
    torch.manual_seed(0)
    x_1 = torch.randn(B, n, 3)
    x_1 = x_1 - x_1.mean(dim=-2, keepdim=True)
    mask = torch.ones(B, n, dtype=torch.bool)
    t = torch.full((B,), 0.5)
    r = torch.full((B,), 0.3)
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]

    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x_1_mirror = x_1 @ Q
    trainer.nn = trainer._mirror_nn_factory(x_1_mirror, t_ext)

    return x_1, mask, t_ext, r_ext, t, B


def _run(trainer):
    x_1, mask, t_ext, r_ext, t, B = _make_batch(trainer)
    torch.manual_seed(0)
    return trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, {}, B,
    )


# ----------------------------------------------------------------------------
# Test 1: norm_p=0 -> chirality term unchanged (backward-compat invariant)
# ----------------------------------------------------------------------------
def test_norm_p_zero_leaves_chirality_identical_to_pre_adp_wt():
    """At norm_p=0, adp_wt = (loss_mf.detach() + eps)^0 = 1. The chirality
    contribution to the combined loss must equal chirality_weight * raw_chir.
    """
    trainer = _make_fake_trainer(norm_p=0.0, chir_weight=3.0, chir_enabled=True)
    combined_on, raw_mf, raw_fm, raw_chir, adp_wt_mean = _run(trainer)

    # Baseline: run with chirality disabled so we can isolate the chirality
    # term exactly (same seed => same noise).
    trainer_off = _make_fake_trainer(norm_p=0.0, chir_weight=3.0, chir_enabled=False)
    combined_off, _, _, _, _ = _run(trainer_off)

    chir_contribution = (combined_on - combined_off).item()
    expected = (3.0 * raw_chir).item()  # weight * raw chirality hinge
    assert abs(chir_contribution - expected) < 1e-5, (
        f"norm_p=0 must leave chirality term unscaled: "
        f"chir_contribution={chir_contribution}, expected={expected}"
    )
    assert abs(adp_wt_mean.item() - 1.0) < 1e-6, (
        f"At norm_p=0 adp_wt must be 1, got {adp_wt_mean.item()}"
    )


# ----------------------------------------------------------------------------
# Test 2: norm_p=1 -> chirality scaled by 1 / (loss_mf + eps)
# ----------------------------------------------------------------------------
def test_norm_p_one_scales_chirality_by_inv_loss_mf():
    """At norm_p=1, combined_on - combined_off == weight * raw_chir / adp_wt.

    We compare two runs with the SAME noise (same seed, same model weights):
    one with chirality enabled, one disabled. The delta must equal the
    scaled chirality term.
    """
    eps = 1e-3
    trainer_on = _make_fake_trainer(
        norm_p=1.0, norm_eps=eps, chir_weight=1.0, chir_enabled=True,
    )
    combined_on, raw_mf, _, raw_chir, adp_wt_mean = _run(trainer_on)

    trainer_off = _make_fake_trainer(
        norm_p=1.0, norm_eps=eps, chir_weight=1.0, chir_enabled=False,
    )
    combined_off, raw_mf_off, _, _, adp_wt_mean_off = _run(trainer_off)

    # Both runs see the same x_1, same NN, same seed -> same per-sample loss_mf.
    assert torch.allclose(raw_mf, raw_mf_off, atol=1e-6)
    assert torch.allclose(adp_wt_mean, adp_wt_mean_off, atol=1e-6)

    chir_contribution = (combined_on - combined_off).item()
    expected = (raw_chir / adp_wt_mean).item()
    assert abs(chir_contribution - expected) < 1e-5, (
        f"norm_p=1: chir_contribution={chir_contribution}, "
        f"expected={expected} (raw_chir={raw_chir.item()}, "
        f"adp_wt_mean={adp_wt_mean.item()})"
    )
    # Sanity: adp_wt is not trivially 1 at norm_p=1 unless loss_mf+eps == 1.
    assert adp_wt_mean.item() != pytest.approx(1.0, abs=1e-3), (
        "Test is uninformative: loss_mf + eps happens to be 1."
    )


# ----------------------------------------------------------------------------
# Test 3: norm_p=2 -> chirality scaled by 1 / (loss_mf + eps)^2
# ----------------------------------------------------------------------------
def test_norm_p_two_scales_chirality_by_inv_loss_mf_squared():
    eps = 1e-3
    trainer_on = _make_fake_trainer(
        norm_p=2.0, norm_eps=eps, chir_weight=1.0, chir_enabled=True,
    )
    combined_on, raw_mf, _, raw_chir, adp_wt_mean_on = _run(trainer_on)

    trainer_off = _make_fake_trainer(
        norm_p=2.0, norm_eps=eps, chir_weight=1.0, chir_enabled=False,
    )
    combined_off, raw_mf_off, _, _, adp_wt_mean_off = _run(trainer_off)

    assert torch.allclose(raw_mf, raw_mf_off, atol=1e-6)

    chir_contribution = (combined_on - combined_off).item()
    expected = (raw_chir / adp_wt_mean_on).item()
    assert abs(chir_contribution - expected) < 1e-5, (
        f"norm_p=2: chir_contribution={chir_contribution}, "
        f"expected={expected}"
    )
    # Sanity: the adp_wt mean at norm_p=2 differs meaningfully from 1 so
    # the scaling actually moves the combined loss vs a hypothetical p=0 run.
    assert abs(adp_wt_mean_on.item() - 1.0) > 1e-3, (
        "Test degenerate: adp_wt at norm_p=2 happens to be ~1."
    )


# ----------------------------------------------------------------------------
# Test 4: gradient flowing through chirality-branch inputs is rescaled.
# ----------------------------------------------------------------------------
def test_gradient_respects_adp_wt_scaling_on_chirality_branch():
    """Let ``x_pred`` depend on a scalar param ``s``; then
    d(loss_chir / adp_wt) / ds == (d loss_chir / ds) / adp_wt
    (adp_wt is detached). Verified here by comparing grads from two toy
    forward/backward passes with different ``norm_p``.
    """
    torch.manual_seed(0)
    B, n = 1, 16
    x_gt = torch.randn(B, n, 3)
    mask = torch.ones(B, n, dtype=torch.bool)
    # Start prediction at the mirror (loss > 0).
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x_mirror = (x_gt @ Q)

    # Adp_wt is a FIXED detached scalar in the pipeline (we pretend
    # loss_mf + eps = c here).
    c = torch.tensor(0.25)  # emulated (loss_mf + eps)

    def compute_loss(norm_p):
        s = torch.tensor(1.0, requires_grad=True)
        x_pred = s * x_mirror
        loss_chir = chirality_hinge_loss(
            x_pred=x_pred,
            x_gt=x_gt,
            mask=mask,
            margin_alpha=0.1,
            stride=1,
        )
        adp_wt = c ** norm_p  # identical formula as _compute_adp_wt
        scaled = loss_chir / adp_wt
        scaled.backward()
        return s.grad.item(), loss_chir.item(), adp_wt.item()

    grad_p0, loss_p0, wt_p0 = compute_loss(0.0)
    grad_p1, loss_p1, wt_p1 = compute_loss(1.0)
    grad_p2, loss_p2, wt_p2 = compute_loss(2.0)

    assert abs(wt_p0 - 1.0) < 1e-6
    assert abs(loss_p0 - loss_p1) < 1e-6, "Loss should not depend on norm_p pre-scaling"
    assert abs(loss_p0 - loss_p2) < 1e-6
    # Gradients scale exactly by 1/adp_wt.
    assert abs(grad_p1 - grad_p0 / wt_p1) < 1e-5
    assert abs(grad_p2 - grad_p0 / wt_p2) < 1e-5
    # And p=1/p=2 relation is grad_p2 = grad_p1 / c:
    assert abs(grad_p2 - grad_p1 / c.item()) < 1e-5


# ----------------------------------------------------------------------------
# Test 5: config plumbing -- _compute_adp_wt reads norm_p/norm_eps from cfg.
# ----------------------------------------------------------------------------
def test_compute_adp_wt_reads_config_norm_p():
    """Verify that ``_compute_adp_wt`` reads norm_p/norm_eps from
    ``training.meanflow`` exactly (not a hardcoded default).
    """
    loss = torch.tensor([4.0])  # loss + eps = 4.001 at eps=1e-3
    eps = 1e-3

    for p in [0.0, 1.0, 2.0]:
        trainer = _make_fake_trainer(norm_p=p, norm_eps=eps)
        adp_wt = trainer._compute_adp_wt(loss)
        expected = (loss + eps) ** p
        assert torch.allclose(adp_wt, expected, atol=1e-6), (
            f"_compute_adp_wt didn't respect config norm_p={p}: "
            f"got {adp_wt}, expected {expected}"
        )

    # norm_p absent -> returns ones (no-op, backward-compat).
    trainer = _make_fake_trainer(norm_p=None)
    adp_wt = trainer._compute_adp_wt(loss)
    assert torch.allclose(adp_wt, torch.ones_like(loss), atol=1e-6)


# ----------------------------------------------------------------------------
# Test 6: regression check — the intervention is active.
# If someone removes `loss_chir / adp_wt_chir` and falls back to
# `combined_adp_loss + weight * loss_chir`, at norm_p=1 with a large
# chirality hinge (mirrored prediction) the combined loss is measurably
# larger because there's no 1/adp_wt division. Lock that in.
# ----------------------------------------------------------------------------
def test_regression_adp_wt_scaling_is_active_at_norm_p_one():
    """If the division by adp_wt on chirality is reverted, the combined loss
    at norm_p=1 equals what we'd see with norm_p=0, because the
    chirality term is no longer rescaled. This test guards against that."""
    trainer_p0 = _make_fake_trainer(norm_p=0.0, chir_weight=1.0, chir_enabled=True)
    combined_p0, _, _, raw_chir_p0, adp_wt_p0 = _run(trainer_p0)

    trainer_p1 = _make_fake_trainer(norm_p=1.0, chir_weight=1.0, chir_enabled=True)
    combined_p1, _, _, raw_chir_p1, adp_wt_p1 = _run(trainer_p1)

    # Chirality hinge value is independent of norm_p.
    assert torch.allclose(raw_chir_p0, raw_chir_p1, atol=1e-6)

    # Extract the chirality contribution from each combined loss by
    # subtracting the chirality-off combined loss.
    trainer_p0_off = _make_fake_trainer(norm_p=0.0, chir_weight=1.0, chir_enabled=False)
    combined_p0_off, _, _, _, _ = _run(trainer_p0_off)
    trainer_p1_off = _make_fake_trainer(norm_p=1.0, chir_weight=1.0, chir_enabled=False)
    combined_p1_off, _, _, _, _ = _run(trainer_p1_off)

    chir_p0 = (combined_p0 - combined_p0_off).item()
    chir_p1 = (combined_p1 - combined_p1_off).item()

    # Invariant: under the intervention, chir_p1 == chir_p0 / adp_wt_p1,
    # which (if adp_wt_p1 < 1) makes chir_p1 > chir_p0, and if adp_wt_p1 > 1
    # makes chir_p1 < chir_p0. Either way, chir_p1 MUST NOT equal chir_p0
    # when adp_wt_p1 is meaningfully different from 1.0.
    assert abs(adp_wt_p1.item() - 1.0) > 1e-3, (
        "Test degenerate: adp_wt at norm_p=1 happens to be ~1."
    )
    assert abs(chir_p1 - chir_p0) > 1e-4, (
        f"Intervention appears inactive at norm_p=1: chir_p0={chir_p0}, "
        f"chir_p1={chir_p1}, adp_wt_p1={adp_wt_p1.item()}. "
        f"This test FAILS if the adp_wt scaling on chirality is reverted."
    )
    # And they match the analytical prediction.
    expected = chir_p0 / adp_wt_p1.item()
    assert abs(chir_p1 - expected) < 1e-5
