"""Tests for hyp/t-gate-chir: t-gate on the chirality hinge loss.

Exercises the mechanistic invariants promised by the hypothesis:

1. Hard gate: with t > t_gate_max on all samples, chirality contribution is 0.
   With t < t_gate_max on all samples, chirality contribution equals the
   ungated value.
2. Soft gate: sigmoid((t_gate_max - t) / t_sharpness). Far above/below
   t_gate_max the contribution is ~0 / ~ungated.
3. Default (t_gate_max=1.0, hard): identity gate, backward-compat.
4. Config plumbing: Proteina constructor reads `t_gate_max` / `t_gate_mode`
   from `training.chirality_loss` and stores them as model attributes.
5. MeanFlow loss is NOT gated: gating chirality off leaves MF loss unchanged.

Runnable with: PYTHONPATH=. pytest tests/test_hyp_t_gate_chir.py -v
"""
import importlib.machinery
import sys
import types


def _stub_module(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m
    return m


# torch_scatter (CUDA-only .so) and a couple of pandas/biotite placeholders
# so pure-Python imports of proteinfoundation work on CPU-only CI.
_ts_stub = _stub_module("torch_scatter")
_ts_stub.scatter_mean = None
if "pandas" not in sys.modules or getattr(sys.modules["pandas"], "__spec__", None) is None:
    _stub_module("pandas")
for _bio_name in ["biotite", "biotite.structure", "biotite.structure.io"]:
    if _bio_name not in sys.modules:
        _stub_module(_bio_name)
sys.modules["biotite"].structure = sys.modules["biotite.structure"]
sys.modules["biotite.structure"].io = sys.modules["biotite.structure.io"]


import pytest
import torch
from omegaconf import OmegaConf


# ----------------------------------------------------------------------------
# Shared fixtures: build a bare Proteina-like trainer with a MirrorNN so the
# chirality hinge is strictly positive when fired. This mirrors the pattern in
# tests/test_selfcond_chirality.py::_make_fake_trainer.
# ----------------------------------------------------------------------------


def _make_mirror_trainer(B=4, n=12, t_gate_max=1.0, t_gate_mode="hard",
                        chirality_weight=10.0):
    from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
    from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase

    torch.manual_seed(0)
    x_1 = torch.randn(B, n, 3)
    x_1 = x_1 - x_1.mean(dim=-2, keepdim=True)
    mask = torch.ones(B, n, dtype=torch.bool)

    # Mirror: reflection across the x-axis.
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x_1_mirror = x_1 @ Q

    class MirrorNN(torch.nn.Module):
        """Returns v_pred so that x_1_pred = z - t*v_pred = Q @ x_1 on FM pass."""

        def __init__(self, x_1_mirror):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("x_1_mirror", x_1_mirror)

        def forward(self, batch):
            x_t = batch["x_t"]
            h = batch["h"]
            t_per_sample = batch["t"]  # [B]
            t_ext = t_per_sample[..., None, None]  # [B,1,1]
            if torch.allclose(h, torch.zeros_like(h)):
                # FM sub-pass: recover x_1 exactly at mirror.
                v_pred = (x_t - self.x_1_mirror) / t_ext.clamp(min=1e-6)
                return {"coors_pred": self.scale * v_pred}
            # JVP sub-pass: arbitrary but grad-connected.
            return {"coors_pred": self.scale * x_t}

    class FakeTrainer:
        pass

    trainer = FakeTrainer()
    trainer.device = torch.device("cpu")
    trainer.fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
    trainer.nn = MirrorNN(x_1_mirror)
    trainer.ot_sampler = None
    trainer.cfg_exp = OmegaConf.create({
        "training": {
            "meanflow": {"ratio": 0.25, "norm_p": 0.0, "norm_eps": 1e-3},
            "chirality_loss": {
                "enabled": True, "weight": chirality_weight,
                "margin_alpha": 0.1, "stride": 1,
                "t_gate_max": t_gate_max, "t_gate_mode": t_gate_mode,
            },
        },
    })
    trainer.meanflow_norm_p = 0.0
    trainer.meanflow_norm_eps = 1e-3
    trainer.chirality_loss_enabled = True
    trainer.chirality_loss_weight = chirality_weight
    trainer.chirality_margin_alpha = 0.1
    trainer.chirality_stride = 1
    trainer.chirality_t_gate_max = t_gate_max
    trainer.chirality_t_gate_mode = t_gate_mode
    trainer.self_cond_prob = 0.0

    # Bind real methods off the class so we hit the production code path.
    trainer.adaptive_loss = ModelTrainerBase.adaptive_loss.__get__(trainer)
    trainer._compute_single_noise_loss = (
        ModelTrainerBase._compute_single_noise_loss.__get__(trainer)
    )
    return trainer, x_1, mask


def _run_single_noise_loss(trainer, x_1, mask, t_values, r_values=None):
    B = x_1.shape[0]
    t = torch.as_tensor(t_values, dtype=torch.float32)
    if r_values is None:
        # Always pair with a smaller r so (t-r)>0 for MeanFlow sub-pass.
        r = torch.clamp(t - 0.1, min=0.0)
    else:
        r = torch.as_tensor(r_values, dtype=torch.float32)
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]
    torch.manual_seed(12345)
    return trainer._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch={}, B=B,
    )


# ----------------------------------------------------------------------------
# 1. Hard gate invariants.
# ----------------------------------------------------------------------------


def test_hard_gate_zeroes_chirality_when_all_t_above_cutoff():
    """With hard gate and every t_i > t_gate_max, raw_loss_chir is exactly 0."""
    trainer, x_1, mask = _make_mirror_trainer(t_gate_max=0.3, t_gate_mode="hard")
    B = x_1.shape[0]
    t_values = torch.full((B,), 0.9)  # all above 0.3

    _, _, _, raw_chir = _run_single_noise_loss(trainer, x_1, mask, t_values)
    assert raw_chir.item() == 0.0, (
        f"Hard gate with all t > t_gate_max should zero chirality; got {raw_chir.item()}"
    )


def test_hard_gate_passes_chirality_when_all_t_below_cutoff():
    """With hard gate and every t_i < t_gate_max, chirality contribution equals
    the ungated value (all per-sample gates = 1)."""
    B, n = 4, 12
    t_values = torch.full((B,), 0.1)  # all below 0.3

    gated_trainer, x_1, mask = _make_mirror_trainer(
        B=B, n=n, t_gate_max=0.3, t_gate_mode="hard")
    _, _, _, raw_chir_gated = _run_single_noise_loss(gated_trainer, x_1, mask, t_values)

    # Reference: direct per-sample helper on x_1_pred = Q @ x_1.
    from proteinfoundation.proteinflow.chirality_loss import (
        chirality_hinge_loss_per_sample,
    )
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    per_sample = chirality_hinge_loss_per_sample(
        x_pred=x_1 @ Q, x_gt=x_1, mask=mask, margin_alpha=0.1, stride=1,
    )
    expected = per_sample.mean().item()
    assert raw_chir_gated.item() > 0.0, "Mirror prediction should give positive chirality"
    assert raw_chir_gated.item() == pytest.approx(expected, rel=1e-5, abs=1e-7), (
        f"Hard gate with all t < t_gate_max should equal ungated value; "
        f"got gated={raw_chir_gated.item()}, expected={expected}"
    )


def test_hard_gate_mixed_t_keeps_only_below_samples():
    """Mixed batch: per-sample gate must zero out only the above-cutoff samples.
    Gated mean must exactly equal (sum of kept per-sample hinges) / B."""
    B, n = 4, 12
    # samples 0,1 below cutoff; 2,3 above
    t_values = torch.tensor([0.1, 0.2, 0.5, 0.8])

    gated_trainer, x_1, mask = _make_mirror_trainer(
        B=B, n=n, t_gate_max=0.3, t_gate_mode="hard")
    _, _, _, raw_chir_gated = _run_single_noise_loss(gated_trainer, x_1, mask, t_values)

    # Expected: compute per-sample hinge directly on x_1_pred = Q @ x_1 (what
    # the MirrorNN produces regardless of x_0). Apply gate [1,1,0,0], mean / B.
    from proteinfoundation.proteinflow.chirality_loss import (
        chirality_hinge_loss_per_sample,
    )
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x_1_pred = x_1 @ Q
    per_sample = chirality_hinge_loss_per_sample(
        x_pred=x_1_pred, x_gt=x_1, mask=mask, margin_alpha=0.1, stride=1,
    )
    gate = (t_values < 0.3).to(per_sample.dtype)
    expected = (per_sample * gate).mean().item()
    assert raw_chir_gated.item() == pytest.approx(expected, rel=1e-5, abs=1e-6), (
        f"Gated mean should equal per_sample*gate mean; got {raw_chir_gated.item()} "
        f"vs expected {expected}"
    )
    assert gate.sum().item() == 2  # sanity: two samples kept


# ----------------------------------------------------------------------------
# 2. Soft gate invariants.
# ----------------------------------------------------------------------------


def test_soft_gate_near_zero_for_large_t():
    """Soft gate with t_gate_max=0.1 and t=0.9 drives sigmoid((0.1-0.9)/0.1) =
    sigmoid(-8) ~= 3.35e-4 per-sample, so the chirality contribution is
    squashed to <1% of the ungated value."""
    B, n = 4, 12
    t_values = torch.full((B,), 0.9)

    trainer, x_1, mask = _make_mirror_trainer(
        B=B, n=n, t_gate_max=0.1, t_gate_mode="soft")
    _, _, _, raw_chir = _run_single_noise_loss(trainer, x_1, mask, t_values)

    # Independent reference: sum of per-sample hinges / B (ungated).
    from proteinfoundation.proteinflow.chirality_loss import (
        chirality_hinge_loss_per_sample,
    )
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    per_sample = chirality_hinge_loss_per_sample(
        x_pred=x_1 @ Q, x_gt=x_1, mask=mask, margin_alpha=0.1, stride=1,
    )
    expected_ungated = per_sample.mean().item()
    # sigmoid((0.1 - 0.9) / 0.1) = sigmoid(-8) ~= 3.35e-4
    sigmoid_val = torch.sigmoid(torch.tensor((0.1 - 0.9) / 0.1)).item()
    expected_gated = expected_ungated * sigmoid_val

    assert expected_ungated > 0.0
    assert raw_chir.item() == pytest.approx(expected_gated, rel=1e-4, abs=1e-7), (
        f"Soft gate at t=9*t_gate_max should give sigmoid(-8)*ungated; "
        f"got gated={raw_chir.item()}, expected={expected_gated}"
    )
    # Sanity on squash ratio.
    assert raw_chir.item() < 0.01 * expected_ungated


def test_soft_gate_near_full_for_small_t():
    """Soft gate with t_gate_max=1.0 and t=0.01 keeps sigmoid((1.0-0.01)/1.0)
    = sigmoid(0.99) ~= 0.7291, so chirality ≈ 0.73 * ungated."""
    B, n = 4, 12
    t_values = torch.full((B,), 0.01)

    trainer, x_1, mask = _make_mirror_trainer(
        B=B, n=n, t_gate_max=1.0, t_gate_mode="soft")
    _, _, _, raw_chir = _run_single_noise_loss(trainer, x_1, mask, t_values)

    from proteinfoundation.proteinflow.chirality_loss import (
        chirality_hinge_loss_per_sample,
    )
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    per_sample = chirality_hinge_loss_per_sample(
        x_pred=x_1 @ Q, x_gt=x_1, mask=mask, margin_alpha=0.1, stride=1,
    )
    expected_ungated = per_sample.mean().item()
    sigmoid_val = torch.sigmoid(torch.tensor(0.99)).item()
    expected_gated = expected_ungated * sigmoid_val

    assert raw_chir.item() == pytest.approx(expected_gated, rel=1e-4, abs=1e-7), (
        f"Soft gate at t << t_gate_max should give sigmoid(0.99)*ungated; "
        f"got gated={raw_chir.item()}, expected={expected_gated}"
    )
    assert raw_chir.item() > 0.5 * expected_ungated


# ----------------------------------------------------------------------------
# 3. Default gate is identity (backward compat).
# ----------------------------------------------------------------------------


def test_default_gate_is_identity():
    """With t_gate_max=1.0 (default) and hard mode, chirality contribution is
    exactly the same as if there were no gate code at all — since (t < 1.0) is
    True for any valid t sampled from [0, 1)."""
    B, n = 4, 12
    t_values = torch.tensor([0.05, 0.2, 0.5, 0.9])

    default_trainer, x_1, mask = _make_mirror_trainer(
        B=B, n=n, t_gate_max=1.0, t_gate_mode="hard")
    _, _, _, raw_default = _run_single_noise_loss(default_trainer, x_1, mask, t_values)

    # Sanity: positive (mirror predictions)
    assert raw_default.item() > 0.0

    # Compute the expected ungated per-sample mean using the helper directly.
    # The FM sub-pass in _compute_single_noise_loss reproduces x_1_pred =
    # z - t_ext * v_pred; with our MirrorNN this equals Q @ x_1 for every t.
    from proteinfoundation.proteinflow.chirality_loss import (
        chirality_hinge_loss_per_sample,
    )
    Q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x_1_pred = x_1 @ Q
    expected_per_sample = chirality_hinge_loss_per_sample(
        x_pred=x_1_pred, x_gt=x_1, mask=mask, margin_alpha=0.1, stride=1,
    )
    expected_mean = expected_per_sample.mean().item()
    assert raw_default.item() == pytest.approx(expected_mean, rel=1e-5, abs=1e-6), (
        f"Default gate (t_gate_max=1.0, hard) should equal ungated value; "
        f"got default={raw_default.item()}, expected={expected_mean}"
    )


# ----------------------------------------------------------------------------
# 4. Config plumbing in Proteina.__init__.
# ----------------------------------------------------------------------------


def test_config_plumbing_reads_t_gate_fields():
    """Proteina reads t_gate_max and t_gate_mode out of
    training.chirality_loss and stores them as attributes."""
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg = _build_minimal_proteina_cfg()
    cfg.training.chirality_loss.t_gate_max = 0.3
    cfg.training.chirality_loss.t_gate_mode = "soft"

    model = Proteina(cfg_exp=cfg)
    assert model.chirality_t_gate_max == 0.3
    assert model.chirality_t_gate_mode == "soft"


def test_config_plumbing_defaults_when_keys_absent():
    """When `t_gate_max`/`t_gate_mode` are not in the config, defaults apply
    (identity gate, hard mode) — ensures configs from before this branch still
    load without changes."""
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg = _build_minimal_proteina_cfg()
    # Remove the new keys if present.
    cl = cfg.training.chirality_loss
    if "t_gate_max" in cl:
        del cl.t_gate_max
    if "t_gate_mode" in cl:
        del cl.t_gate_mode

    model = Proteina(cfg_exp=cfg)
    assert model.chirality_t_gate_max == 1.0
    assert model.chirality_t_gate_mode == "hard"


# ----------------------------------------------------------------------------
# 5. MF loss is NOT gated: turning off chirality must leave MF unchanged.
# ----------------------------------------------------------------------------


def test_mf_loss_unchanged_when_chirality_disabled():
    """Gating the chirality hinge off by setting t_gate_max=0.0 (hard) must
    not perturb the MeanFlow / FM losses returned by _compute_single_noise_loss.
    This protects the MF gradient from leaking through the new code path."""
    B, n = 4, 12
    t_values = torch.full((B,), 0.5)

    # Run A: chirality enabled but gate fully OFF (t_gate_max=0, hard -> mask=False for all).
    trainer_a, x_1, mask = _make_mirror_trainer(
        B=B, n=n, t_gate_max=0.0, t_gate_mode="hard", chirality_weight=10.0)
    loss_a, raw_mf_a, raw_fm_a, raw_chir_a = _run_single_noise_loss(
        trainer_a, x_1, mask, t_values)

    # Run B: chirality weight=0 so chirality contribution is zero regardless.
    trainer_b, x_1_b, mask_b = _make_mirror_trainer(
        B=B, n=n, t_gate_max=1.0, t_gate_mode="hard", chirality_weight=0.0)
    trainer_b.chirality_loss_enabled = False  # further skip branch entirely
    loss_b, raw_mf_b, raw_fm_b, raw_chir_b = _run_single_noise_loss(
        trainer_b, x_1_b, mask_b, t_values)

    # Chirality contribution zero in both cases.
    assert raw_chir_a.item() == 0.0
    assert raw_chir_b.item() == 0.0
    # MF and FM losses should match exactly (same seed + identical NN weights).
    assert raw_mf_a.item() == pytest.approx(raw_mf_b.item(), rel=1e-5, abs=1e-6), (
        f"MF loss should be unaffected by the t-gate path when chirality is off; "
        f"gated-off MF={raw_mf_a.item()}, branch-off MF={raw_mf_b.item()}"
    )
    assert raw_fm_a.item() == pytest.approx(raw_fm_b.item(), rel=1e-5, abs=1e-6)
    # Combined loss should also match (no chirality term either way).
    assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-5, abs=1e-6)


# ----------------------------------------------------------------------------
# Proteina config helper
# ----------------------------------------------------------------------------


def _build_minimal_proteina_cfg():
    """Minimal Proteina config matching the tiny NN used in test_loss_accumulation.py."""
    return OmegaConf.create({
        "model": {
            "target_pred": "v",
            "augmentation": {"global_rotation": False, "naug_rot": 1},
            "nn": {
                "name": "ca_af3",
                "token_dim": 32,
                "nlayers": 1,
                "nheads": 2,
                "residual_mha": True,
                "residual_transition": True,
                "parallel_mha_transition": False,
                "use_attn_pair_bias": True,
                "strict_feats": False,
                "feats_init_seq": ["res_seq_pdb_idx", "chain_break_per_res"],
                "feats_cond_seq": ["time_emb", "delta_t_emb"],
                "t_emb_dim": 16,
                "dim_cond": 32,
                "idx_emb_dim": 16,
                "fold_emb_dim": 16,
                "feats_pair_repr": ["rel_seq_sep", "xt_pair_dists"],
                "feats_pair_cond": ["time_emb", "delta_t_emb"],
                "xt_pair_dist_dim": 8,
                "xt_pair_dist_min": 0.1,
                "xt_pair_dist_max": 3,
                "x_sc_pair_dist_dim": 8,
                "x_sc_pair_dist_min": 0.1,
                "x_sc_pair_dist_max": 3,
                "x_motif_pair_dist_dim": 8,
                "x_motif_pair_dist_min": 0.1,
                "x_motif_pair_dist_max": 3,
                "seq_sep_dim": 127,
                "pair_repr_dim": 16,
                "update_pair_repr": False,
                "update_pair_repr_every_n": 2,
                "use_tri_mult": False,
                "num_registers": 2,
                "use_qkln": True,
                "num_buckets_predict_pair": 8,
                "multilabel_mode": "sample",
                "cath_code_dir": ".",
            },
        },
        "loss": {
            "t_distribution": {"name": "uniform", "p1": 0.0, "p2": 1.0},
            "loss_t_clamp": 0.9,
            "use_aux_loss": False,
            "aux_loss_t_lim": 0.3,
            "thres_aux_2d_loss": 0.6,
            "aux_loss_weight": 1.0,
            "num_dist_buckets": 16,
            "max_dist_boundary": 1.0,
        },
        "training": {
            "loss_accumulation_steps": 1,
            "self_cond": False,
            "fold_cond": False,
            "mask_T_prob": 0.5,
            "mask_A_prob": 0.5,
            "mask_C_prob": 0.5,
            "motif_conditioning": False,
            "ot_coupling": {"enabled": False, "method": "exact", "reg": 0.05,
                            "reg_m": 1.0, "normalize_cost": False},
            "meanflow": {
                "ratio": 0.25, "P_mean": -0.4, "P_std": 1.0,
                "norm_p": 0.0, "norm_eps": 1e-3, "nsteps_sample": 1,
            },
            "chirality_loss": {
                "enabled": False, "weight": 1.0, "margin_alpha": 0.1, "stride": 1,
                "t_gate_max": 1.0, "t_gate_mode": "hard",
            },
        },
        "opt": {"lr": 1e-4, "accumulate_grad_batches": 1},
    })
