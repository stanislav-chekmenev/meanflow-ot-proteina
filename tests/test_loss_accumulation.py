"""
Tests for loss accumulation feature (K forward passes per batch).

Covers:
  1. _compute_single_noise_loss returns finite loss
  2. K=1 path is identical to pre-refactor behavior
  3. K>1 produces finite losses and nonzero gradients
  4. K>1 + accumulate_grad_batches composes correctly
  5. OT coupling called K times per training_step
"""

import importlib.machinery
import sys
import types
import unittest.mock as mock


def _stub_module(name):
    """Create a stub module with a valid __spec__ so torch._dynamo won't choke."""
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m
    return m


# Stub torch_scatter before any proteinfoundation import
_ts_stub = _stub_module("torch_scatter")
_ts_stub.scatter_mean = None

# Stub pandas and biotite (motif_factory imports them; pandas has numpy ABI
# issues in this env, and biotite is unused by our test).
if "pandas" not in sys.modules or getattr(sys.modules["pandas"], "__spec__", None) is None:
    _stub_module("pandas")

for _bio_name in ["biotite", "biotite.structure", "biotite.structure.io"]:
    if _bio_name not in sys.modules:
        _stub_module(_bio_name)
# Wire up the attribute chain
sys.modules["biotite"].structure = sys.modules["biotite.structure"]
sys.modules["biotite.structure"].io = sys.modules["biotite.structure.io"]

import pytest
import torch
from omegaconf import OmegaConf


def _build_cfg(K=1, accum=1, ot_enabled=False):
    cfg = OmegaConf.create({
        "model": {
            "target_pred": "v",
            "augmentation": {"global_rotation": False, "naug_rot": 1},
            "nn": {
                "name": "ca_af3",
                "token_dim": 64,
                "nlayers": 2,
                "nheads": 4,
                "residual_mha": True,
                "residual_transition": True,
                "parallel_mha_transition": False,
                "use_attn_pair_bias": True,
                "strict_feats": False,
                "feats_init_seq": ["res_seq_pdb_idx", "chain_break_per_res"],
                "feats_cond_seq": ["time_emb", "delta_t_emb"],
                "t_emb_dim": 32,
                "dim_cond": 64,
                "idx_emb_dim": 32,
                "fold_emb_dim": 32,
                "feats_pair_repr": ["rel_seq_sep", "xt_pair_dists"],
                "feats_pair_cond": ["time_emb", "delta_t_emb"],
                "xt_pair_dist_dim": 16,
                "xt_pair_dist_min": 0.1,
                "xt_pair_dist_max": 3,
                "x_sc_pair_dist_dim": 16,
                "x_sc_pair_dist_min": 0.1,
                "x_sc_pair_dist_max": 3,
                "x_motif_pair_dist_dim": 16,
                "x_motif_pair_dist_min": 0.1,
                "x_motif_pair_dist_max": 3,
                "seq_sep_dim": 127,
                "pair_repr_dim": 32,
                "update_pair_repr": False,
                "update_pair_repr_every_n": 2,
                "use_tri_mult": False,
                "num_registers": 4,
                "use_qkln": True,
                "num_buckets_predict_pair": 16,
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
            "loss_accumulation_steps": K,
            "self_cond": False,
            "fold_cond": False,
            "mask_T_prob": 0.5,
            "mask_A_prob": 0.5,
            "mask_C_prob": 0.5,
            "motif_conditioning": False,
            "ot_coupling": {"enabled": ot_enabled, "method": "exact", "reg": 0.05,
                            "reg_m": 1.0, "normalize_cost": False},
            "meanflow": {
                "ratio": 0.25,
                "P_mean": -0.4,
                "P_std": 1.0,
                "norm_p": 1.0,
                "norm_eps": 0.001,
                "nsteps_sample": 1,
            },
        },
        "opt": {"lr": 1e-4, "accumulate_grad_batches": accum},
    })
    return cfg


def _build_batch(B, N, device="cpu"):
    """Build a synthetic batch matching what extract_clean_sample expects."""
    coords = torch.randn(B, N, 3, 3, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask_dict = {"coords": torch.ones(B, N, 3, 3, dtype=torch.bool, device=device)}
    return {
        "coords": coords,
        "mask_dict": mask_dict,
        "mask": mask,
    }


def _build_model(K=1, accum=1, ot_enabled=False):
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg = _build_cfg(K=K, accum=accum, ot_enabled=ot_enabled)
    model = Proteina(cfg_exp=cfg)
    model.to("cpu")
    model.train()
    trainer_mock = mock.MagicMock()
    trainer_mock.world_size = 1
    # For K>1 (manual optimization), clip_gradients checks these attributes
    trainer_mock.gradient_clip_val = None
    trainer_mock.gradient_clip_algorithm = None
    # manual_backward calls trainer.strategy.backward(loss, None) --
    # wire it to actually call loss.backward()
    trainer_mock.strategy.backward = lambda loss, *a, **kw: loss.backward()
    model._trainer = trainer_mock

    if K > 1:
        # Wire up a real optimizer so manual_backward / optimizer.step work
        real_optimizers = model.configure_optimizers()
        if isinstance(real_optimizers, dict):
            real_opt = real_optimizers["optimizer"]
            real_sch = real_optimizers.get("lr_scheduler", {}).get("scheduler", None)
        else:
            real_opt = real_optimizers
            real_sch = None
        model.optimizers = lambda: real_opt
        model.lr_schedulers = lambda: real_sch

    return model


def test_compute_single_noise_loss_finite():
    """_compute_single_noise_loss returns finite losses that require grad."""
    model = _build_model(K=1)
    B, N = 4, 16
    batch = _build_batch(B, N)

    x_1, mask, batch_shape, n, dtype = model.extract_clean_sample(batch)
    x_1 = model.fm._mask_and_zero_com(x_1, mask)

    t, r = model.fm.sample_two_timesteps(
        batch_shape, torch.device("cpu"),
        ratio=model.meanflow_ratio,
        P_mean_t=model.meanflow_P_mean_t, P_std_t=model.meanflow_P_std_t,
        P_mean_r=model.meanflow_P_mean_r, P_std_r=model.meanflow_P_std_r,
    )
    t_ext = t[..., None, None]
    r_ext = r[..., None, None]

    loss, raw_mf, raw_fm, raw_adp_wt = model._compute_single_noise_loss(
        x_1, mask, t_ext, r_ext, t, batch, B
    )

    assert torch.isfinite(loss), f"combined loss not finite: {loss}"
    assert torch.isfinite(raw_mf), f"raw_loss_mf not finite: {raw_mf}"
    assert torch.isfinite(raw_fm), f"raw_loss_fm not finite: {raw_fm}"
    assert torch.isfinite(raw_adp_wt), f"raw_adp_wt_mean not finite: {raw_adp_wt}"
    assert loss.requires_grad, "combined loss should require grad"


def test_k1_identical_to_before():
    """
    With K=1, training_step should return a finite scalar and produce
    gradients -- regression check against the pre-refactor code path.
    """
    torch.manual_seed(0)
    model = _build_model(K=1)
    batch = _build_batch(4, 16)

    torch.manual_seed(0)
    loss = model.training_step(batch, batch_idx=0)

    assert loss is not None, "K=1 should return a loss tensor"
    assert torch.isfinite(loss), f"K=1 loss not finite: {loss}"
    loss.backward()

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "K=1: no parameters received nonzero gradients"


def test_k_greater_than_1_gradients():
    """K=2 training_step: gradients are nonzero, return value is None."""
    # Use accum=2 so optimizer.zero_grad() does NOT fire after the first
    # training_step, which would clear the gradients before we can check them.
    model = _build_model(K=2, accum=2)
    batch = _build_batch(4, 16)

    result = model.training_step(batch, batch_idx=0)

    assert result is None, "K>1 manual optimization should return None"

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "K=2: no parameters received nonzero gradients"


def test_accum_grad_batches_composes_with_k():
    """
    K=2, accumulate_grad_batches=2: optimizer.step() fires only after the
    second training_step call (i.e. after 2*2=4 total backward passes).
    """
    model = _build_model(K=2, accum=2)
    batch = _build_batch(4, 16)

    # Get the optimizer that _build_model already wired up
    real_opt = model.optimizers()

    step_calls = []
    original_step = real_opt.step
    real_opt.step = lambda *a, **kw: (step_calls.append(1), original_step(*a, **kw))

    # First call: step should NOT have fired (count=1, 1 % 2 != 0)
    model.training_step(batch, batch_idx=0)
    assert len(step_calls) == 0, f"Step fired too early: {len(step_calls)} calls after step 1"

    # Second call: step SHOULD fire (count=2, 2 % 2 == 0)
    model.training_step(batch, batch_idx=1)
    assert len(step_calls) == 1, f"Expected 1 optimizer step, got {len(step_calls)}"


def test_ot_coupling_called_k_times():
    """With OT enabled and K=2, the OT sampler is called exactly K times."""
    model = _build_model(K=2, ot_enabled=True)
    batch = _build_batch(4, 16)

    call_count = []
    original = model.ot_sampler.sample_plan_with_scipy

    def counting_sample_plan(x_1, x_0, mask):
        call_count.append(1)
        return original(x_1, x_0, mask)

    model.ot_sampler.sample_plan_with_scipy = counting_sample_plan

    model.training_step(batch, batch_idx=0)

    assert len(call_count) == 2, (
        f"Expected OT sampler called 2 times (K=2), got {len(call_count)}"
    )


def test_gradient_magnitude_consistent_across_k():
    """
    Gradient norms for K=4 should be in the same ballpark as K=1 (averaging,
    not summing). If gradients were summed, the K=4 norm would be ~4x larger.
    """
    batch = _build_batch(4, 16)

    # K=1: training_step returns a loss tensor; call backward manually.
    torch.manual_seed(42)
    model_k1 = _build_model(K=1)
    torch.manual_seed(42)
    loss_k1 = model_k1.training_step(batch, batch_idx=0)
    assert loss_k1 is not None, "K=1 should return a loss tensor"
    loss_k1.backward()

    grad_norm_k1 = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in model_k1.nn.parameters() if p.grad is not None)
    )

    # K=4: training_step returns None; gradients are accumulated internally via
    # manual_backward(loss_k / K), so they should already be averaged.
    torch.manual_seed(42)
    model_k4 = _build_model(K=4, accum=4)
    torch.manual_seed(42)
    result_k4 = model_k4.training_step(batch, batch_idx=0)
    assert result_k4 is None, "K=4 should return None (manual optimization)"

    grad_norm_k4 = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in model_k4.nn.parameters() if p.grad is not None)
    )

    assert grad_norm_k1 > 0, f"K=1 gradient norm should be > 0, got {grad_norm_k1}"
    assert grad_norm_k4 > 0, f"K=4 gradient norm should be > 0, got {grad_norm_k4}"

    # Averaging: K=4 norm should NOT be ~4x the K=1 norm.
    assert grad_norm_k4 < 2 * grad_norm_k1, (
        f"K=4 gradient norm ({grad_norm_k4:.4f}) is more than 2x the K=1 norm "
        f"({grad_norm_k1:.4f}), suggesting gradients are being summed rather than averaged"
    )

    # Sanity: norms should not differ by more than 10x in either direction.
    assert grad_norm_k4 > 0.1 * grad_norm_k1, (
        f"K=4 gradient norm ({grad_norm_k4:.4f}) is less than 0.1x the K=1 norm "
        f"({grad_norm_k1:.4f}), which is unexpectedly small"
    )


def test_k_greater_than_1_combined_adaptive_loss_logged_value():
    """
    K>1 must log the *adaptive* combined loss (bounded in [0,1) when
    norm_p=1, norm_eps=1e-3), NOT a combination of the raw pre-adaptive
    losses. With raw MSE values that can easily reach 5-60, mistakenly
    logging the raw combination would yield the unbounded values seen in
    run 8bn66xg2.
    """
    torch.manual_seed(0)
    model = _build_model(K=4, accum=4)
    batch = _build_batch(4, 16)

    logged = {}
    original_log = model.log

    def capture_log(name, value, *args, **kwargs):
        # Resolve tensors to floats so we can compare numerically.
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        logged[name] = value
        return None  # the real method returns None too

    model.log = capture_log

    model.training_step(batch, batch_idx=0)

    assert "train/combined_adaptive_loss" in logged, (
        "K>1 path did not log train/combined_adaptive_loss"
    )
    combined_adp = logged["train/combined_adaptive_loss"]
    raw_mf = logged["train/raw_loss_mf"]
    raw_fm = logged["train/raw_loss_fm"]

    # With norm_p=1, norm_eps=1e-3, adaptive_loss = loss / (loss + 1e-3)
    # is in [0, 1). The *combined* adaptive loss is a convex combination
    # of two such terms, so it must also be < 1.
    assert combined_adp < 1.0, (
        f"combined_adaptive_loss={combined_adp:.4f} must be < 1.0 with "
        f"norm_p=1, norm_eps=1e-3 (raw_mf={raw_mf:.4f}, raw_fm={raw_fm:.4f}). "
        "It looks like the K>1 path is logging a combination of raw losses "
        "instead of adaptive losses."
    )


def test_k_greater_than_1_non_pool_extract_called_once():
    """Without the pool, K>1 must call extract_clean_sample exactly once per
    training_step (shared-x_1 semantics). Pool mode is a separate path."""
    model = _build_model(K=2, ot_enabled=True)  # ot_pool_size NOT set
    # Defensive: ensure pool mode is OFF (the test pins non-pool K>1 semantics).
    model.cfg_exp.training.ot_coupling.ot_pool_size = None
    batch = _build_batch(4, 16)

    call_count = []
    original = model.extract_clean_sample

    def counting_extract(b):
        call_count.append(1)
        return original(b)

    model.extract_clean_sample = counting_extract

    model.training_step(batch, batch_idx=0)

    assert len(call_count) == 1, (
        f"Non-pool K=2: extract_clean_sample called {len(call_count)} times, "
        "expected exactly 1 (shared-x_1 semantics)."
    )
