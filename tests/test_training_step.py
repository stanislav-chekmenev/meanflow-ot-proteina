"""
Integration tests for MeanFlow training_step.

Verifies that training_step produces finite loss and gradient flow
for the three batch-splitting code paths:
  1. All FM (ratio=0.0 => n_mf==0)
  2. All MeanFlow (ratio=1.0 => n_mf==B)
  3. Mixed (ratio=0.5 => else branch)

Works on CPU (stubs out CUDA-only deps like torch_scatter).
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


def _build_cfg(ratio: float):
    """Build a minimal OmegaConf config matching what Proteina.__init__ expects."""
    cfg = OmegaConf.create({
        "model": {
            "target_pred": "v",
            "augmentation": {
                "global_rotation": False,
                "naug_rot": 1,
            },
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
            "self_cond": False,
            "fold_cond": False,
            "mask_T_prob": 0.5,
            "mask_A_prob": 0.5,
            "mask_C_prob": 0.5,
            "motif_conditioning": False,
            "ot_coupling": {"enabled": False},
            "meanflow": {
                "ratio": ratio,
                "P_mean": -0.4,
                "P_std": 1.0,
                "norm_p": 1.0,
                "norm_eps": 0.001,
                "nsteps_sample": 1,
            },
        },
        "opt": {"lr": 1e-4},
    })
    return cfg


def _build_batch(B, N, device="cpu"):
    """Build a synthetic batch matching what extract_clean_sample expects."""
    coords = torch.randn(B, N, 3, 3, device=device)  # [b, n, 3_atoms, 3_xyz]
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask_dict = {"coords": torch.ones(B, N, 3, 3, dtype=torch.bool, device=device)}
    return {
        "coords": coords,
        "mask_dict": mask_dict,
        "mask": mask,
    }


def _build_model(ratio: float):
    """Build and configure a Proteina model for testing."""
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg = _build_cfg(ratio)
    model = Proteina(cfg_exp=cfg)
    model.to("cpu")
    model.train()
    # Mock the trainer attribute that LightningModule expects.
    # Use MagicMock so any attribute (barebones, world_size, ...) auto-resolves.
    trainer_mock = mock.MagicMock()
    trainer_mock.world_size = 1
    model._trainer = trainer_mock
    return model


def _run_training_step(model, B, N):
    """Run a training_step and return (loss, model) for verification."""
    batch = _build_batch(B, N)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    return loss


def test_training_step_all_fm():
    """ratio=0.0 => all samples are standard FM (n_mf==0 path)."""
    model = _build_model(ratio=0.0)
    loss = _run_training_step(model, B=4, N=16)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "No parameters received gradients in all-FM path"


def test_training_step_all_mf():
    """ratio=1.0 => all samples are MeanFlow (n_mf==B path)."""
    model = _build_model(ratio=1.0)
    loss = _run_training_step(model, B=4, N=16)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "No parameters received gradients in all-MF path"


def test_training_step_mixed():
    """ratio=0.5 with seeded RNG to deterministically hit the mixed split path."""
    torch.manual_seed(42)
    model = _build_model(ratio=0.5)
    B, N = 16, 16
    batch = _build_batch(B, N)

    # Verify the seed produces a mixed batch (some FM, some MF)
    with torch.no_grad():
        t, r = model.fm.sample_two_timesteps(
            (B,), torch.device("cpu"), ratio=0.5,
            P_mean_t=-0.4, P_std_t=1.0, P_mean_r=-0.4, P_std_r=1.0,
        )
        n_mf = ((t - r).abs() > 1e-7).sum().item()
        assert 0 < n_mf < B, (
            f"Seed 42 did not produce a mixed batch (n_mf={n_mf}, B={B}). "
            "Update the seed to one that gives a mixed split."
        )

    # Re-seed so training_step gets the same timesteps
    torch.manual_seed(42)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    n_with_grad = sum(
        1 for p in model.nn.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "No parameters received gradients in mixed path"
