"""
Verify that validation training_step logs under the "val/" prefix (not
"validation_loss/").

This covers Change 1 of the WandB logging patch: all validation metrics
emitted from `ModelTrainerBase.training_step(..., val_step=True)` must live
under the "val/" WandB section so they stack alongside other validation
panels in the UI.

Uses the same hermetic fixtures as `tests/test_training_step.py` (CPU-only,
torch_scatter/pandas/biotite stubbed).
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
if "torch_scatter" not in sys.modules or getattr(sys.modules["torch_scatter"], "__spec__", None) is None:
    _ts_stub = _stub_module("torch_scatter")
    _ts_stub.scatter_mean = None

# Stub pandas and biotite (motif_factory imports them; pandas has numpy ABI
# issues in this env, and biotite is unused by our test).
if "pandas" not in sys.modules or getattr(sys.modules["pandas"], "__spec__", None) is None:
    _stub_module("pandas")

for _bio_name in ["biotite", "biotite.structure", "biotite.structure.io"]:
    if _bio_name not in sys.modules:
        _stub_module(_bio_name)
sys.modules["biotite"].structure = sys.modules["biotite.structure"]
sys.modules["biotite.structure"].io = sys.modules["biotite.structure.io"]

import torch
from omegaconf import OmegaConf


def _build_cfg(ratio: float = 0.5):
    """Minimal OmegaConf config matching Proteina.__init__ expectations."""
    return OmegaConf.create({
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


def _build_batch(B, N, device="cpu"):
    coords = torch.randn(B, N, 3, 3, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask_dict = {"coords": torch.ones(B, N, 3, 3, dtype=torch.bool, device=device)}
    return {"coords": coords, "mask_dict": mask_dict, "mask": mask}


def _build_model(ratio: float = 0.5):
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg = _build_cfg(ratio)
    model = Proteina(cfg_exp=cfg)
    model.to("cpu")
    model.train()
    trainer_mock = mock.MagicMock()
    trainer_mock.world_size = 1
    model._trainer = trainer_mock
    return model


def test_val_step_logs_under_val_prefix():
    """training_step(..., val_step=True) must log under "val/", never "validation_loss/"."""
    torch.manual_seed(0)
    model = _build_model(ratio=0.5)
    batch = _build_batch(B=2, N=8)

    # Capture every self.log call. Replace bound method with a plain MagicMock
    # so keyword arguments like sync_dist=True don't collide with LightningModule's
    # own parameter validation.
    logged_keys = []

    def fake_log(key, value, *args, **kwargs):
        logged_keys.append(key)

    with mock.patch.object(model, "log", side_effect=fake_log) as log_mock:
        out = model.training_step(batch, batch_idx=0, val_step=True)

    # Sanity: a scalar loss came back from K=1 path.
    assert out is not None
    assert torch.isfinite(out), f"Val loss not finite: {out}"

    # At least one key must live under "val/"
    val_keys = [k for k in logged_keys if k.startswith("val/")]
    assert val_keys, (
        f"Expected at least one key under 'val/' prefix, got: {logged_keys}"
    )
    assert "val/combined_adaptive_loss" in logged_keys, (
        f"Expected 'val/combined_adaptive_loss' in logged keys, got: {logged_keys}"
    )
    # Raw losses must also appear on val so train/val raw-loss curves can be
    # compared directly. Previously gated behind `if not val_step:`.
    for required in ("val/raw_loss_mf", "val/raw_loss_fm", "val/raw_loss_chirality", "val/raw_adp_wt_mean"):
        assert required in logged_keys, (
            f"Expected {required!r} in logged keys, got: {logged_keys}"
        )

    # No key should use the old "validation_loss/" prefix.
    stale_keys = [k for k in logged_keys if k.startswith("validation_loss/")]
    assert not stale_keys, (
        f"Found stale 'validation_loss/' keys: {stale_keys}. "
        "Validation metrics should live under 'val/'."
    )

    # log was actually called at least once with val/ prefix
    assert log_mock.call_count >= 1


def test_training_step_does_not_use_val_prefix():
    """Guardrail: non-val training_step must NOT leak into 'val/' section."""
    torch.manual_seed(0)
    model = _build_model(ratio=0.5)
    batch = _build_batch(B=2, N=8)

    logged_keys = []

    def fake_log(key, value, *args, **kwargs):
        logged_keys.append(key)

    with mock.patch.object(model, "log", side_effect=fake_log):
        model.training_step(batch, batch_idx=0, val_step=False)

    val_keys = [k for k in logged_keys if k.startswith("val/")]
    assert not val_keys, (
        f"Training step should not log under 'val/'; found: {val_keys}"
    )
    stale_keys = [k for k in logged_keys if k.startswith("validation_loss/")]
    assert not stale_keys, (
        f"Training step should not log under 'validation_loss/'; found: {stale_keys}"
    )
    # Sanity: train keys were emitted.
    train_keys = [k for k in logged_keys if k.startswith("train/")]
    assert train_keys, f"Expected 'train/' keys, got: {logged_keys}"
