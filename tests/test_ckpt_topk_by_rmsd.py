# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Tests for the val/rmsd_mf1-driven top-k checkpoint wiring.

Covers:
  1. ProteinValEvalCallback forwards val/rmsd_mf1 to pl_module.log so that
     Lightning's callback_metrics is populated for ModelCheckpoint(monitor=...).
  2. EmaModelCheckpoint accepts the expected (monitor, mode, save_top_k)
     arg shape for the top-k callback.
  3. EmaModelCheckpoint still accepts the existing (save_last, every_n_train_steps)
     arg shape for the "last" callback.
  4. The _build_ckpt_callbacks helper in train.py builds the right number
     of callbacks depending on top_k_by_val_rmsd_mf1.
"""

import types
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Test 1: ProteinValEvalCallback forwards val/rmsd_mf1 to pl_module.log
# ---------------------------------------------------------------------------


def _make_stub_graph(pid: str = "P1"):
    """Build a fake PyG-like graph with .id, .coords, .coord_mask."""
    import numpy as np
    import torch

    n_res = 8
    coords = torch.zeros(n_res, 37, 3)
    coords[:, 1, :] = torch.arange(n_res).float().unsqueeze(-1).repeat(1, 3)
    coord_mask = torch.zeros(n_res, 37, dtype=torch.bool)
    coord_mask[:, 1] = True

    graph = types.SimpleNamespace()
    graph.id = pid
    graph.coords = coords
    graph.coord_mask = coord_mask
    return graph


def test_val_eval_callback_logs_rmsd_mf1_to_callback_metrics(monkeypatch, tmp_path):
    """Callback must call pl_module.log("val/rmsd_mf1", ...) after aggregation."""
    from proteinfoundation.callbacks import protein_val_eval as pve_mod
    from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

    # Stub out sampling / IO utils so we don't run generation or touch disk beyond tmp.
    def fake_extract_gt_ca(graph):
        return (graph.coords[:, 1, :].cpu().numpy(), int(graph.coords.shape[0]))

    def fake_write_gt_pdb(gt_ca, pid, tmp_dir):
        return str(tmp_path / f"gt_{pid}.pdb")

    def fake_score_protein(pl_module, protein, nsamples, nsteps_base, tmp_dir, step):
        return {
            "mf1_pdb": str(tmp_path / "mf1.pdb"),
            "mf10x_pdb": str(tmp_path / "mf10x.pdb"),
            "rmsd_mf1": 1.23,
            "rmsd_mf10x": 2.34,
            "rmsd_refl_mf1": 1.10,
            "rmsd_refl_mf10x": 2.20,
            "chir_mf1": 0.5,
            "chir_mf10x": 0.6,
            "rmsd_mf1_best": 1.0,
            "rmsd_mf10x_best": 2.0,
        }

    def fake_aggregate_log_dict(prefix, proteins, results, step):
        return {
            f"{prefix}/rmsd_mf1": 1.23,
            f"{prefix}/rmsd_mf10x": 2.34,
            f"{prefix}/rmsd_reflected_mf1": 1.10,
            f"{prefix}/rmsd_reflected_mf10x": 2.20,
            f"{prefix}/chirality_mf1": 0.5,
            f"{prefix}/chirality_mf10x": 0.6,
            f"{prefix}/rmsd_mf1_best": 1.0,
            f"{prefix}/rmsd_mf10x_best": 2.0,
        }

    def fake_build_samples_table(proteins, results, step, current_lr):
        return MagicMock(name="samples_table")

    def fake_current_optimizer_lr(trainer):
        return 1e-4

    monkeypatch.setattr(pve_mod, "extract_gt_ca", fake_extract_gt_ca)
    monkeypatch.setattr(pve_mod, "write_gt_pdb", fake_write_gt_pdb)
    monkeypatch.setattr(pve_mod, "score_protein", fake_score_protein)
    monkeypatch.setattr(pve_mod, "aggregate_log_dict", fake_aggregate_log_dict)
    monkeypatch.setattr(pve_mod, "build_samples_table", fake_build_samples_table)
    monkeypatch.setattr(pve_mod, "current_optimizer_lr", fake_current_optimizer_lr)

    cb = ProteinValEvalCallback(run_name="unit-test", n_val_proteins=1, nsamples=1)

    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.global_rank = 0
    trainer.global_step = 42
    val_ds = [_make_stub_graph("P1")]
    trainer.datamodule.val_ds = val_ds

    pl_module = MagicMock()
    pl_module.training = False
    pl_module.meanflow_nsteps_sample = 1
    # .logger.experiment.log must be a no-op MagicMock.
    pl_module.logger = MagicMock()

    cb.on_validation_epoch_end(trainer, pl_module)

    # Assert pl_module.log was called with val/rmsd_mf1.
    log_calls = [c for c in pl_module.log.call_args_list]
    rmsd_calls = [c for c in log_calls if c.args and c.args[0] == "val/rmsd_mf1"]
    assert len(rmsd_calls) == 1, f"Expected exactly one pl_module.log call for val/rmsd_mf1, got calls: {log_calls}"

    call = rmsd_calls[0]
    # Second positional arg is the value; must be a float.
    assert isinstance(call.args[1], float)
    assert call.args[1] == pytest.approx(1.23)
    # Kwargs must be the exact shape expected by ModelCheckpoint monitor path.
    assert call.kwargs.get("on_step") is False
    assert call.kwargs.get("on_epoch") is True
    assert call.kwargs.get("logger") is False


# ---------------------------------------------------------------------------
# Test 2: EmaModelCheckpoint accepts top-k arg shape
# ---------------------------------------------------------------------------


def test_checkpoint_callback_wiring_top_k_3(tmp_path):
    from proteinfoundation.utils.ema_utils.ema_callback import EmaModelCheckpoint

    cb = EmaModelCheckpoint(
        monitor="val/rmsd_mf1",
        mode="min",
        save_top_k=3,
        dirpath=str(tmp_path),
        save_last=False,
        filename="chk_{epoch:08d}_{step:012d}_{val/rmsd_mf1:.4f}",
        save_weights_only=False,
    )
    assert cb.monitor == "val/rmsd_mf1"
    assert cb.mode == "min"
    assert cb.save_top_k == 3


# ---------------------------------------------------------------------------
# Test 3: EmaModelCheckpoint accepts "last" arg shape
# ---------------------------------------------------------------------------


def test_checkpoint_callback_wiring_last(tmp_path):
    from proteinfoundation.utils.ema_utils.ema_callback import EmaModelCheckpoint

    cb = EmaModelCheckpoint(
        save_last=True,
        every_n_train_steps=3000,
        save_weights_only=False,
        dirpath=str(tmp_path),
        filename="ignore",
    )
    assert cb.save_last is True
    assert cb._every_n_train_steps == 3000


# ---------------------------------------------------------------------------
# Test 4: _build_ckpt_callbacks helper honours top_k == 0 vs top_k > 0
# ---------------------------------------------------------------------------


def _make_log_cfg(top_k: int, *, ckpt_every: int = 10000, last_every: int = 3000):
    return OmegaConf.create(
        {
            "checkpoint_every_n_steps": ckpt_every,
            "last_ckpt_every_n_steps": last_every,
            "top_k_by_val_rmsd_mf1": top_k,
        }
    )


def test_top_k_zero_disables_topk_callback(tmp_path):
    """top_k=0 -> only last-ckpt callback. top_k=3 -> last + top-k callbacks."""
    from proteinfoundation.train import _build_ckpt_callbacks
    from proteinfoundation.utils.ema_utils.ema_callback import EmaModelCheckpoint

    # top_k == 0 -> exactly 1 callback, which must be the "save_last" one.
    cbs_zero = _build_ckpt_callbacks(str(tmp_path), _make_log_cfg(0))
    assert len(cbs_zero) == 1
    assert isinstance(cbs_zero[0], EmaModelCheckpoint)
    assert cbs_zero[0].save_last is True
    assert cbs_zero[0]._every_n_train_steps == 3000

    # top_k == 3 -> 2 callbacks: last + top-k on val/rmsd_mf1.
    cbs_three = _build_ckpt_callbacks(str(tmp_path), _make_log_cfg(3))
    assert len(cbs_three) == 2
    assert all(isinstance(c, EmaModelCheckpoint) for c in cbs_three)

    monitors = [c.monitor for c in cbs_three]
    save_tops = [c.save_top_k for c in cbs_three]
    assert "val/rmsd_mf1" in monitors
    assert 3 in save_tops
    # Exactly one of the two is the top-k ckpt; the other is the last-ckpt.
    topk_cb = next(c for c in cbs_three if c.monitor == "val/rmsd_mf1")
    last_cb = next(c for c in cbs_three if c.monitor != "val/rmsd_mf1")
    assert topk_cb.mode == "min"
    assert topk_cb.save_top_k == 3
    assert last_cb.save_last is True
    assert last_cb._every_n_train_steps == 3000
