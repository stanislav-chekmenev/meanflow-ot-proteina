# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Tests for the val/raw_loss_mf_epoch-driven top-k checkpoint wiring.

Covers:
  1. EmaModelCheckpoint accepts the expected (monitor, mode, save_top_k)
     arg shape for the top-k callback — targeting val/raw_loss_mf_epoch.
  2. EmaModelCheckpoint still accepts the existing (save_last, every_n_train_steps)
     arg shape for the "last" callback.
  3. The _build_ckpt_callbacks helper in train.py builds the right number
     of callbacks depending on top_k_by_val_loss.
  4. _build_ckpt_callbacks reads top_k_by_val_loss (not top_k_by_val_rmsd_mf1).

Note: val/raw_loss_mf_epoch is emitted automatically by Lightning via
on_epoch=True + sync_dist=True in model_trainer_base.py — no custom callback
broadcast is needed, so the DDP-rank tests from the old test file are removed.
"""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_log_cfg(
    top_k: int,
    *,
    ckpt_every: int = 0,
    last_every: int = 3000,
    keep_last_periodic: int = 5,
):
    return OmegaConf.create(
        {
            "checkpoint_every_n_steps": ckpt_every,
            "last_ckpt_every_n_steps": last_every,
            "top_k_by_val_loss": top_k,
            "keep_last_periodic": keep_last_periodic,
        }
    )


# ---------------------------------------------------------------------------
# Test 1: EmaModelCheckpoint accepts top-k arg shape with new monitor
# ---------------------------------------------------------------------------


def test_checkpoint_callback_wiring_top_k_3_val_loss(tmp_path):
    """EmaModelCheckpoint must accept monitor=val/raw_loss_mf_epoch with mode=min."""
    from proteinfoundation.utils.ema_utils.ema_callback import EmaModelCheckpoint

    cb = EmaModelCheckpoint(
        monitor="val/raw_loss_mf_epoch",
        mode="min",
        save_top_k=3,
        dirpath=str(tmp_path),
        save_last=False,
        filename="chk_{epoch:08d}_{step:012d}_{val/raw_loss_mf_epoch:.6f}",
        save_weights_only=False,
    )
    assert cb.monitor == "val/raw_loss_mf_epoch"
    assert cb.mode == "min"
    assert cb.save_top_k == 3
    assert "val/raw_loss_mf_epoch" in cb.filename


# ---------------------------------------------------------------------------
# Test 2: EmaModelCheckpoint accepts "last" arg shape (unchanged)
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
# Test 3: top_k=0 disables the top-k callback; top_k=3 adds it
# ---------------------------------------------------------------------------


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

    # top_k == 3 -> 2 callbacks: last + top-k on val/raw_loss_mf_epoch.
    cbs_three = _build_ckpt_callbacks(str(tmp_path), _make_log_cfg(3))
    assert len(cbs_three) == 2
    assert all(isinstance(c, EmaModelCheckpoint) for c in cbs_three)

    monitors = [c.monitor for c in cbs_three]
    save_tops = [c.save_top_k for c in cbs_three]
    assert "val/raw_loss_mf_epoch" in monitors
    assert 3 in save_tops
    # Exactly one of the two is the top-k ckpt; the other is the last-ckpt.
    topk_cb = next(c for c in cbs_three if c.monitor == "val/raw_loss_mf_epoch")
    last_cb = next(c for c in cbs_three if c.monitor != "val/raw_loss_mf_epoch")
    assert topk_cb.mode == "min"
    assert topk_cb.save_top_k == 3
    assert last_cb.save_last is True
    assert last_cb._every_n_train_steps == 3000


# ---------------------------------------------------------------------------
# Test 4: _build_ckpt_callbacks reads top_k_by_val_loss (not top_k_by_val_rmsd_mf1)
# ---------------------------------------------------------------------------


def test_build_ckpt_callbacks_reads_top_k_by_val_loss(tmp_path):
    """Passing top_k_by_val_loss=5 must produce a top-k callback with save_top_k==5."""
    from proteinfoundation.train import _build_ckpt_callbacks

    cfg = _make_log_cfg(5)
    cbs = _build_ckpt_callbacks(str(tmp_path), cfg)
    assert len(cbs) == 2
    monitor_cb = next(c for c in cbs if getattr(c, "monitor", None) == "val/raw_loss_mf_epoch")
    assert monitor_cb.save_top_k == 5


# ---------------------------------------------------------------------------
# Test 5: filename template of the built monitor callback embeds the monitor key
# ---------------------------------------------------------------------------


def test_built_monitor_callback_filename_template_embeds_monitor_key(tmp_path):
    """The top-k callback built by _build_ckpt_callbacks must embed val/raw_loss_mf_epoch in its filename.

    Without this, Lightning cannot format the monitored value into the saved filename,
    making inspection of best-vs-last checkpoints ambiguous.
    """
    from proteinfoundation.train import _build_ckpt_callbacks

    cbs = _build_ckpt_callbacks(str(tmp_path), _make_log_cfg(3))
    monitor_cb = next(c for c in cbs if getattr(c, "monitor", None) == "val/raw_loss_mf_epoch")
    assert "val/raw_loss_mf_epoch" in monitor_cb.filename, (
        f"Expected 'val/raw_loss_mf_epoch' in filename template; got {monitor_cb.filename!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: mode='min' on the built monitor callback (not just the manually-made one)
# ---------------------------------------------------------------------------


def test_built_monitor_callback_mode_is_min(tmp_path):
    """_build_ckpt_callbacks must set mode='min' on the monitor callback — lower val loss is better."""
    from proteinfoundation.train import _build_ckpt_callbacks

    cbs = _build_ckpt_callbacks(str(tmp_path), _make_log_cfg(3))
    monitor_cb = next(c for c in cbs if getattr(c, "monitor", None) == "val/raw_loss_mf_epoch")
    assert monitor_cb.mode == "min", (
        f"monitor callback must use mode='min'; got {monitor_cb.mode!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: built "last" callback has no monitor (so it never trips the top-k logic)
# ---------------------------------------------------------------------------


def test_built_last_callback_has_no_monitor(tmp_path):
    """The save_last callback in _build_ckpt_callbacks must not have a monitor set.

    It fires on a fixed step cadence, not on val-loss improvement.
    """
    from proteinfoundation.train import _build_ckpt_callbacks

    cbs = _build_ckpt_callbacks(str(tmp_path), _make_log_cfg(3))
    last_cb = next(c for c in cbs if getattr(c, "monitor", None) != "val/raw_loss_mf_epoch")
    assert last_cb.monitor is None, (
        f"'last' callback must have monitor=None; got {last_cb.monitor!r}"
    )
    assert last_cb.save_last is True
