# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""End-to-end smoke test for the periodic checkpoint callback.

Runs a real Lightning Trainer on a 1-parameter dummy module on CPU and asserts:
  * exactly `keep_last_periodic` periodic_*.ckpt files remain on disk,
  * those files correspond to the most recent firings (highest step values).

This is the load-bearing claim of the design — that monitor='step' + mode='max'
+ save_top_k=M with every_n_train_steps=N retains the M most recent firings.
The unit tests in test_ckpt_topk_by_val_loss.py only check callback config; this
test exercises the actual rotation behaviour through Lightning.
"""

import re
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TinyModule(nn.Module):
    """Plain nn.Module used only to satisfy LightningModule's needs."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)


def _make_lightning_module():
    """Return a minimal LightningModule that does one optimizer step per batch."""
    import lightning.pytorch as pl

    class _LM(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = _TinyModule()

        def training_step(self, batch, batch_idx):
            x, y = batch
            return ((self.net.lin(x) - y) ** 2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    return _LM()


def _make_loader():
    x = torch.randn(64, 2)
    y = torch.randn(64, 1)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def _periodic_ckpt_files(dirpath: Path):
    return sorted(dirpath.glob("periodic_*.ckpt"))


_STEP_RE = re.compile(r"periodic_epoch=\d+_step=(\d+)\.ckpt")


def _step_in_filename(p: Path) -> int:
    m = _STEP_RE.match(p.name)
    assert m, f"unexpected periodic filename {p.name!r}"
    return int(m.group(1))


def test_periodic_ckpt_rotation_keeps_most_recent_M_on_disk(tmp_path):
    """Run a real Trainer. Periodic callback should retain only the M most-recent ckpts."""
    import lightning.pytorch as pl

    from proteinfoundation.train import _build_ckpt_callbacks

    every_n = 5
    keep_last = 3
    max_steps = every_n * (keep_last + 2)  # 5 firings, keep last 3

    cfg_log = OmegaConf.create(
        {
            "checkpoint_every_n_steps": every_n,
            "last_ckpt_every_n_steps": 9999,  # too high to fire in this run
            "top_k_by_val_loss": 0,           # disable top-k for this smoke test
            "keep_last_periodic": keep_last,
        }
    )

    callbacks = _build_ckpt_callbacks(str(tmp_path), cfg_log)
    # 1 (last) + 1 (periodic) = 2; top_k=0 disables that callback.
    assert len(callbacks) == 2

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(_make_lightning_module(), train_dataloaders=_make_loader())

    files = _periodic_ckpt_files(tmp_path)
    assert len(files) == keep_last, (
        f"expected exactly {keep_last} periodic ckpts on disk; "
        f"got {[p.name for p in files]}"
    )

    # The retained files should be the most recent firings: steps {3*every_n,
    # 4*every_n, 5*every_n} = {15, 20, 25}.
    steps_on_disk = sorted(_step_in_filename(p) for p in files)
    expected_steps = [every_n * (i + 1) for i in range(keep_last + 2 - keep_last, keep_last + 2)]
    # i.e. for keep_last=3 and 5 firings: indices 2,3,4 -> steps 15,20,25.
    assert steps_on_disk == expected_steps, (
        f"expected most-recent step values {expected_steps}; got {steps_on_disk}"
    )


def test_periodic_ckpt_disabled_writes_no_periodic_files(tmp_path):
    """checkpoint_every_n_steps=0 must not write any periodic_*.ckpt files."""
    import lightning.pytorch as pl

    from proteinfoundation.train import _build_ckpt_callbacks

    cfg_log = OmegaConf.create(
        {
            "checkpoint_every_n_steps": 0,
            "last_ckpt_every_n_steps": 9999,
            "top_k_by_val_loss": 0,
            "keep_last_periodic": 5,
        }
    )

    callbacks = _build_ckpt_callbacks(str(tmp_path), cfg_log)
    assert len(callbacks) == 1  # just the "last" callback

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=10,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(_make_lightning_module(), train_dataloaders=_make_loader())

    files = _periodic_ckpt_files(tmp_path)
    assert files == [], f"expected no periodic_*.ckpt files; got {[p.name for p in files]}"
