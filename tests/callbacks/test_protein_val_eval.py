# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Unit tests for ProteinValEvalCallback."""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
import torch_geometric.data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_val_ds(n_proteins=3, n_res_list=None):
    """Return a list of fake PyG Data objects simulating val_ds entries."""
    if n_res_list is None:
        n_res_list = [10, 15, 20]
    assert len(n_res_list) == n_proteins
    graphs = []
    for i, n_res in enumerate(n_res_list):
        graph = torch_geometric.data.Data(
            coords=torch.randn(n_res, 37, 3),
            coord_mask=torch.ones(n_res, 37, dtype=torch.bool),
            id=f"prot{i}",
        )
        graphs.append(graph)
    return graphs


def _make_trainer(val_ds, global_rank=0, global_step=100, sanity_checking=False):
    trainer = mock.MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.global_rank = global_rank
    trainer.global_step = global_step
    trainer.optimizers = []  # no EMAOptimizer -> nullcontext fallback
    trainer.datamodule.val_ds = val_ds
    return trainer


def _make_pl_module(n_res):
    """Return a MagicMock pl_module whose generate/samples_to_atom37 return zeros."""
    pl_module = mock.MagicMock()
    pl_module.training = False

    def fake_generate(nsamples, n, nsteps, mask=None):
        return torch.zeros(nsamples, n, 3)

    def fake_samples_to_atom37(samples):
        # samples: [nsamples, n, 3] — return [nsamples, n, 37, 3]
        nsamples, n, _ = samples.shape
        return torch.zeros(nsamples, n, 37, 3)

    pl_module.generate = fake_generate
    pl_module.samples_to_atom37 = fake_samples_to_atom37
    pl_module.meanflow_nsteps_sample = 1
    return pl_module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProteinValEvalCallback:

    def test_lazy_init_caches_val_proteins(self, tmp_path):
        """Lazy init caches n_val_proteins entries and writes GT PDB files."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        val_ds = _make_val_ds(n_proteins=3, n_res_list=[10, 15, 20])
        trainer = _make_trainer(val_ds, global_step=100)
        pl_module = _make_pl_module(10)

        cb = ProteinValEvalCallback(run_name="test_run_lazy", n_val_proteins=2)
        # Redirect tmp dir to tmp_path to avoid polluting the repo.
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch("proteinfoundation.callbacks.protein_val_eval.wandb.log"),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # Cache should have exactly 2 proteins (n_val_proteins=2, ds has 3).
        assert cb._val_proteins is not None
        assert len(cb._val_proteins) == 2

        # GT PDB files must exist on disk.
        for p in cb._val_proteins:
            assert os.path.exists(p["gt_pdb_path"]), (
                f"GT PDB missing: {p['gt_pdb_path']}"
            )

        # GT should have been logged once.
        assert cb._gt_logged is True

    def test_sanity_check_skipped(self, tmp_path):
        """When sanity_checking=True the callback must not touch anything."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        val_ds = _make_val_ds()
        trainer = _make_trainer(val_ds, sanity_checking=True)
        pl_module = _make_pl_module(10)

        cb = ProteinValEvalCallback(run_name="test_sanity", n_val_proteins=2)
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch("proteinfoundation.callbacks.protein_val_eval.wandb.log") as mock_log,
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # Nothing should have been cached or written.
        assert cb._val_proteins is None
        assert cb._gt_logged is False
        assert list(tmp_path.iterdir()) == []
        mock_log.assert_not_called()

    def test_rank_nonzero_skipped(self, tmp_path):
        """When global_rank != 0 the callback returns without doing anything."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        val_ds = _make_val_ds()
        trainer = _make_trainer(val_ds, global_rank=1)
        pl_module = _make_pl_module(10)

        cb = ProteinValEvalCallback(run_name="test_rank1", n_val_proteins=2)
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch("proteinfoundation.callbacks.protein_val_eval.wandb.log") as mock_log,
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._val_proteins is None
        assert cb._gt_logged is False
        mock_log.assert_not_called()

    def test_second_call_skips_reinit(self, tmp_path):
        """After the first call, _val_proteins must not be overwritten."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        val_ds = _make_val_ds(n_proteins=3, n_res_list=[10, 15, 20])
        trainer = _make_trainer(val_ds, global_step=200)
        pl_module = _make_pl_module(10)

        cb = ProteinValEvalCallback(run_name="test_second_call", n_val_proteins=2)
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch("proteinfoundation.callbacks.protein_val_eval.wandb.log"),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)
            first_cache = cb._val_proteins

            trainer.global_step = 300
            cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._val_proteins is first_cache, (
            "_val_proteins list object should not be replaced on second call"
        )

    def test_wandb_log_called_with_aggregate_keys(self, tmp_path):
        """The per-round wandb.log call must contain the aggregate scalar keys."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        n_res_list = [8, 12]
        val_ds = _make_val_ds(n_proteins=2, n_res_list=n_res_list)
        trainer = _make_trainer(val_ds, global_step=50)
        pl_module = _make_pl_module(8)

        cb = ProteinValEvalCallback(run_name="test_keys", n_val_proteins=2)
        cb._tmp_dir = str(tmp_path)

        logged_dicts = []

        def capture_log(d, step=None):
            logged_dicts.append(dict(d))

        with (
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.log",
                side_effect=capture_log,
            ),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # First log call is GT; second is per-round metrics.
        assert len(logged_dicts) == 2
        round_dict = logged_dicts[1]
        for key in [
            "val/rmsd_mf1",
            "val/rmsd_mf10x",
            "val/rmsd_reflected_mf1",
            "val/rmsd_reflected_mf10x",
            "val/chirality_mf1",
            "val/chirality_mf10x",
            "trainer/global_step",
        ]:
            assert key in round_dict, f"Missing key in wandb log: {key}"
