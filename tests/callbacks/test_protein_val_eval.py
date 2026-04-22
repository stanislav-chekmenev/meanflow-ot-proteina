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


def _make_pl_module(n_res, experiment_log=None):
    """Return a MagicMock pl_module whose generate/samples_to_atom37 return zeros.

    If ``experiment_log`` is provided, it is installed as the
    ``pl_module.logger.experiment.log`` callable so tests can assert on the
    arguments passed to the WandB logger.
    """
    pl_module = mock.MagicMock()
    pl_module.training = False
    if experiment_log is not None:
        pl_module.logger.experiment.log = experiment_log

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
        """Lazy init caches n_val_proteins entries and writes GT PDB files.

        GT molecules are no longer logged during lazy init — they now appear
        as rows of the per-round `samples/protein_table` wandb.Table.
        """
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
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                return_value=mock.MagicMock(),
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
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                return_value=mock.MagicMock(),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # Nothing should have been cached or written.
        assert cb._val_proteins is None
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
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                return_value=mock.MagicMock(),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._val_proteins is None
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
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                return_value=mock.MagicMock(),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)
            first_cache = cb._val_proteins

            trainer.global_step = 300
            cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._val_proteins is first_cache, (
            "_val_proteins list object should not be replaced on second call"
        )

    def test_zero_mask_protein_skipped(self, tmp_path):
        """Middle protein with all-False CA mask is skipped; no crash; warning logged."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        # Build 3 proteins; zero out CA mask for the middle one.
        graphs = _make_val_ds(n_proteins=3, n_res_list=[10, 8, 12])
        graphs[1].coord_mask[:, 1] = torch.zeros(8, dtype=torch.bool)

        trainer = _make_trainer(graphs, global_step=42)
        pl_module = _make_pl_module(10)

        cb = ProteinValEvalCallback(run_name="test_zero_mask", n_val_proteins=3)
        cb._tmp_dir = str(tmp_path)

        warning_calls = []

        with (
            mock.patch("proteinfoundation.callbacks.protein_val_eval.wandb.log"),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.logger.warning",
                side_effect=lambda msg, *a, **kw: warning_calls.append(msg),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # Middle protein must have been skipped.
        assert cb._val_proteins is not None
        assert len(cb._val_proteins) == 2

        # A warning must have been logged for the skipped protein.
        all_warnings = " ".join(warning_calls).lower()
        assert "skipping val protein" in all_warnings

    def test_wandb_log_called_with_aggregate_keys(self, tmp_path):
        """The per-round log call must contain the aggregate scalar keys plus
        a `samples/protein_table` wandb.Table, must NOT include any per-protein
        molecule keys, and must NOT pass an explicit ``step=`` (that would
        break the ``trainer/global_step`` step_metric alignment set in
        train.py)."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        n_res_list = [8, 12]
        val_ds = _make_val_ds(n_proteins=2, n_res_list=n_res_list)
        trainer = _make_trainer(val_ds, global_step=50)

        logged_calls = []

        def capture_log(d, *args, **kwargs):
            logged_calls.append((dict(d), kwargs))

        pl_module = _make_pl_module(8, experiment_log=capture_log)
        # GT is no longer logged via trainer.logger during lazy init, but keep
        # a safety net: if _init_val_proteins ever calls logger.experiment.log
        # again it will get recorded here and blow the count assertion.
        trainer.logger.experiment.log = capture_log

        cb = ProteinValEvalCallback(run_name="test_keys", n_val_proteins=2)
        cb._tmp_dir = str(tmp_path)

        # Mock wandb.Table so we can verify the columns and the number of rows
        # added, independent of the real wandb runtime.
        class _FakeTable:
            def __init__(self, columns=None):
                self.columns = columns
                self.rows = []

            def add_data(self, *args):
                self.rows.append(args)

        fake_table_instances = []

        def _table_factory(*args, **kwargs):
            t = _FakeTable(columns=kwargs.get("columns"))
            fake_table_instances.append(t)
            return t

        with (
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                side_effect=_table_factory,
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # Only the per-round metrics log call should happen — GT is no longer
        # logged separately.
        assert len(logged_calls) == 1, (
            f"Expected exactly 1 log call (per-round); got {len(logged_calls)}"
        )
        round_dict, kwargs = logged_calls[0]
        assert "step" not in kwargs, (
            "log must not pass explicit step= (would desync with "
            "trainer/global_step step_metric)"
        )

        # All existing aggregate scalar keys must still be present.
        for key in [
            "val/rmsd_mf1",
            "val/rmsd_mf10x",
            "val/rmsd_reflected_mf1",
            "val/rmsd_reflected_mf10x",
            "val/chirality_mf1",
            "val/chirality_mf10x",
            "trainer/global_step",
        ]:
            assert key in round_dict, f"Missing key in log: {key}"

        # New: samples/protein_table must be present and must be a wandb.Table
        # (our _FakeTable stands in for the mocked wandb.Table return value).
        assert "samples/protein_table" in round_dict, (
            "Missing key in log: samples/protein_table"
        )
        table = round_dict["samples/protein_table"]
        assert isinstance(table, _FakeTable), (
            f"samples/protein_table must be a wandb.Table; got {type(table)!r}"
        )
        assert table.columns == [
            "global_step",
            "lr",
            "protein_id",
            "rmsd_mf1",
            "rmsd_mf10x",
            "rmsd_reflected_mf1",
            "rmsd_reflected_mf10x",
            "chirality_mf1",
            "chirality_mf10x",
            "rmsd_mf1_best",
            "rmsd_mf10x_best",
            "ground_truth",
            "mf1",
            "mf10x",
        ], f"Unexpected column order: {table.columns!r}"
        assert len(table.rows) == 2, (
            f"Expected 2 rows (one per val protein); got {len(table.rows)}"
        )
        # First column is the annotating global_step; every row must carry it.
        for row in table.rows:
            assert row[0] == 50, (
                f"Expected global_step=50 in every row; got {row[0]}"
            )

        # No per-protein val/<pid>/* molecule keys must appear in the log.
        per_protein_keys = [
            k
            for k in round_dict
            if k.startswith("val/")
            and k.endswith(("/gt", "/mf1", "/mf10x"))
        ]
        assert per_protein_keys == [], (
            f"Per-protein molecule keys should be gone, but found: "
            f"{per_protein_keys}"
        )

    def test_nsamples_triggers_multiple_generate_calls(self, tmp_path):
        """With nsamples=3 the callback must call generate() 6 times per protein
        (3 for mf1 + 3 for mf10x) so per-draw metrics can be averaged."""
        from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback

        val_ds = _make_val_ds(n_proteins=1, n_res_list=[10])
        trainer = _make_trainer(val_ds, global_step=77)
        pl_module = _make_pl_module(10)

        # Wrap fake_generate in a MagicMock so we can count calls.
        generate_calls = []
        original_generate = pl_module.generate

        def counting_generate(nsamples, n, nsteps, mask=None):
            generate_calls.append(nsteps)
            return original_generate(nsamples=nsamples, n=n, nsteps=nsteps, mask=mask)

        pl_module.generate = counting_generate

        cb = ProteinValEvalCallback(
            run_name="test_nsamples", n_val_proteins=1, nsamples=3
        )
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
                return_value="fake-mol",
            ),
            mock.patch(
                "proteinfoundation.callbacks.protein_val_eval.wandb.Table",
                return_value=mock.MagicMock(add_data=lambda *a: None),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # 3 mf1 calls (nsteps=1) + 3 mf10x calls (nsteps=10) = 6 total.
        assert len(generate_calls) == 6, (
            f"Expected 6 generate calls (3 per mode); got {len(generate_calls)}: {generate_calls}"
        )
        assert generate_calls.count(1) == 3
        assert generate_calls.count(10) == 3
