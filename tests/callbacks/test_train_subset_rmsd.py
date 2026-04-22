# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Unit tests for TrainSubsetRmsdCallback."""

import os
from unittest import mock

import numpy as np
import torch
import torch_geometric.data


def _make_train_ds(n_proteins=20, seed=0):
    """Return a list of fake PyG Data objects simulating a training dataset."""
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n_proteins):
        n_res = int(rng.integers(8, 24))
        graphs.append(
            torch_geometric.data.Data(
                coords=torch.randn(n_res, 37, 3),
                coord_mask=torch.ones(n_res, 37, dtype=torch.bool),
                id=f"train_prot{i}",
            )
        )
    return graphs


def _make_trainer(train_ds, val_ds=None, global_rank=0, global_step=100, sanity_checking=False):
    trainer = mock.MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.global_rank = global_rank
    trainer.global_step = global_step
    trainer.optimizers = []  # nullcontext fallback
    trainer.datamodule.train_ds = train_ds
    trainer.datamodule.val_ds = val_ds
    return trainer


def _make_pl_module(experiment_log=None):
    pl_module = mock.MagicMock()
    pl_module.training = False
    if experiment_log is not None:
        pl_module.logger.experiment.log = experiment_log

    def fake_generate(nsamples, n, nsteps, mask=None):
        return torch.zeros(nsamples, n, 3)

    def fake_samples_to_atom37(samples):
        nsamples, n, _ = samples.shape
        return torch.zeros(nsamples, n, 37, 3)

    pl_module.generate = fake_generate
    pl_module.samples_to_atom37 = fake_samples_to_atom37
    pl_module.meanflow_nsteps_sample = 1
    return pl_module


class TestTrainSubsetRmsdCallback:

    def test_subset_is_deterministic_for_same_seed(self, tmp_path):
        """Two callbacks with the same seed must pick the same subset indices."""
        from proteinfoundation.callbacks.protein_train_eval import (
            TrainSubsetRmsdCallback,
        )

        train_ds = _make_train_ds(n_proteins=20)
        trainer = _make_trainer(train_ds)
        pl_module = _make_pl_module()

        def run_callback(run_name):
            cb = TrainSubsetRmsdCallback(
                run_name=run_name, n_train_proteins=5, nsamples=1, seed=123
            )
            cb._tmp_dir = str(tmp_path / run_name)
            os.makedirs(cb._tmp_dir, exist_ok=True)
            with (
                mock.patch(
                    "proteinfoundation.callbacks._sampling_utils.wandb.Molecule",
                    return_value="fake-mol",
                ),
                mock.patch(
                    "proteinfoundation.callbacks._sampling_utils.wandb.Table",
                    return_value=mock.MagicMock(add_data=lambda *a: None),
                ),
            ):
                cb.on_validation_epoch_end(trainer, pl_module)
            return [p["idx"] for p in cb._train_proteins]

        idxs_a = run_callback("det_a")
        idxs_b = run_callback("det_b")
        assert idxs_a == idxs_b, (
            f"Same seed must produce same subset; got {idxs_a} vs {idxs_b}"
        )

    def test_subset_differs_for_different_seed(self, tmp_path):
        from proteinfoundation.callbacks.protein_train_eval import (
            TrainSubsetRmsdCallback,
        )

        train_ds = _make_train_ds(n_proteins=20)
        trainer = _make_trainer(train_ds)
        pl_module = _make_pl_module()

        def run_callback(seed):
            cb = TrainSubsetRmsdCallback(
                run_name=f"seed{seed}", n_train_proteins=5, nsamples=1, seed=seed
            )
            cb._tmp_dir = str(tmp_path / f"seed{seed}")
            os.makedirs(cb._tmp_dir, exist_ok=True)
            with (
                mock.patch(
                    "proteinfoundation.callbacks._sampling_utils.wandb.Molecule",
                    return_value="fake-mol",
                ),
                mock.patch(
                    "proteinfoundation.callbacks._sampling_utils.wandb.Table",
                    return_value=mock.MagicMock(add_data=lambda *a: None),
                ),
            ):
                cb.on_validation_epoch_end(trainer, pl_module)
            return [p["idx"] for p in cb._train_proteins]

        # With 5 out of 20 indices the probability of collision under two
        # independent seeds is very small; treat any difference as success.
        assert run_callback(1) != run_callback(2)

    def test_log_keys_under_train_subset_prefix(self, tmp_path):
        """The aggregate log must contain train_subset/* keys, not val/*."""
        from proteinfoundation.callbacks.protein_train_eval import (
            TrainSubsetRmsdCallback,
        )

        train_ds = _make_train_ds(n_proteins=10)
        trainer = _make_trainer(train_ds, global_step=77)

        logged = []

        def capture_log(d, *args, **kwargs):
            logged.append(dict(d))

        pl_module = _make_pl_module(experiment_log=capture_log)

        cb = TrainSubsetRmsdCallback(
            run_name="prefix_test", n_train_proteins=3, nsamples=1, seed=7
        )
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch(
                "proteinfoundation.callbacks._sampling_utils.wandb.Molecule",
                return_value="fake-mol",
            ),
            mock.patch(
                "proteinfoundation.callbacks._sampling_utils.wandb.Table",
                return_value=mock.MagicMock(add_data=lambda *a: None),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        assert len(logged) == 1
        d = logged[0]
        for k in (
            "train_subset/rmsd_mf1",
            "train_subset/rmsd_mf10x",
            "train_subset/rmsd_reflected_mf1",
            "train_subset/rmsd_reflected_mf10x",
            "train_subset/chirality_mf1",
            "train_subset/chirality_mf10x",
            "samples/train_subset_table",
            "trainer/global_step",
        ):
            assert k in d, f"Missing {k}; got {list(d)}"
        # Must not leak into val/.
        stale = [k for k in d if k.startswith("val/")]
        assert not stale, f"Train-subset callback wrote under val/: {stale}"

    def test_sanity_and_rank_guards(self, tmp_path):
        from proteinfoundation.callbacks.protein_train_eval import (
            TrainSubsetRmsdCallback,
        )

        train_ds = _make_train_ds(n_proteins=10)

        pl_module = _make_pl_module()

        cb = TrainSubsetRmsdCallback(
            run_name="guards", n_train_proteins=3, nsamples=1, seed=1
        )
        cb._tmp_dir = str(tmp_path)

        # sanity_checking -> no-op
        trainer = _make_trainer(train_ds, sanity_checking=True)
        cb.on_validation_epoch_end(trainer, pl_module)
        assert cb._train_proteins is None

        # rank != 0 -> no-op
        trainer = _make_trainer(train_ds, global_rank=1)
        cb.on_validation_epoch_end(trainer, pl_module)
        assert cb._train_proteins is None

    def test_nsamples_triggers_multiple_generates(self, tmp_path):
        """nsamples=2 -> 2 mf1 + 2 mf10x calls per protein."""
        from proteinfoundation.callbacks.protein_train_eval import (
            TrainSubsetRmsdCallback,
        )

        train_ds = _make_train_ds(n_proteins=5)
        trainer = _make_trainer(train_ds, global_step=10)
        pl_module = _make_pl_module()

        generate_calls = []
        original_generate = pl_module.generate

        def counting_generate(nsamples, n, nsteps, mask=None):
            generate_calls.append(nsteps)
            return original_generate(nsamples=nsamples, n=n, nsteps=nsteps, mask=mask)

        pl_module.generate = counting_generate

        cb = TrainSubsetRmsdCallback(
            run_name="nsamples_test", n_train_proteins=1, nsamples=2, seed=3
        )
        cb._tmp_dir = str(tmp_path)

        with (
            mock.patch(
                "proteinfoundation.callbacks._sampling_utils.wandb.Molecule",
                return_value="fake-mol",
            ),
            mock.patch(
                "proteinfoundation.callbacks._sampling_utils.wandb.Table",
                return_value=mock.MagicMock(add_data=lambda *a: None),
            ),
        ):
            cb.on_validation_epoch_end(trainer, pl_module)

        # 1 protein * (2 mf1 + 2 mf10x) = 4 generate calls.
        assert len(generate_calls) == 4, f"Got {generate_calls}"
        assert generate_calls.count(1) == 2
        assert generate_calls.count(10) == 2
