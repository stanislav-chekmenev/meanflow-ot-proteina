# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Unit tests for SamplesLoggingCallback."""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer(global_rank=0, global_step=100, sanity_checking=False):
    trainer = mock.MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.global_rank = global_rank
    trainer.global_step = global_step
    trainer.optimizers = []  # no EMAOptimizer -> nullcontext fallback
    return trainer


def _make_pl_module(experiment_log=None):
    """Return a MagicMock pl_module whose generate/samples_to_atom37 return zeros."""
    pl_module = mock.MagicMock()
    pl_module.training = True

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
    return pl_module


def _make_callback(every_n_steps=100, n_samples=8, lengths=None, nsteps=1, run_name="test_run"):
    if lengths is None:
        lengths = [64, 128, 192, 256]
    from proteinfoundation.callbacks.protein_val_eval import SamplesLoggingCallback
    return SamplesLoggingCallback(
        every_n_steps=every_n_steps,
        n_samples=n_samples,
        lengths=lengths,
        nsteps=nsteps,
        run_name=run_name,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSamplesLoggingCallback:

    def _patch_write_pdb(self, monkeypatch):
        """Patch write_prot_to_pdb to a no-op so no real disk writes happen."""
        monkeypatch.setattr(
            "proteinfoundation.callbacks.protein_val_eval.write_prot_to_pdb",
            lambda *args, **kwargs: None,
        )

    def _patch_wandb_molecule(self, monkeypatch):
        """Patch wandb.Molecule to avoid needing a live wandb context."""
        monkeypatch.setattr(
            "proteinfoundation.callbacks.protein_val_eval.wandb.Molecule",
            lambda path: f"molecule:{path}",
        )

    def test_sanity_check_skipped(self, monkeypatch, tmp_path):
        """When sanity_checking=True, pl_module.generate must not be called."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        cb = _make_callback()
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(sanity_checking=True, global_step=100)
        pl_module = _make_pl_module()
        generate_called = []

        original_generate = pl_module.generate
        pl_module.generate = lambda *a, **kw: generate_called.append(1) or original_generate(*a, **kw)

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert generate_called == [], "generate must not be called during sanity check"

    def test_non_zero_rank_skipped(self, monkeypatch, tmp_path):
        """When global_rank != 0, pl_module.generate must not be called."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        cb = _make_callback()
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_rank=1, global_step=100)
        pl_module = _make_pl_module()
        generate_called = []

        original_generate = pl_module.generate
        pl_module.generate = lambda *a, **kw: generate_called.append(1) or original_generate(*a, **kw)

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert generate_called == [], "generate must not be called on non-zero rank"

    def test_cadence_fires_at_exact_step(self, monkeypatch, tmp_path):
        """every_n_steps=100: fires at step=100, does not fire at step=99."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        generate_counts = []

        # First: global_step=100 should fire
        cb = _make_callback(every_n_steps=100, n_samples=4, lengths=[64, 128])
        cb._tmp_dir = str(tmp_path)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append(d))

        original_generate = pl_module.generate
        def counting_generate(nsamples, n, nsteps, mask=None):
            generate_counts.append(n)
            return original_generate(nsamples=nsamples, n=n, nsteps=nsteps)
        pl_module.generate = counting_generate

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert len(generate_counts) > 0, "generate must be called at step=100"

        # Second: a fresh callback with global_step=99 should NOT fire
        cb2 = _make_callback(every_n_steps=100, n_samples=4, lengths=[64, 128])
        cb2._tmp_dir = str(tmp_path)

        generate_counts2 = []
        pl_module2 = _make_pl_module()
        original2 = pl_module2.generate
        pl_module2.generate = lambda *a, **kw: generate_counts2.append(1) or original2(*a, **kw)

        trainer2 = _make_trainer(global_step=99)
        cb2.on_train_batch_end(trainer2, pl_module2, outputs=None, batch=None, batch_idx=0)

        assert generate_counts2 == [], "generate must NOT be called at step=99 with every_n_steps=100"

    def test_grad_accumulation_guard(self, monkeypatch, tmp_path):
        """Two sequential calls with the same global_step → generate called only once."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        cb = _make_callback(every_n_steps=100, n_samples=4, lengths=[64, 128])
        cb._tmp_dir = str(tmp_path)

        generate_counts = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: None)

        original_generate = pl_module.generate
        def counting_generate(nsamples, n, nsteps, mask=None):
            generate_counts.append(n)
            return original_generate(nsamples=nsamples, n=n, nsteps=nsteps)
        pl_module.generate = counting_generate

        trainer = _make_trainer(global_step=100)

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        first_count = len(generate_counts)

        # Same step again (simulating grad accumulation micro-batch)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=1)

        assert len(generate_counts) == first_count, (
            f"generate should not be called again for same step; "
            f"first_count={first_count}, total={len(generate_counts)}"
        )

    def test_table_columns(self, monkeypatch, tmp_path):
        """After a firing, the wandb.Table must have columns ['global_step', 'lr', 'length', 'image']."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        cb = _make_callback(every_n_steps=100, n_samples=4, lengths=[64, 128])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert len(logged_calls) == 1, f"Expected exactly 1 log call, got {len(logged_calls)}"
        log_dict, _ = logged_calls[0]
        assert "samples/free_gen_table" in log_dict, "Missing key 'samples/free_gen_table'"

        table = log_dict["samples/free_gen_table"]
        assert table.columns == ["global_step", "lr", "length", "image"], (
            f"Unexpected columns: {table.columns!r}"
        )

    def test_table_row_count_equals_n_samples(self, monkeypatch, tmp_path):
        """One firing produces a table with exactly n_samples rows."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        n_samples = 8
        cb = _make_callback(every_n_steps=100, n_samples=n_samples, lengths=[64, 128, 192, 256])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert len(logged_calls) == 1
        log_dict, _ = logged_calls[0]
        table = log_dict["samples/free_gen_table"]

        assert len(table.data) == n_samples, (
            f"Expected {n_samples} rows; got {len(table.data)}"
        )

    def test_table_lengths_distributed_round_robin(self, monkeypatch, tmp_path):
        """With lengths=[64,128,192,256] and n_samples=8, length column has 2 of each."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        lengths = [64, 128, 192, 256]
        cb = _make_callback(every_n_steps=100, n_samples=8, lengths=lengths)
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert len(logged_calls) == 1
        log_dict, _ = logged_calls[0]
        table = log_dict["samples/free_gen_table"]

        # Extract length column (index 2 in columns: global_step, lr, length, image)
        length_col = [row[2] for row in table.data]

        # Round-robin: 8 samples across 4 lengths -> 2 per length
        assert sorted(length_col) == sorted([64, 64, 128, 128, 192, 192, 256, 256]), (
            f"Round-robin distribution incorrect: {length_col}"
        )

    def test_no_step_kwarg_in_log_call(self, monkeypatch, tmp_path):
        """The log call must not pass a step= kwarg."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        cb = _make_callback(every_n_steps=100, n_samples=4, lengths=[64, 128])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert len(logged_calls) == 1
        _, kwargs = logged_calls[0]
        assert "step" not in kwargs, (
            f"log must not be called with step= kwarg; got kwargs={kwargs!r}"
        )
