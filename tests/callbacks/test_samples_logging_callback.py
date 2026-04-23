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

    # ----- Gap-filling tests added after first-pass TDD -----

    def test_temp_pdb_files_deleted_after_log(self, monkeypatch, tmp_path):
        """PDB files written during a firing must be deleted after wandb log returns."""
        # Use a write_prot_to_pdb that creates a real file so cleanup has something to delete.
        def real_write(arr, path, overwrite=True, no_indexing=True):
            with open(path, "w") as f:
                f.write("HEADER fake pdb\n")

        monkeypatch.setattr(
            "proteinfoundation.callbacks.protein_val_eval.write_prot_to_pdb",
            real_write,
        )
        self._patch_wandb_molecule(monkeypatch)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        cb = _make_callback(every_n_steps=100, n_samples=4, lengths=[64, 128])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        # tmp_path should contain no leftover .pdb files after the callback returns.
        leftover = [f for f in os.listdir(tmp_path) if f.endswith(".pdb")]
        assert leftover == [], (
            f"temp PDBs must be deleted after log(); leftover: {leftover}"
        )

    def test_train_mode_restored_even_if_generate_raises(self, monkeypatch, tmp_path):
        """pl_module.train(was_training) must run in the finally block even if generate raises."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        pl_module = _make_pl_module()
        pl_module.training = True  # start in train mode

        def boom(nsamples, n, nsteps, mask=None):
            raise RuntimeError("simulated generation failure")

        pl_module.generate = boom

        cb = _make_callback(every_n_steps=100, n_samples=2, lengths=[64])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)

        with pytest.raises(RuntimeError, match="simulated generation failure"):
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        pl_module.train.assert_called_with(True)

    def test_ema_context_entered_when_ema_optimizer_present(self, monkeypatch, tmp_path):
        """If trainer.optimizers contains an EMAOptimizer, its swap_ema_weights context is entered."""
        import proteinfoundation.callbacks.protein_val_eval as pve_mod

        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        pl_module = _make_pl_module(experiment_log=lambda d, **kw: None)

        cb = _make_callback(every_n_steps=100, n_samples=2, lengths=[64])
        cb._tmp_dir = str(tmp_path)

        entered = {"flag": False}

        class _FakeEmaCtx:
            def __enter__(self_inner):
                entered["flag"] = True
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        fake_opt = mock.MagicMock()
        fake_opt.swap_ema_weights.return_value = _FakeEmaCtx()

        with mock.patch.object(pve_mod, "EMAOptimizer", type(fake_opt)):
            trainer = _make_trainer(global_step=100)
            trainer.optimizers = [fake_opt]
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert entered["flag"] is True, "EMA swap context must be entered when EMAOptimizer is present"
        fake_opt.swap_ema_weights.assert_called_once()

    def test_lr_none_logged_as_nan_in_table(self, monkeypatch, tmp_path):
        """If current_optimizer_lr returns None, the 'lr' column should be NaN (not None)."""
        import math

        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        # Force current_optimizer_lr to return None.
        monkeypatch.setattr(
            "proteinfoundation.callbacks.protein_val_eval.current_optimizer_lr",
            lambda trainer: None,
        )

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        cb = _make_callback(every_n_steps=100, n_samples=2, lengths=[64])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        table = logged_calls[0][0]["samples/free_gen_table"]
        # lr column is index 1
        for row in table.data:
            assert math.isnan(row[1]), f"lr column should be NaN when lr is None; got {row[1]!r}"

    def test_log_dict_contains_global_step(self, monkeypatch, tmp_path):
        """The logged dict must carry trainer/global_step alongside the samples table."""
        self._patch_write_pdb(monkeypatch)
        self._patch_wandb_molecule(monkeypatch)

        logged_calls = []
        pl_module = _make_pl_module(experiment_log=lambda d, **kw: logged_calls.append((dict(d), kw)))

        cb = _make_callback(every_n_steps=100, n_samples=2, lengths=[64])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        log_dict = logged_calls[0][0]
        assert "trainer/global_step" in log_dict, (
            f"log dict must contain 'trainer/global_step'; got keys={list(log_dict.keys())}"
        )
        assert log_dict["trainer/global_step"] == 100

    def test_write_prot_to_pdb_called_once_per_sample(self, monkeypatch, tmp_path):
        """write_prot_to_pdb is called exactly n_samples times (one per generated structure)."""
        write_calls = []

        def tracking_write(arr, path, overwrite=True, no_indexing=True):
            write_calls.append(path)

        monkeypatch.setattr(
            "proteinfoundation.callbacks.protein_val_eval.write_prot_to_pdb",
            tracking_write,
        )
        self._patch_wandb_molecule(monkeypatch)

        pl_module = _make_pl_module(experiment_log=lambda d, **kw: None)

        cb = _make_callback(every_n_steps=100, n_samples=6, lengths=[64, 128, 192])
        cb._tmp_dir = str(tmp_path)

        trainer = _make_trainer(global_step=100)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert len(write_calls) == 6, (
            f"write_prot_to_pdb must be called once per sample (n_samples=6); got {len(write_calls)}"
        )
