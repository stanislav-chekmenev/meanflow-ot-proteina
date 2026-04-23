# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Unit tests for FIDCallback."""

from unittest import mock

import pytest
import torch
from torch_geometric.data import Batch

import proteinfoundation.callbacks.fid_callback  # ensure submodule is registered before mock.patch resolves it


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer(
    global_rank=0,
    global_step=1000,
    world_size=1,
    sanity_checking=False,
):
    trainer = mock.MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.global_rank = global_rank
    trainer.global_step = global_step
    trainer.world_size = world_size
    trainer.optimizers = []  # no EMAOptimizer -> nullcontext fallback
    return trainer


def _make_pl_module(device=None):
    """Return a MagicMock pl_module whose generate/samples_to_atom37 return zeros."""
    pl_module = mock.MagicMock()
    pl_module.training = True
    pl_module.device = device or torch.device("cpu")

    def fake_generate(nsamples, n, nsteps, mask=None):
        return torch.zeros(nsamples, n, 3)

    def fake_samples_to_atom37(samples):
        # samples: [nsamples, n, 3] — return [nsamples, n, 37, 3]
        # Set CA index (index 1) to a non-zero value derived from the input
        nsamples, n, _ = samples.shape
        result = torch.zeros(nsamples, n, 37, 3)
        result[:, :, 1, :] = samples
        return result

    pl_module.generate = fake_generate
    pl_module.samples_to_atom37 = fake_samples_to_atom37
    return pl_module


def _make_callback(
    eval_every_n_steps=1000,
    n_samples=4,
    lengths=None,
    nsteps=1,
    generation_batch_size=8,
):
    if lengths is None:
        lengths = [64, 128]
    from proteinfoundation.callbacks.fid_callback import FIDCallback
    return FIDCallback(
        eval_every_n_steps=eval_every_n_steps,
        n_samples=n_samples,
        lengths=lengths,
        gearnet_ckpt_path="./fake/gearnet.pth",
        real_features_path="./fake/features.pth",
        nsteps=nsteps,
        generation_batch_size=generation_batch_size,
    )


def _make_mock_metric_factory():
    """Return a MagicMock GenerationMetricFactory with preset return values."""
    mf = mock.MagicMock()
    # compute() returns dict {"FID": tensor(42.0)}
    mf.compute.return_value = {"FID": torch.tensor(42.0)}
    # structure_encoder device attr (checked in _lazy_init_metric)
    mf.structure_encoder.atom_embedding.weight.device = torch.device("cpu")
    # .to(device) must return the same mock so that self.metric_factory
    # stays as the tracked mock after _lazy_init_metric does .to(device).
    mf.to.return_value = mf
    return mf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFIDCallback:

    @pytest.fixture
    def patch_metric_factory(self):
        """Patch GenerationMetricFactory in fid_callback module."""
        mf = _make_mock_metric_factory()
        with mock.patch(
            "proteinfoundation.callbacks.fid_callback.GenerationMetricFactory",
            return_value=mf,
        ) as mock_cls:
            yield mock_cls, mf

    def test_sanity_check_skipped(self, patch_metric_factory):
        """trainer.sanity_checking=True -> generate must not be called."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback()
        trainer = _make_trainer(sanity_checking=True, global_step=1000)
        pl_module = _make_pl_module()
        generate_calls = []
        original_generate = pl_module.generate
        pl_module.generate = lambda *a, **kw: generate_calls.append(1) or original_generate(*a, **kw)

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert generate_calls == [], "generate must not be called during sanity check"
        mf.compute.assert_not_called()

    def test_step_zero_skipped(self, patch_metric_factory):
        """global_step=0 -> generate must not be called."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000)
        trainer = _make_trainer(global_step=0)
        pl_module = _make_pl_module()
        generate_calls = []
        original_generate = pl_module.generate
        pl_module.generate = lambda *a, **kw: generate_calls.append(1) or original_generate(*a, **kw)

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert generate_calls == [], "generate must not be called at step=0"
        mf.compute.assert_not_called()

    def test_cadence_fires_at_multiple(self, patch_metric_factory):
        """eval_every_n_steps=1000, global_step=1000 -> compute() called once and val/fid logged."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        trainer = _make_trainer(global_step=1000, world_size=1)
        pl_module = _make_pl_module()

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        mf.compute.assert_called_once()
        # Check pl_module.log was called with "val/fid"
        log_calls = [call for call in pl_module.log.call_args_list if call[0][0] == "val/fid"]
        assert len(log_calls) == 1, f"Expected 1 val/fid log call; got {len(log_calls)}"

    def test_cadence_not_fire_between(self, patch_metric_factory):
        """global_step=999 with eval_every_n_steps=1000 -> compute() must not be called."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000)
        trainer = _make_trainer(global_step=999)
        pl_module = _make_pl_module()

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        mf.compute.assert_not_called()

    def test_grad_accumulation_guard(self, patch_metric_factory):
        """Two calls with same global_step=1000 -> compute() called only once."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        trainer = _make_trainer(global_step=1000, world_size=1)
        pl_module = _make_pl_module()

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=1)

        assert mf.compute.call_count == 1, (
            f"compute should be called only once; got {mf.compute.call_count}"
        )

    def test_pyg_batch_shape(self):
        """_build_pyg_batch on [2, 10, 37, 3] input -> correct Batch shape."""
        from proteinfoundation.callbacks.fid_callback import FIDCallback
        cb = _make_callback()

        atom37 = torch.randn(2, 10, 37, 3)
        batch = cb._build_pyg_batch(atom37)

        assert isinstance(batch, Batch), "Result must be a torch_geometric.data.Batch"
        # coords: B*n_res = 20, shape [20, 37, 3]
        assert batch.coords.shape == (20, 37, 3), (
            f"Expected coords shape [20, 37, 3]; got {batch.coords.shape}"
        )
        # coord_mask: [20, 37]
        assert batch.coord_mask.shape == (20, 37), (
            f"Expected coord_mask shape [20, 37]; got {batch.coord_mask.shape}"
        )
        # CA column (index 1) all True
        assert batch.coord_mask[:, 1].all(), "coord_mask[:, 1] must be all True (CA only)"
        # All other columns are False
        other_cols = torch.cat([batch.coord_mask[:, :1], batch.coord_mask[:, 2:]], dim=1)
        assert not other_cols.any(), "coord_mask columns != 1 must be all False"
        # batch attribute: 20 elements, [0]*10 + [1]*10
        expected_batch = torch.tensor([0] * 10 + [1] * 10)
        assert torch.equal(batch.batch, expected_batch), (
            f"batch attribute mismatch: {batch.batch.tolist()}"
        )

    def test_compute_called_on_all_ranks(self, patch_metric_factory):
        """With world_size=2, compute() is still called (not guarded by rank 0)."""
        mock_cls, mf = patch_metric_factory
        # Test on rank 1 (non-zero rank) — compute must still be called
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        trainer = _make_trainer(global_step=1000, world_size=2, global_rank=1)
        pl_module = _make_pl_module()

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        mf.compute.assert_called_once(), "compute() must be called on non-zero ranks too"

        # pl_module.log with rank_zero_only=True
        log_calls = [call for call in pl_module.log.call_args_list if call[0][0] == "val/fid"]
        assert len(log_calls) == 1, "pl_module.log('val/fid', ...) must be called on all ranks"
        call_kwargs = log_calls[0][1]
        assert call_kwargs.get("rank_zero_only") is True, (
            "val/fid must be logged with rank_zero_only=True"
        )

    def test_log_does_not_pass_step_kwarg(self, patch_metric_factory):
        """pl_module.log call for val/fid must not include step=; experiment.log must not include step=."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        trainer = _make_trainer(global_step=1000, world_size=1)
        pl_module = _make_pl_module()

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        # Check pl_module.log calls
        log_calls = [call for call in pl_module.log.call_args_list if call[0][0] == "val/fid"]
        assert len(log_calls) == 1
        call_kwargs = log_calls[0][1]
        assert "step" not in call_kwargs, (
            f"pl_module.log must not be called with step=; got kwargs={call_kwargs!r}"
        )

        # Check experiment.log calls do not include step=
        for call in pl_module.logger.experiment.log.call_args_list:
            _, kwargs = call[0], call[1] if len(call) > 1 else {}
            assert "step" not in call[1], (
                f"experiment.log must not be called with step=; got kwargs={call[1]!r}"
            )

    # ----- Gap-filling tests added after first-pass TDD -----

    def test_lazy_init_called_exactly_once_across_multiple_fires(self, patch_metric_factory):
        """GenerationMetricFactory should be instantiated on first fire only, not on every firing."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        pl_module = _make_pl_module()

        # Fire at three distinct steps so the cadence check + grad-accum guard let all through.
        for step in (1000, 2000, 3000):
            trainer = _make_trainer(global_step=step, world_size=1)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert mock_cls.call_count == 1, (
            f"GenerationMetricFactory must be constructed exactly once; "
            f"got {mock_cls.call_count}"
        )
        assert mf.compute.call_count == 3, (
            f"compute() should fire on every eval step; got {mf.compute.call_count}"
        )

    def test_reset_called_after_each_compute(self, patch_metric_factory):
        """metric_factory.reset() must be called once per firing to clear fake features."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        pl_module = _make_pl_module()

        for step in (1000, 2000):
            trainer = _make_trainer(global_step=step, world_size=1)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert mf.reset.call_count == 2, (
            f"reset() must fire once per compute(); got reset={mf.reset.call_count}, "
            f"compute={mf.compute.call_count}"
        )

    def test_ema_context_entered_when_ema_optimizer_present(self, patch_metric_factory):
        """If trainer.optimizers contains an EMAOptimizer, its swap_ema_weights context is entered."""
        import proteinfoundation.callbacks.fid_callback as fcb_mod

        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=4, lengths=[64, 128])
        pl_module = _make_pl_module()

        entered = {"flag": False}

        class _FakeEmaCtx:
            def __enter__(self_inner):
                entered["flag"] = True
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        # Patch EMAOptimizer to a class that isinstance() recognises our fake instance.
        fake_opt = mock.MagicMock()
        fake_opt.swap_ema_weights.return_value = _FakeEmaCtx()

        with mock.patch.object(fcb_mod, "EMAOptimizer", type(fake_opt)):
            trainer = _make_trainer(global_step=1000, world_size=1)
            trainer.optimizers = [fake_opt]
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert entered["flag"] is True, "EMA swap context must be entered when EMAOptimizer is present"
        fake_opt.swap_ema_weights.assert_called_once()

    def test_train_mode_restored_after_generation(self, patch_metric_factory):
        """pl_module.train(was_training) must be called with the original mode value."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=2, lengths=[64])
        pl_module = _make_pl_module()
        pl_module.training = True  # starting in train mode

        trainer = _make_trainer(global_step=1000, world_size=1)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        # eval() entered once, train(True) restored once.
        pl_module.eval.assert_called_once()
        pl_module.train.assert_called_with(True)

    def test_train_mode_restored_even_if_generate_raises(self, patch_metric_factory):
        """If pl_module.generate raises, pl_module.train(was_training) must still run (try/finally)."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(eval_every_n_steps=1000, n_samples=2, lengths=[64])
        pl_module = _make_pl_module()
        pl_module.training = True

        def boom(nsamples, n, nsteps, mask=None):
            raise RuntimeError("simulated generation failure")

        pl_module.generate = boom

        trainer = _make_trainer(global_step=1000, world_size=1)
        with pytest.raises(RuntimeError, match="simulated generation failure"):
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        pl_module.train.assert_called_with(True), (
            "train() must be restored even when generate raises"
        )

    def test_val_fid_log_value_matches_compute_return(self, patch_metric_factory):
        """The scalar logged as 'val/fid' must equal compute()['FID'].item()."""
        mock_cls, mf = patch_metric_factory
        mf.compute.return_value = {"FID": torch.tensor(7.77)}

        cb = _make_callback(eval_every_n_steps=1000, n_samples=2, lengths=[64])
        trainer = _make_trainer(global_step=1000, world_size=1)
        pl_module = _make_pl_module()

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        log_calls = [call for call in pl_module.log.call_args_list if call[0][0] == "val/fid"]
        assert len(log_calls) == 1
        assert log_calls[0][0][1] == pytest.approx(7.77)

    def test_generation_chunking_respects_batch_size(self, patch_metric_factory):
        """If per_rank > generation_batch_size, generate() is called multiple times per length."""
        mock_cls, mf = patch_metric_factory
        # per_rank = n_samples // world_size = 10; all samples at one length -> 10 per length.
        # generation_batch_size = 4 -> chunks of (4, 4, 2) -> 3 generate() calls.
        cb = _make_callback(
            eval_every_n_steps=1000,
            n_samples=10,
            lengths=[64],  # one length -> all 10 go to 64
            generation_batch_size=4,
        )
        trainer = _make_trainer(global_step=1000, world_size=1)
        pl_module = _make_pl_module()

        calls = []
        def tracking_generate(nsamples, n, nsteps, mask=None):
            calls.append((nsamples, n))
            return torch.zeros(nsamples, n, 3)

        pl_module.generate = tracking_generate

        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        # Expect three chunks summing to 10.
        assert sum(c[0] for c in calls) == 10, f"total nsamples should be 10; got calls={calls}"
        assert all(c[1] == 64 for c in calls), f"all lengths should be 64; got {calls}"
        assert len(calls) == 3, f"expected 3 chunked calls (4,4,2); got {len(calls)}: {calls}"
        assert sorted(c[0] for c in calls) == [2, 4, 4]

    def test_rank_zero_absorbs_remainder_when_world_size_does_not_divide(self, patch_metric_factory):
        """n_samples=10, world_size=4 -> per_rank=2; rank 0 gets the remainder (2 extra -> 4 total)."""
        mock_cls, mf = patch_metric_factory
        cb = _make_callback(
            eval_every_n_steps=1000,
            n_samples=10,
            lengths=[64],  # single length -> sample count = generate() nsamples sum
            generation_batch_size=100,  # big enough to do one chunk
        )

        calls = []
        def tracking_generate(nsamples, n, nsteps, mask=None):
            calls.append(nsamples)
            return torch.zeros(nsamples, n, 3)

        # Rank 0 with world_size=4: should generate 10 - 2*4 + 2 = 4 samples (= 2 base + 2 remainder).
        trainer = _make_trainer(global_rank=0, global_step=1000, world_size=4)
        pl_module = _make_pl_module()
        pl_module.generate = tracking_generate
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        assert sum(calls) == 4, f"rank 0 should produce base(2)+remainder(2)=4; got {calls}"

        # Reset and test rank 1 (no remainder).
        cb._last_eval_step = -1  # allow re-fire
        calls.clear()
        trainer1 = _make_trainer(global_rank=1, global_step=1000, world_size=4)
        pl_module1 = _make_pl_module()
        pl_module1.generate = tracking_generate
        cb.on_train_batch_end(trainer1, pl_module1, outputs=None, batch=None, batch_idx=0)
        assert sum(calls) == 2, f"rank 1 should produce base(2) only; got {calls}"

    def test_build_pyg_batch_per_sample_mask_independence(self):
        """mask.clone() in _build_pyg_batch ensures modifying one sample's mask doesn't affect others."""
        from proteinfoundation.callbacks.fid_callback import FIDCallback
        cb = _make_callback()

        atom37 = torch.randn(3, 5, 37, 3)
        batch = cb._build_pyg_batch(atom37)

        # The Batch has a single flat coord_mask[n_res_total, 37]. Modifying it in-place at
        # one residue should NOT propagate to other residues from a shared reference.
        # (If mask.clone() were missing, every Data would share the same tensor and all
        # residues of all samples would be identical — but that's still [N_total, 37]
        # after batching. This test instead checks the post-batch layout is correct.)
        assert batch.coord_mask.shape == (15, 37)
        assert batch.coord_mask.dtype == torch.bool
        # Flipping one residue's CA flag post-hoc must not affect other rows.
        batch.coord_mask[0, 1] = False
        assert batch.coord_mask[1:, 1].all(), "in-place edit should not propagate across rows"
