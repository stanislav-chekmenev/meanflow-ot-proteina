"""
Tests for GlobalStepWandbLogger — a WandbLogger subclass that forces the
x-axis step metric (``trainer/global_step``) to reflect the real optimizer
global_step regardless of what step value Lightning passes in.

Background: Lightning's eval loop passes the per-dataloader val-batch counter
as the ``step`` argument to ``WandbLogger.log_metrics`` (see
``evaluation_loop.py:459``). The stock WandbLogger then writes that counter
into ``trainer/global_step`` on the wandb event, which corrupts the x-axis
for every ``val/*_step`` metric.
"""

import sys
import types
import unittest.mock as mock

# Stub torch_scatter (CUDA-only) and wandb so we can import the subclass
# without a live wandb run.
_ts_stub = types.ModuleType("torch_scatter")
_ts_stub.scatter_mean = None
sys.modules.setdefault("torch_scatter", _ts_stub)


def _make_logger_with_fake_experiment():
    """Instantiate GlobalStepWandbLogger with a fake wandb experiment that
    records every ``log(...)`` call, so we can assert what got written."""
    from proteinfoundation.train import GlobalStepWandbLogger

    logger = GlobalStepWandbLogger(project="test", name="test", offline=True)
    fake_experiment = mock.MagicMock()
    logger._experiment = fake_experiment
    return logger, fake_experiment


def test_log_metrics_overrides_step_with_trainer_global_step():
    """When a trainer is attached, log_metrics must rewrite ``step`` to
    ``trainer.global_step`` — ignoring whatever Lightning passed in."""
    logger, fake_experiment = _make_logger_with_fake_experiment()

    fake_trainer = mock.MagicMock()
    fake_trainer.global_step = 1342
    logger.attach_trainer(fake_trainer)

    # Lightning's eval loop passes the val-batch counter (e.g. 0) as ``step``.
    logger.log_metrics({"val/combined_adaptive_loss_step": 0.9995}, step=0)

    fake_experiment.log.assert_called_once()
    logged_dict = fake_experiment.log.call_args[0][0]
    assert logged_dict["trainer/global_step"] == 1342, (
        f"expected trainer/global_step=1342, got {logged_dict.get('trainer/global_step')}"
    )
    assert logged_dict["val/combined_adaptive_loss_step"] == 0.9995


def test_log_metrics_without_trainer_falls_back_to_passed_step():
    """Before the trainer is attached (e.g. during logger construction), the
    override must be a no-op so Lightning's default behavior still works."""
    logger, fake_experiment = _make_logger_with_fake_experiment()

    logger.log_metrics({"some/metric": 1.0}, step=42)

    fake_experiment.log.assert_called_once()
    logged_dict = fake_experiment.log.call_args[0][0]
    assert logged_dict["trainer/global_step"] == 42


def test_log_metrics_with_step_none_uses_trainer_global_step():
    """Some Lightning code paths call log_metrics with step=None (end-of-epoch
    flush). The override must still write trainer.global_step, not skip the
    step_metric field entirely."""
    logger, fake_experiment = _make_logger_with_fake_experiment()

    fake_trainer = mock.MagicMock()
    fake_trainer.global_step = 2687
    logger.attach_trainer(fake_trainer)

    logger.log_metrics({"val/combined_adaptive_loss_epoch": 0.998}, step=None)

    fake_experiment.log.assert_called_once()
    logged_dict = fake_experiment.log.call_args[0][0]
    assert logged_dict["trainer/global_step"] == 2687
    assert logged_dict["val/combined_adaptive_loss_epoch"] == 0.998
