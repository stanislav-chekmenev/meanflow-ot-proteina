# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import random
import time

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from loguru import logger


def log_metrics(pl_module, metrics):
    for m in metrics:
        pl_module.log(
            m, metrics[m], on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


class CheckGradientsCallback(Callback):
    def on_after_backward(self, trainer, pl_module):
        print("Checking for parameters without gradients:")
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                print(f"Parameter without gradient: {name}")


class StartupInfoCallback(Callback):
    """Logs the training dataloader length and optimizer-steps-per-epoch at
    training start (guardrail #3 for the RMSD-plateau hypothesis sweep).

    ``accumulate_grad_batches`` is silently a no-op when there is only one
    micro-batch per epoch (Lightning 2.5 flushes optimizer state on the final
    batch of every epoch). Logging the dataloader length at startup makes this
    behaviour visible in job logs so it cannot silently bite hypothesis runs.
    """

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            try:
                dl = trainer.train_dataloader
                dl_len = len(dl) if dl is not None else -1
            except Exception as e:
                logger.warning(f"StartupInfoCallback: could not read len(train_dataloader): {e}")
                dl_len = -1
            accum = getattr(trainer, "accumulate_grad_batches", 1) or 1
            # With 1 micro-batch/epoch and single-GPU, 1 epoch == 1 optimizer
            # step regardless of accumulate_grad_batches (Lightning flushes on
            # the final batch). Log the expected number to aid diagnosis.
            if dl_len <= 0:
                opt_steps_per_epoch_msg = "unknown"
            else:
                # PL 2.5 force-steps on the final batch, so the effective
                # steps/epoch is the ceiling of dl_len / accum, bounded below
                # by 1.
                eff = max(1, (dl_len + accum - 1) // accum)
                opt_steps_per_epoch_msg = f"{eff} (dl_len={dl_len}, accum={accum})"
            logger.info(
                f"StartupInfoCallback: len(train_dataloader)={dl_len} | "
                f"optimizer_steps_per_epoch={opt_steps_per_epoch_msg}"
            )


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log"""

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        pl_module.log(
            "train_info/epoch_duration_secs",
            duration,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        if pl_module.current_epoch % 10 == 0:
            logger.info(
                f"Done training epoch {pl_module.current_epoch}, epoch took {duration} seconds"
            )


class LogSetpTimeCallback(Callback):
    """Simple callback that logs how long each training step takes, in seconds, to a pytorch lightning log"""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        curr_time = time.time()
        duration = curr_time - self.step_start
        pl_module.log(
            "train_info/step_duration_secs",
            duration,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )


class GradAndWeightAnalysisCallback(Callback):
    """Some functionality to observe how are weights and gradientsbehaving during trainnig."""

    def __init__(self, debug=True):
        super(GradAndWeightAnalysisCallback, self).__init__()
        self.debug = debug

    def _get_avg_and_max_w(self, pl_module):
        """Computes average and max weight in module."""
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(pl_module.parameters()).abs()
            return params.sum() / params.numel(), params.max()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Before updating log max and average weight."""
        avg_w, max_w = self._get_avg_and_max_w(pl_module)

        if self.debug and (avg_w.isnan().any() or max_w.isnan().any()):
            print("w (bef opt step) is nan", avg_w.isnan().any(), max_w.isnan().any())
            params = torch.nn.utils.parameters_to_vector(pl_module.parameters()).abs()
            print("num NaNs w (bef opt step)", params.isnan().sum(), params.numel())

        metrics = {
            "avg_w_bef_step": avg_w,
            "max_w_bef_step": max_w,
        }
        log_metrics(pl_module, metrics)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """First thing on training step."""
        avg_w, max_w = self._get_avg_and_max_w(pl_module)

        if self.debug and (avg_w.isnan().any() or max_w.isnan().any()):
            print("w (batch start) is nan", avg_w.isnan().any(), max_w.isnan().any())

    def _get_avg_and_max_grad(self, pl_module):
        """Computes average and max grad in module, if no grad computed zero."""
        grad_sum = torch.tensor(0.0, device=pl_module.device)
        max_grad = torch.tensor(0.0, device=pl_module.device)
        count = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                abs_grad = p.grad.abs()
                grad_sum += abs_grad.sum()
                max_grad = torch.max(max_grad, abs_grad.max())
                count += p.grad.numel()
        if count == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        return grad_sum / count, max_grad

    def _count_nan_grad(self, pl_module):
        numels, num_nans = 0, 0
        for p in pl_module.parameters():
            if p.grad is not None:
                numels += p.grad.numel()
                num_nans += p.grad.isnan().sum().item()
        return numels, num_nans

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """Before zero-ing grad log max and average grad
        (should be shifted one back from next)."""
        avg_g, max_g = self._get_avg_and_max_grad(pl_module)

        if self.debug and (avg_g.isnan().any() or max_g.isnan().any()):
            print("g (bef zero g) is nan", avg_g.isnan().any(), max_g.isnan().any())

        metrics = {
            "avg_g_bef_zerog": avg_g,
            "max_g_bef_zerog": max_g,
        }
        log_metrics(pl_module, metrics)

    def on_after_backward(self, trainer, pl_module):
        """After computing gradient log max and average grad.
        This same value should be returned by <on_before_zxero_grad>
        in the next iteration."""
        avg_g, max_g = self._get_avg_and_max_grad(pl_module)
        numels, num_nans = self._count_nan_grad(pl_module)

        if self.debug and (avg_g.isnan().any() or max_g.isnan().any()):
            print("g (after bwd) is nan", avg_g.isnan().any(), max_g.isnan().any())
            print("#els, #nans in grad (after bwd)", numels, num_nans)

        metrics = {
            "avg_g_after_bwd": avg_g,
            "max_g_after_bwd": max_g,
        }
        log_metrics(pl_module, metrics)


class SkipNanGradCallback(Callback):
    """Callback to skip gradient updates with NaN in them."""

    def __init__(self, debug=True):
        super(SkipNanGradCallback, self).__init__()
        self.count = 0
        self.iter = 0

    def on_after_backward(self, trainer, pl_module):
        nan_flag = False
        self.iter += 1
        for p in pl_module.parameters():
            if p.grad is not None:
                if p.grad.isnan().any():
                    nan_flag = True
        if nan_flag:
            self.count += 1
            print("Nan grad, skipping update", self.count, self.iter)
            print(pl_module.losses[-5:])
            pl_module.zero_grad()


class RandomStateCheckpoint(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["random_state"] = random.getstate()
        checkpoint["np_random_state"] = np.random.get_state()
        checkpoint["torch_random_state"] = torch.get_rng_state()

        if torch.cuda.is_available():
            checkpoint["cuda_rng_states"] = {}
            for device_idx in range(torch.cuda.device_count()):
                checkpoint["cuda_rng_states"][device_idx] = torch.cuda.get_rng_state(
                    device_idx
                )

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["np_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])

        if torch.cuda.is_available() and "cuda_rng_states" in checkpoint:
            for device_idx, state in checkpoint["cuda_rng_states"].items():
                torch.cuda.set_rng_state(state, device_idx)
