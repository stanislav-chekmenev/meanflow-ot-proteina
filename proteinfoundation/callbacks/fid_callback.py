# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FIDCallback: log GearNet-based FID during training every N optimizer steps."""

from contextlib import nullcontext
from typing import List, Optional

import lightning as L
import torch
from torch_geometric.data import Batch, Data

from proteinfoundation.metrics.metric_factory import GenerationMetricFactory
from proteinfoundation.utils.ema_utils.ema_callback import EMAOptimizer


class FIDCallback(L.Callback):
    """Log Fréchet Inception Distance (FID / FPSD) to WandB every N optimizer steps.

    Each rank generates ``n_samples // world_size`` free-generation samples with
    1-step MeanFlow, encodes them via GearNet (CA-only), and torchmetrics'
    ``dist_reduce_fx="sum"`` aggregates features across ranks at ``compute()``.

    Real reference features are loaded once from ``real_features_path``
    (pre-computed PDB eval set, Proteína-paper convention) and survive across
    ``reset()`` cycles thanks to ``reset_real_features=False``.
    """

    def __init__(
        self,
        eval_every_n_steps: int,
        n_samples: int,
        lengths: List[int],
        gearnet_ckpt_path: str,
        real_features_path: str,
        nsteps: int = 1,
        generation_batch_size: int = 32,
    ):
        super().__init__()
        self.eval_every_n_steps = eval_every_n_steps
        self.n_samples = n_samples
        self.lengths = list(lengths)
        self.gearnet_ckpt_path = gearnet_ckpt_path
        self.real_features_path = real_features_path
        self.nsteps = nsteps
        self.generation_batch_size = generation_batch_size

        self._last_eval_step: int = -1
        self.metric_factory: Optional[GenerationMetricFactory] = None

    def _get_ema_context(self, trainer):
        """Return a context manager that swaps model weights to EMA.

        Falls back to nullcontext if no EMAOptimizer is found.
        """
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                return optimizer.swap_ema_weights()
        return nullcontext()

    def _lazy_init_metric(self, device):
        """Instantiate GenerationMetricFactory on first fire and move to device.

        Delayed so that: (1) GearNet checkpoint I/O doesn't slow startup when
        fid.enabled=False, and (2) GearNet is moved to the correct device after
        Lightning has set it.
        """
        self.metric_factory = GenerationMetricFactory(
            metrics=["FID"],
            ckpt_path=self.gearnet_ckpt_path,
            ca_only=True,
            reset_real_features=False,
            real_features_path=self.real_features_path,
        )
        self.metric_factory = self.metric_factory.to(device)

    def _build_pyg_batch(self, atom37_batch: torch.Tensor) -> Batch:
        """Build a PyG Batch from an atom37 tensor without writing to disk.

        Args:
            atom37_batch: ``[B, n_res, 37, 3]`` in Angstrom (output of
                ``pl_module.samples_to_atom37``).

        Returns:
            A ``torch_geometric.data.Batch`` with ``coords``, ``coord_mask``,
            ``node_id``, and the standard ``.batch`` attribute.
        """
        B, n_res = atom37_batch.shape[0], atom37_batch.shape[1]
        # coord_mask: only CA (openfold index 1) is True
        mask = torch.zeros((n_res, 37), dtype=torch.bool, device=atom37_batch.device)
        mask[:, 1] = True  # CA only

        data_list = []
        for b in range(B):
            d = Data(
                coords=atom37_batch[b],  # [n_res, 37, 3]
                coord_mask=mask.clone(),  # [n_res, 37] bool
                node_id=torch.arange(n_res, device=atom37_batch.device).unsqueeze(-1),
            )
            data_list.append(d)
        return Batch.from_data_list(data_list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1. Skip during Lightning's sanity check
        if trainer.sanity_checking:
            return

        step = trainer.global_step

        # 2a. Never fire at step 0 (model not yet trained)
        if step == 0:
            return

        # 2b. Gradient-accumulation guard: on_train_batch_end fires per
        # micro-batch; global_step only increments on an actual optimizer step.
        if step == self._last_eval_step:
            return

        # 2c. Cadence check
        if step % self.eval_every_n_steps != 0:
            return

        # 3. Record that we're evaluating at this step
        self._last_eval_step = step

        # 4. Lazy-init metric factory on first fire
        if self.metric_factory is None:
            self._lazy_init_metric(pl_module.device)

        # 5. Determine per-rank sample count
        world_size = trainer.world_size
        per_rank = self.n_samples // world_size
        # Rank 0 absorbs the remainder
        if trainer.global_rank == 0:
            per_rank += self.n_samples - per_rank * world_size

        # 6. Distribute per_rank samples across lengths round-robin
        per_rank_lengths: List[int] = []
        for i in range(per_rank):
            per_rank_lengths.append(self.lengths[i % len(self.lengths)])

        # 7. Generate samples in eval+no_grad+EMA context
        was_training = pl_module.training
        pl_module.eval()

        try:
            ema_ctx = self._get_ema_context(trainer)
            with ema_ctx, torch.no_grad():
                # Group by length and process in chunks of generation_batch_size
                # Build list of (length, count) by grouping the round-robin list.
                length_counts = {}
                for ln in per_rank_lengths:
                    length_counts[ln] = length_counts.get(ln, 0) + 1

                for length, count in length_counts.items():
                    processed = 0
                    while processed < count:
                        chunk_size = min(self.generation_batch_size, count - processed)
                        samples = pl_module.generate(
                            nsamples=chunk_size, n=length, nsteps=self.nsteps
                        )  # [chunk_size, length, 3] nm
                        atom37 = pl_module.samples_to_atom37(samples)  # [chunk_size, length, 37, 3] Å
                        pyg_batch = self._build_pyg_batch(atom37).to(pl_module.device)
                        self.metric_factory.update(pyg_batch, real=False)
                        processed += chunk_size
        finally:
            pl_module.train(was_training)

        # 8. Compute on ALL ranks — torchmetrics DDP all-reduce fires here.
        # MUST NOT be guarded by rank 0 to avoid NCCL deadlock.
        results = self.metric_factory.compute()

        # 9. Extract FID value
        fid_value = results["FID"].item()

        # 10. Reset fake-side; real features are preserved (reset_real_features=False)
        self.metric_factory.reset()

        # 11. Log: route through pl_module.log so GlobalStepWandbLogger picks up
        # the correct trainer.global_step x-axis (per project_wandb_global_step_fix).
        # on_step=True is required because on_step=False + on_epoch=False is a no-op
        # in Lightning's ResultCollection — the metric would never reach WandB.
        # rank_zero_only=True means WandB only receives the value from rank 0.
        # val/fid is NOT monitored by ModelCheckpoint, so non-zero ranks not
        # recording it does not cause MisconfigurationException.
        pl_module.log(
            "val/fid",
            fid_value,
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

        # 12. Also emit trainer/global_step to WandB so the x-axis is correct
        # (mirrors SamplesLoggingCallback pattern; commit=False defers the flush).
        if trainer.global_rank == 0:
            pl_module.logger.experiment.log(
                {"trainer/global_step": step},
                commit=False,
            )
