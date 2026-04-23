# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import tempfile
from contextlib import nullcontext
from typing import List

import lightning as L
import torch
import wandb

from proteinfoundation.callbacks._sampling_utils import current_optimizer_lr
from proteinfoundation.utils.ema_utils.ema_callback import EMAOptimizer
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb


class SamplesLoggingCallback(L.Callback):
    """Periodically generate free (unconditional) protein samples and log them to WandB.

    Fires at ``on_train_batch_end`` every ``every_n_steps`` optimizer steps on
    rank 0 only. Generates ``n_samples`` structures distributed across
    ``lengths`` (round-robin), writes each to a temp PDB, and logs a
    ``wandb.Table`` with columns ``[global_step, lr, length, image]`` under
    the key ``samples/free_gen_table``.
    """

    def __init__(
        self,
        every_n_steps: int,
        n_samples: int,
        lengths: List[int],
        nsteps: int = 1,
        run_name: str = "run",
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.lengths = list(lengths)
        self.nsteps = nsteps
        self.run_name = run_name

        self._last_eval_step = -1

        # Tmp dir for PDB files
        self._tmp_dir = os.path.join("tmp", "samples", run_name)
        os.makedirs(self._tmp_dir, exist_ok=True)

    def _get_ema_context(self, trainer):
        """Return a context manager that swaps model weights to EMA.

        Falls back to nullcontext if no EMAOptimizer is found.
        """
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                return optimizer.swap_ema_weights()
        return nullcontext()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Skip during Lightning's sanity check
        if trainer.sanity_checking:
            return

        # Only rank 0 generates and logs
        if trainer.global_rank != 0:
            return

        step = trainer.global_step

        # Gradient-accumulation guard: on_train_batch_end fires per micro-batch;
        # global_step only increments on an actual optimizer step.
        if step == self._last_eval_step:
            return

        if step % self.every_n_steps != 0:
            return

        self._last_eval_step = step

        lr = current_optimizer_lr(trainer)

        # Distribute n_samples round-robin across lengths.
        # e.g. lengths=[64,128,192,256], n_samples=8 -> [64,128,192,256,64,128,192,256]
        sample_lengths: List[int] = []
        for i in range(self.n_samples):
            sample_lengths.append(self.lengths[i % len(self.lengths)])

        # Generate in eval mode with EMA weights
        was_training = pl_module.training
        pl_module.eval()

        pdb_pairs: List[tuple] = []  # (length, pdb_path)
        try:
            ema_ctx = self._get_ema_context(trainer)
            with ema_ctx, torch.no_grad():
                for n_res in sample_lengths:
                    samples = pl_module.generate(nsamples=1, n=n_res, nsteps=self.nsteps)
                    atom37 = pl_module.samples_to_atom37(samples).detach().cpu().numpy()

                    # Write to a uniquely named temp file to avoid race conditions
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdb", delete=False, dir=self._tmp_dir
                    ) as f:
                        pdb_path = f.name

                    write_prot_to_pdb(atom37[0], pdb_path, overwrite=True, no_indexing=True)
                    pdb_pairs.append((n_res, pdb_path))
        finally:
            pl_module.train(was_training)

        # Build and log wandb.Table
        table = wandb.Table(columns=["global_step", "lr", "length", "image"])
        for length, pdb_path in pdb_pairs:
            table.add_data(
                step,
                lr if lr is not None else float("nan"),
                length,
                wandb.Molecule(pdb_path),
            )

        pl_module.logger.experiment.log(
            {"samples/free_gen_table": table, "trainer/global_step": step},
            commit=False,
        )
