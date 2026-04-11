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

import torch
import wandb
from lightning.pytorch.callbacks import Callback
from loguru import logger

from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb


class ProteinEvalCallback(Callback):
    """Periodically generate a protein with the current model weights and log it
    to WandB as a 3-D interactive ``wandb.Molecule`` visualisation.

    Also logs the ground-truth PDB/CIF once (on the first eval step) so the two
    can be compared side-by-side in the WandB dashboard.
    """

    def __init__(
        self,
        eval_every_n_steps: int,
        n_residues: int,
        ground_truth_pdb_path: str = None,
    ):
        super().__init__()
        self.eval_every_n_steps = eval_every_n_steps
        self.n_residues = n_residues
        self.ground_truth_pdb_path = ground_truth_pdb_path

        # Guards ----------------------------------------------------------
        # Tracks the last global_step at which we ran eval so that we do not
        # trigger multiple times per optimizer step when gradient accumulation
        # is active (on_train_batch_end fires per micro-batch).
        self._last_eval_step = -1
        self._gt_logged = False

    # ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Multi-GPU safety: only rank 0 should generate / log.
        if trainer.global_rank != 0:
            return

        step = trainer.global_step

        # Skip step 0 and non-eval steps.
        if step == 0 or step % self.eval_every_n_steps != 0:
            return

        # Gradient-accumulation guard: global_step only increments on an
        # optimiser step, but on_train_batch_end fires for every micro-batch.
        if step == self._last_eval_step:
            return
        self._last_eval_step = step

        try:
            logger.info(f"ProteinEvalCallback: generating protein at step {step}")

            # --- Switch to eval mode for clean generation --------------------
            was_training = pl_module.training
            pl_module.eval()

            try:
                # --- Generate a protein structure ----------------------------
                nsteps = getattr(pl_module, 'meanflow_nsteps_sample', 1)
                with torch.no_grad():
                    samples = pl_module.generate(
                        nsamples=1, n=self.n_residues, nsteps=nsteps
                    )
                    atom37 = pl_module.samples_to_atom37(samples)  # [1, n, 37, 3]
            finally:
                # --- Restore training mode -----------------------------------
                pl_module.train(was_training)

            # --- Write to a temporary PDB file -------------------------------
            fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            write_prot_to_pdb(
                atom37[0].cpu().numpy(),
                tmp_path,
                overwrite=True,
                no_indexing=True,
            )

            # --- Log generated protein to WandB -----------------------------
            trainer.logger.experiment.log(
                {"eval/generated_protein": wandb.Molecule(tmp_path)},
                step=trainer.global_step,
            )

            # --- Log ground truth once ---------------------------------------
            if (
                not self._gt_logged
                and self.ground_truth_pdb_path
                and os.path.exists(self.ground_truth_pdb_path)
            ):
                trainer.logger.experiment.log(
                    {
                        "eval/ground_truth_protein": wandb.Molecule(
                            self.ground_truth_pdb_path
                        )
                    },
                    step=trainer.global_step,
                )
                self._gt_logged = True

        except Exception as e:
            logger.warning(f"ProteinEvalCallback failed at step {step}: {e}")
            return
