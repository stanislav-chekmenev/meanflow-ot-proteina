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

import torch
import wandb
from lightning.pytorch.callbacks import Callback
from loguru import logger

from proteinfoundation.callbacks.protein_eval import ProteinEvalCallback
from proteinfoundation.callbacks._sampling_utils import (
    aggregate_log_dict,
    build_samples_table,
    current_optimizer_lr,
    extract_gt_ca,
    score_protein,
    write_gt_pdb,
)

# Project-level tmp directory (mirrors protein_eval.py convention).
_PROJECT_TMP_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tmp")


def _get_ema_context(trainer):
    """Return a context manager that swaps model weights to EMA.

    Delegates to ProteinEvalCallback._get_ema_context which checks for
    EMAOptimizer in trainer.optimizers and falls back to nullcontext.
    """
    return ProteinEvalCallback._get_ema_context(trainer)


class ProteinValEvalCallback(Callback):
    """Validation callback that generates MeanFlow samples for fixed val proteins.

    Fires on on_validation_epoch_end. For each of the first n_val_proteins
    proteins in the validation dataset, generates both a 1-step and a 10-step
    MeanFlow sample, computes RMSD and chirality against ground truth, and logs
    per-protein Molecule visualisations plus aggregate scalar metrics to wandb.
    """

    def __init__(self, run_name: str, n_val_proteins: int = 16, nsamples: int = 1):
        """Construct the callback.

        Args:
            run_name: used to name the per-run tmp directory so parallel runs
                do not overwrite each other.
            n_val_proteins: how many val proteins to track.
            nsamples: number of generations per protein per mode; metrics are
                averaged over draws to reduce single-draw variance. The PDB
                written to the wandb table is the draw with lowest rmsd_mf1.
        """
        super().__init__()
        self._run_name = run_name
        self._n_val_proteins = n_val_proteins
        self._nsamples = max(1, int(nsamples))

        # Per-run tmp dir: tmp/<run_name>/val/
        project_tmp = os.path.normpath(_PROJECT_TMP_DIR)
        self._tmp_dir = os.path.join(project_tmp, run_name, "val")
        os.makedirs(self._tmp_dir, exist_ok=True)

        # Cache: populated on first real validation round.
        self._val_proteins = None  # list of dicts once populated

    # ------------------------------------------------------------------
    def _init_val_proteins(self, trainer):
        """Populate self._val_proteins from the val dataset and write GT PDBs.

        Called once on the first real on_validation_epoch_end. GT molecules
        themselves are not logged here — they are emitted as rows of the
        per-round `samples/protein_table` wandb.Table in
        `on_validation_epoch_end`.
        """
        val_ds = trainer.datamodule.val_ds
        n = min(self._n_val_proteins, len(val_ds))
        proteins = []

        for i in range(n):
            graph = val_ds[i]
            pid = graph.id

            gt_ca, n_res = extract_gt_ca(graph)
            if n_res == 0:
                logger.warning(f"Skipping val protein {pid}: no valid CA atoms")
                continue

            gt_path = write_gt_pdb(gt_ca, pid, self._tmp_dir)
            proteins.append(
                {
                    "id": pid,
                    "gt_ca": gt_ca,
                    "n_res": n_res,
                    "gt_pdb_path": gt_path,
                }
            )

        if not proteins:
            logger.warning(
                "ProteinValEvalCallback: no valid val proteins found (all had n_res=0);"
                " will retry on next val round"
            )
            return

        self._val_proteins = proteins
        logger.info(
            f"ProteinValEvalCallback: initialised {len(proteins)} val proteins"
        )

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate samples for cached val proteins and log metrics to wandb."""
        # Sanity-check guard — Lightning runs a 2-batch sanity check before
        # training; we must not generate here as the model is untrained.
        if trainer.sanity_checking:
            return

        # Multi-GPU guard: only rank 0 logs to wandb.
        if trainer.global_rank != 0:
            return

        # Lazy init: populate val protein cache on first real call.
        if self._val_proteins is None:
            try:
                self._init_val_proteins(trainer)
            except Exception:
                logger.exception(
                    "ProteinValEvalCallback: failed during lazy init; skipping"
                )
                return

        step = trainer.global_step
        logger.info(
            f"ProteinValEvalCallback: generating val proteins at step {step}"
        )

        try:
            was_training = pl_module.training
            pl_module.eval()
            ema_ctx = _get_ema_context(trainer)

            per_protein_results = []
            try:
                with ema_ctx, torch.no_grad():
                    nsteps_base = getattr(pl_module, "meanflow_nsteps_sample", 1)
                    for protein in self._val_proteins:
                        per_protein_results.append(
                            score_protein(
                                pl_module,
                                protein,
                                nsamples=self._nsamples,
                                nsteps_base=nsteps_base,
                                tmp_dir=self._tmp_dir,
                                step=step,
                            )
                        )
            finally:
                pl_module.train(was_training)

            # --- Log to wandb in a single batched call ---
            current_lr = current_optimizer_lr(trainer)
            log_dict = aggregate_log_dict(
                "val", self._val_proteins, per_protein_results, step
            )
            log_dict["samples/protein_table"] = build_samples_table(
                self._val_proteins, per_protein_results, step, current_lr
            )
            log_dict["trainer/global_step"] = step

            if pl_module.logger is not None:
                pl_module.logger.experiment.log(log_dict, commit=False)

            logger.info(
                f"ProteinValEvalCallback: logged metrics for "
                f"{len(per_protein_results)} proteins at step {step} | "
                f"rmsd_mf1={log_dict['val/rmsd_mf1']:.2f} "
                f"rmsd_mf10x={log_dict['val/rmsd_mf10x']:.2f}"
            )

        except Exception:
            logger.exception(
                f"ProteinValEvalCallback: failed at step {step}; skipping"
            )
            return
