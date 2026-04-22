# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Train-subset eval callback: mirrors ProteinValEvalCallback on a fixed random
subset of the **training** dataset, so train-subset RMSD can be compared
directly against val RMSD under identical sampling code.

The gap between these two curves is the direct overfitting signal on this
setup. When val RMSD plateaus but train-subset RMSD keeps decreasing, the
model has memorised the training distribution.
"""

import os

import numpy as np
import torch
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

_PROJECT_TMP_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tmp")


def _get_ema_context(trainer):
    """Re-use the same EMA swap context used by ProteinEvalCallback."""
    return ProteinEvalCallback._get_ema_context(trainer)


class TrainSubsetRmsdCallback(Callback):
    """Samples from a fixed random subset of the training set on every
    validation epoch end and logs RMSD/chirality under `train_subset/*`.

    The subset indices are chosen deterministically from
    ``np.random.default_rng(seed).choice`` so the curve is comparable across
    restarts with the same seed. The callback uses the same scoring code as
    ProteinValEvalCallback (shared via _sampling_utils.score_protein) to
    guarantee no implementation drift between val and train-subset metrics.
    """

    def __init__(
        self,
        run_name: str,
        n_train_proteins: int = 16,
        nsamples: int = 1,
        seed: int = 42,
    ):
        """Construct the callback.

        Args:
            run_name: used to name the per-run tmp directory.
            n_train_proteins: subset size. Default matches n_val_proteins=16;
                bump in lockstep with n_val_proteins if you want 1-to-1 variance.
            nsamples: draws per protein per mode (mirrors val callback).
            seed: RNG seed for subset index selection. Fixed default so a
                re-started run picks the *same* subset; override only when
                you deliberately want a different subset.
        """
        super().__init__()
        self._run_name = run_name
        self._n_train_proteins = n_train_proteins
        self._nsamples = max(1, int(nsamples))
        self._seed = int(seed)

        project_tmp = os.path.normpath(_PROJECT_TMP_DIR)
        self._tmp_dir = os.path.join(project_tmp, run_name, "train_subset")
        os.makedirs(self._tmp_dir, exist_ok=True)

        self._train_proteins = None  # list of dicts once populated

    # ------------------------------------------------------------------
    def _init_train_proteins(self, trainer):
        """Pick a deterministic subset of the train set and cache GT CA coords.

        Called once, lazily, on the first real on_validation_epoch_end. We
        read directly from ``trainer.datamodule.train_ds[i]`` rather than
        iterating the dataloader so the training data order is not perturbed
        and no OT-pool state is touched.
        """
        train_ds = trainer.datamodule.train_ds
        total = len(train_ds)
        n = min(self._n_train_proteins, total)
        if n == 0:
            logger.warning(
                "TrainSubsetRmsdCallback: train_ds is empty; disabling callback"
            )
            return

        rng = np.random.default_rng(self._seed)
        idxs = rng.choice(total, size=n, replace=False)
        proteins = []
        for i in sorted(int(x) for x in idxs):
            graph = train_ds[i]
            pid = graph.id
            gt_ca, n_res = extract_gt_ca(graph)
            if n_res == 0:
                logger.warning(
                    f"TrainSubsetRmsdCallback: skipping train protein {pid} "
                    f"(idx={i}): no valid CA atoms"
                )
                continue
            gt_path = write_gt_pdb(gt_ca, pid, self._tmp_dir)
            proteins.append(
                {
                    "id": pid,
                    "idx": i,
                    "gt_ca": gt_ca,
                    "n_res": n_res,
                    "gt_pdb_path": gt_path,
                }
            )

        if not proteins:
            logger.warning(
                "TrainSubsetRmsdCallback: no valid train proteins in subset;"
                " will retry on next round"
            )
            return

        self._train_proteins = proteins
        logger.info(
            f"TrainSubsetRmsdCallback: initialised {len(proteins)} train "
            f"proteins (seed={self._seed}, first idx={proteins[0]['idx']})"
        )

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate samples for the cached train subset and log to wandb."""
        if trainer.sanity_checking:
            return
        if trainer.global_rank != 0:
            return

        if self._train_proteins is None:
            try:
                self._init_train_proteins(trainer)
            except Exception:
                logger.exception(
                    "TrainSubsetRmsdCallback: failed during lazy init; skipping"
                )
                return
            if self._train_proteins is None:
                return

        step = trainer.global_step
        logger.info(
            f"TrainSubsetRmsdCallback: generating train-subset proteins at step {step}"
        )

        try:
            was_training = pl_module.training
            pl_module.eval()
            ema_ctx = _get_ema_context(trainer)

            per_protein_results = []
            try:
                with ema_ctx, torch.no_grad():
                    nsteps_base = getattr(pl_module, "meanflow_nsteps_sample", 1)
                    for protein in self._train_proteins:
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

            current_lr = current_optimizer_lr(trainer)
            log_dict = aggregate_log_dict(
                "train_subset", self._train_proteins, per_protein_results, step
            )
            log_dict["samples/train_subset_table"] = build_samples_table(
                self._train_proteins, per_protein_results, step, current_lr
            )
            log_dict["trainer/global_step"] = step

            if pl_module.logger is not None:
                pl_module.logger.experiment.log(log_dict, commit=False)

            logger.info(
                f"TrainSubsetRmsdCallback: logged metrics for "
                f"{len(per_protein_results)} proteins at step {step} | "
                f"rmsd_mf1={log_dict['train_subset/rmsd_mf1']:.2f} "
                f"rmsd_mf10x={log_dict['train_subset/rmsd_mf10x']:.2f}"
            )

        except Exception:
            logger.exception(
                f"TrainSubsetRmsdCallback: failed at step {step}; skipping"
            )
            return
