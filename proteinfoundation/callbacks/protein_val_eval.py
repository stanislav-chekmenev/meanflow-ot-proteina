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

import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from loguru import logger

from proteinfoundation.callbacks.protein_eval import (
    ProteinEvalCallback,
    _ca_rmsd,
    _ca_rmsd_with_reflection,
    _chirality_sign,
)
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

# Project-level tmp directory (mirrors protein_eval.py convention).
_PROJECT_TMP_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tmp")

# C-alpha is at index 1 in the atom37 representation.
_CA_INDEX = 1


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

    def __init__(self, run_name: str, n_val_proteins: int = 16):
        """Construct the callback.

        Args:
            run_name: used to name the per-run tmp directory so parallel runs
                do not overwrite each other.
            n_val_proteins: how many val proteins to track.
        """
        super().__init__()
        self._run_name = run_name
        self._n_val_proteins = n_val_proteins

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

            # Extract CA coords; drop residues where CA mask is False.
            ca_mask = graph.coord_mask[:, _CA_INDEX]  # [n_res] bool
            gt_ca = graph.coords[:, _CA_INDEX, :][ca_mask].cpu().numpy()  # [m, 3]
            n_res = gt_ca.shape[0]

            if n_res == 0:
                logger.warning(f"Skipping val protein {pid}: no valid CA atoms")
                continue

            # Write GT CA-only PDB.
            atom37 = np.zeros((n_res, 37, 3), dtype=np.float32)
            atom37[:, _CA_INDEX, :] = gt_ca
            gt_path = os.path.join(self._tmp_dir, f"gt_{pid}.pdb")
            write_prot_to_pdb(atom37, gt_path, overwrite=True, no_indexing=True)

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
                        pid = protein["id"]
                        n_res = protein["n_res"]
                        gt_ca = protein["gt_ca"]

                        # --- MF-1 generation ---
                        samples_mf1 = pl_module.generate(
                            nsamples=1, n=n_res, nsteps=nsteps_base, mask=None
                        )
                        atom37_mf1 = (
                            pl_module.samples_to_atom37(samples_mf1).cpu().numpy()
                        )  # [1, n, 37, 3]
                        gen_ca_mf1 = atom37_mf1[0, :, _CA_INDEX, :]  # [n, 3]
                        pdb_mf1 = os.path.join(
                            self._tmp_dir, f"gen_{pid}_mf1_step{step}.pdb"
                        )
                        write_prot_to_pdb(
                            atom37_mf1[0], pdb_mf1, overwrite=True, no_indexing=True
                        )

                        # --- MF-10x generation ---
                        samples_mf10x = pl_module.generate(
                            nsamples=1,
                            n=n_res,
                            nsteps=nsteps_base * 10,
                            mask=None,
                        )
                        atom37_mf10x = (
                            pl_module.samples_to_atom37(samples_mf10x).cpu().numpy()
                        )
                        gen_ca_mf10x = atom37_mf10x[0, :, _CA_INDEX, :]
                        pdb_mf10x = os.path.join(
                            self._tmp_dir, f"gen_{pid}_mf10x_step{step}.pdb"
                        )
                        write_prot_to_pdb(
                            atom37_mf10x[0], pdb_mf10x, overwrite=True, no_indexing=True
                        )

                        # --- Metrics ---
                        rmsd_mf1 = _ca_rmsd(gen_ca_mf1, gt_ca)
                        rmsd_refl_mf1 = _ca_rmsd_with_reflection(gen_ca_mf1, gt_ca)
                        chir_mf1 = _chirality_sign(gen_ca_mf1, gt_ca)

                        rmsd_mf10x = _ca_rmsd(gen_ca_mf10x, gt_ca)
                        rmsd_refl_mf10x = _ca_rmsd_with_reflection(
                            gen_ca_mf10x, gt_ca
                        )
                        chir_mf10x = _chirality_sign(gen_ca_mf10x, gt_ca)

                        per_protein_results.append(
                            {
                                "mf1_pdb": pdb_mf1,
                                "mf10x_pdb": pdb_mf10x,
                                "rmsd_mf1": rmsd_mf1,
                                "rmsd_mf10x": rmsd_mf10x,
                                "rmsd_refl_mf1": rmsd_refl_mf1,
                                "rmsd_refl_mf10x": rmsd_refl_mf10x,
                                "chir_mf1": chir_mf1,
                                "chir_mf10x": chir_mf10x,
                            }
                        )
            finally:
                pl_module.train(was_training)

            # --- Log to wandb in a single batched call ---
            log_dict = {}

            # Build the per-round protein sample table. One row per protein,
            # with protein_id + aligned RMSDs + three interactive molecule
            # viewers (GT, MF-1, MF-10x).
            table = wandb.Table(
                columns=[
                    "protein_id",
                    "rmsd_mf1",
                    "rmsd_mf10x",
                    "rmsd_reflected_mf1",
                    "rmsd_reflected_mf10x",
                    "ground_truth",
                    "mf1",
                    "mf10x",
                ]
            )
            for protein, res in zip(self._val_proteins, per_protein_results):
                table.add_data(
                    protein["id"],
                    float(res["rmsd_mf1"]),
                    float(res["rmsd_mf10x"]),
                    float(res["rmsd_refl_mf1"]),
                    float(res["rmsd_refl_mf10x"]),
                    wandb.Molecule(protein["gt_pdb_path"]),
                    wandb.Molecule(res["mf1_pdb"]),
                    wandb.Molecule(res["mf10x_pdb"]),
                )
            log_dict["samples/protein_table"] = table

            log_dict["val/rmsd_mf1"] = float(
                np.mean([r["rmsd_mf1"] for r in per_protein_results])
            )
            log_dict["val/rmsd_mf10x"] = float(
                np.mean([r["rmsd_mf10x"] for r in per_protein_results])
            )
            log_dict["val/rmsd_reflected_mf1"] = float(
                np.mean([r["rmsd_refl_mf1"] for r in per_protein_results])
            )
            log_dict["val/rmsd_reflected_mf10x"] = float(
                np.mean([r["rmsd_refl_mf10x"] for r in per_protein_results])
            )
            log_dict["val/chirality_mf1"] = float(
                np.mean([r["chir_mf1"] for r in per_protein_results])
            )
            log_dict["val/chirality_mf10x"] = float(
                np.mean([r["chir_mf10x"] for r in per_protein_results])
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
