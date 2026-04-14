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
from Bio.PDB import MMCIFParser, PDBParser
from lightning.pytorch.callbacks import Callback
from loguru import logger

from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

# Project-level tmp directory for easy debugging.
_PROJECT_TMP_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tmp")

# C-alpha is at index 1 in the atom37 representation.
_CA_INDEX = 1


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
        run_name: str,
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

        # Per-run tmp directory so parallel runs don't overwrite each other.
        project_tmp = os.path.normpath(_PROJECT_TMP_DIR)
        self._tmp_dir = os.path.join(project_tmp, run_name)
        os.makedirs(self._tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ca_coords(path: str) -> np.ndarray:
        """Parse a CIF or PDB file and return C-alpha coordinates ``[n, 3]``."""
        if path.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure("gt", path)
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    if "CA" in residue:
                        ca_coords.append(residue["CA"].get_coord())
            break  # first model only
        return np.array(ca_coords)

    def _prepare_gt_pdb(self) -> str | None:
        """Extract C-alpha atoms from the GT structure and write a CA-only PDB.

        This makes the GT directly comparable to the generated samples (which
        are also CA-only) and avoids CIF format issues with ``wandb.Molecule``.
        """
        if not self.ground_truth_pdb_path or not os.path.exists(self.ground_truth_pdb_path):
            return None
        ca_coords = self._extract_ca_coords(self.ground_truth_pdb_path)
        if len(ca_coords) == 0:
            return None
        atom37 = np.zeros((len(ca_coords), 37, 3), dtype=np.float32)
        atom37[:, _CA_INDEX, :] = ca_coords
        gt_path = os.path.join(self._tmp_dir, "ground_truth_ca.pdb")
        write_prot_to_pdb(atom37, gt_path, overwrite=True, no_indexing=True)
        return gt_path

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

            # --- Write to a PDB file in the project tmp dir -------------------
            tmp_path = os.path.join(self._tmp_dir, f"generated_step{step}.pdb")
            write_prot_to_pdb(
                atom37[0].cpu().numpy(),
                tmp_path,
                overwrite=True,
                no_indexing=True,
            )

            # --- Log to WandB -----------------------------------------------
            # Use commit=False so this data is merged into Lightning's next
            # log call, which carries the correct trainer/global_step.
            log_dict = {
                "eval/generated_protein": wandb.Molecule(tmp_path),
            }

            if not self._gt_logged:
                gt_path = self._prepare_gt_pdb()
                if gt_path:
                    log_dict["eval/ground_truth_protein"] = wandb.Molecule(gt_path)
                self._gt_logged = True

            pl_module.logger.experiment.log(log_dict, commit=False)

        except Exception as e:
            logger.warning(f"ProteinEvalCallback failed at step {step}: {e}")
            return
