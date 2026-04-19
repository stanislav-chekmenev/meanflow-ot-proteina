# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib
import os

import numpy as np
import torch
import wandb
from Bio.PDB import MMCIFParser, PDBParser
from lightning.pytorch.callbacks import Callback
from loguru import logger

from proteinfoundation.utils.coors_utils import ang_to_nm
from proteinfoundation.utils.ema_utils.ema_callback import EMAOptimizer
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

# Project-level tmp directory for easy debugging.
_PROJECT_TMP_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tmp")

# C-alpha is at index 1 in the atom37 representation.
_CA_INDEX = 1


def _kabsch_svd(coords_a: np.ndarray, coords_b: np.ndarray):
    """Run the Kabsch SVD and return the pieces needed for several metrics.

    Returns:
        a:     centered coords_a, shape [n, 3]
        b:     centered coords_b, shape [n, 3]
        U, Vt: SVD factors of H = a.T @ b (so H = U @ diag(S) @ Vt)
        d_opt: det(V @ U.T) — sign encodes chirality:
               > 0 means the optimal rotation aligning a->b is proper (SO(3)),
               < 0 means it requires a reflection (a is a mirror image of b).
    """
    assert coords_a.shape == coords_b.shape
    a = coords_a - coords_a.mean(axis=0)
    b = coords_b - coords_b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d_opt = float(np.linalg.det(Vt.T @ U.T))
    return a, b, U, Vt, d_opt


def _ca_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """C-alpha RMSD after proper-rotation-only Kabsch alignment (no reflection)."""
    a, b, U, Vt, d_opt = _kabsch_svd(coords_a, coords_b)
    S = np.eye(3)
    if d_opt < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    diff = a @ R.T - b
    return float(np.sqrt((diff ** 2).sum() / len(diff)))


def _ca_rmsd_with_reflection(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """RMSD after optimal O(3) alignment (reflection allowed).

    Useful as a sanity metric: if a is a mirror image of b with the correct
    shape, this should be ~1 Å while _ca_rmsd stays ~10 Å.
    """
    a, b, U, Vt, _ = _kabsch_svd(coords_a, coords_b)
    R = Vt.T @ U.T  # may be improper (det = -1) if chiralities differ
    diff = a @ R.T - b
    return float(np.sqrt((diff ** 2).sum() / len(diff)))


def _chirality_sign(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    """+1 if coords_a has the same chirality as coords_b, -1 if it's a mirror."""
    _, _, _, _, d_opt = _kabsch_svd(coords_a, coords_b)
    return 1 if d_opt > 0 else -1


class ProteinEvalCallback(Callback):
    """Periodically generate a protein with the current model weights and log it
    to WandB as a 3-D interactive ``wandb.Molecule`` visualisation.

    Generates three variants for diagnostics:
      1. MeanFlow 1-step (nsteps_sample from config)
      2. MeanFlow 10x-step (10 * nsteps_sample)
      3. FM-style ODE (100-step Euler with h=0, instantaneous velocity)

    Also logs the ground-truth PDB/CIF once (on the first eval step) so the two
    can be compared side-by-side in the WandB dashboard.
    """

    def __init__(
        self,
        eval_every_n_steps: int,
        n_residues: int,
        run_name: str,
        ground_truth_pdb_path: str = None,
        nsamples: int = 1,
    ):
        super().__init__()
        self.eval_every_n_steps = eval_every_n_steps
        self.n_residues = n_residues
        self.ground_truth_pdb_path = ground_truth_pdb_path
        # nsamples: number of generations to draw per eval call. Per-sample
        # RMSDs and chirality are averaged so the logged metric is lower-variance.
        # nsamples=1 preserves the original 1-bit chirality behaviour.
        assert nsamples >= 1, f"nsamples must be >= 1, got {nsamples}"
        self.nsamples = int(nsamples)

        # Guards ----------------------------------------------------------
        # Tracks the last global_step at which we ran eval so that we do not
        # trigger multiple times per optimizer step when gradient accumulation
        # is active (on_train_batch_end fires per micro-batch).
        self._last_eval_step = -1
        self._gt_logged = False
        self._gt_ca_coords = None  # cached for RMSD computation

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
        self._gt_ca_coords = ca_coords
        atom37 = np.zeros((len(ca_coords), 37, 3), dtype=np.float32)
        atom37[:, _CA_INDEX, :] = ca_coords
        gt_path = os.path.join(self._tmp_dir, "ground_truth_ca.pdb")
        write_prot_to_pdb(atom37, gt_path, overwrite=True, no_indexing=True)
        return gt_path

    def _get_gt_ca_coords(self) -> np.ndarray | None:
        """Return cached GT CA coordinates, parsing on first call."""
        if self._gt_ca_coords is not None:
            return self._gt_ca_coords
        if not self.ground_truth_pdb_path or not os.path.exists(self.ground_truth_pdb_path):
            return None
        self._gt_ca_coords = self._extract_ca_coords(self.ground_truth_pdb_path)
        return self._gt_ca_coords if len(self._gt_ca_coords) > 0 else None

    @staticmethod
    def _get_ema_context(trainer):
        """Return a context manager that swaps model weights to EMA.

        Falls back to a no-op if no EMAOptimizer is found (e.g. EMA disabled).
        """
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                return optimizer.swap_ema_weights()
        return contextlib.nullcontext()

    def _fm_loss_mirror_diagnostic(self, pl_module, n_noise_samples: int = 8):
        """Check that the FM loss is invariant under reflection of the target.

        For a reflection Q and matched noise, the velocity target transforms as
        ``v = x_0 - x_1  ->  Q v = (Q x_0) - (Q x_1)``. If the network is
        O(3)-equivariant in its output, ``u_pred(Q x_t) = Q u_pred(x_t)``, so
        ``||u_pred(Q x_t) - Q v||^2 = ||Q (u_pred(x_t) - v)||^2 = ||u_pred(x_t) - v||^2``.

        This function averages the FM loss over several random noises + times,
        once with the ground-truth x_1 and once with Q @ x_1 (and Q @ x_0). If
        the two averages agree within numerical noise, the loss cannot tell the
        two chiralities apart — which is why generation lands on either with
        ~50/50 probability.

        Returns (loss_orig, loss_mirror, abs_diff) or (None, None, None) if
        no ground truth is available.
        """
        gt_ca = self._get_gt_ca_coords()
        if gt_ca is None:
            return None, None, None

        device = pl_module.device
        fm = pl_module.fm
        dtype = torch.float32

        x_1 = torch.from_numpy(gt_ca).to(device=device, dtype=dtype).unsqueeze(0)  # [1, n, 3]
        x_1 = ang_to_nm(x_1)  # model works in nm; GT PDB is in Å
        n = x_1.shape[1]
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        x_1 = fm._mask_and_zero_com(x_1, mask)

        # Reflection across the xy-plane (any improper orthogonal matrix works).
        Q = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
                         device=device, dtype=dtype)
        x_1_mirror = x_1 @ Q  # Q is symmetric, so x @ Q == (Q @ x^T)^T

        # Deterministic noise + times so the two losses are directly comparable.
        gen = torch.Generator(device=device).manual_seed(42)
        ts = torch.empty(n_noise_samples, device=device, dtype=dtype).uniform_(
            0.05, 0.9, generator=gen
        )

        def _fm_single(x1, x0, t):
            t_ext = t.view(1, 1, 1)
            z_t = (1 - t_ext) * x1 + t_ext * x0
            v_true = x0 - x1
            batch_nn = {
                "x_t": z_t,
                "t": t.view(1),
                "h": torch.zeros(1, device=device, dtype=dtype),
                "mask": mask,
            }
            v_pred = pl_module.nn(batch_nn)["coors_pred"]
            v_pred = fm._mask_and_zero_com(v_pred, mask)
            nres = mask.sum().item() * 3
            return ((v_pred - v_true) ** 2 * mask[..., None]).sum().item() / nres

        losses_orig = []
        losses_mirror = []
        with torch.no_grad():
            for i in range(n_noise_samples):
                x_0 = fm.sample_reference(
                    n=n, shape=(1,), device=device, dtype=dtype, mask=mask
                )
                t = ts[i]
                losses_orig.append(_fm_single(x_1, x_0, t))
                losses_mirror.append(_fm_single(x_1_mirror, x_0 @ Q, t))

        loss_orig = float(np.mean(losses_orig))
        loss_mirror = float(np.mean(losses_mirror))
        return loss_orig, loss_mirror, abs(loss_orig - loss_mirror)

    def _generate_and_save(self, pl_module, step, label, nsteps, use_fm_euler=False):
        """Generate ``self.nsamples`` proteins, save the first PDB, and
        aggregate RMSD / chirality over the batch.

        Args:
            pl_module: the Proteina LightningModule.
            step: current global step.
            label: string label for this generation mode (e.g. "mf1", "mf10", "fm100").
            nsteps: number of integration steps.
            use_fm_euler: if True, use FM-style ODE (h=0) instead of MeanFlow.

        Returns:
            Tuple (pdb_path, rmsd_mean, rmsd_reflected_mean, chirality_mean).
            When nsamples > 1, RMSDs and chirality are averaged across the
            batch — chirality in particular becomes a real-valued estimator
            in [-1, +1] rather than a 1-bit Bernoulli sample.
        """
        nsamples = self.nsamples
        with torch.no_grad():
            if use_fm_euler:
                samples = pl_module.generate_fm_euler(
                    nsamples=nsamples, n=self.n_residues, nsteps=nsteps
                )
            else:
                samples = pl_module.generate(
                    nsamples=nsamples, n=self.n_residues, nsteps=nsteps
                )
            atom37 = pl_module.samples_to_atom37(samples)  # [nsamples, n, 37, 3]

        # Write PDB for the first sample only (WandB molecule viewer).
        pdb_path = os.path.join(self._tmp_dir, f"generated_{label}_step{step}.pdb")
        atom37_np_all = atom37.cpu().numpy()
        write_prot_to_pdb(atom37_np_all[0], pdb_path, overwrite=True, no_indexing=True)

        # Compute RMSD vs ground truth, plus chirality metrics, aggregated
        # over the nsamples generated samples.
        rmsd = None
        rmsd_reflected = None
        chirality = None
        gt_ca = self._get_gt_ca_coords()
        if gt_ca is not None:
            rmsds = []
            rmsds_refl = []
            chiralities = []
            for i in range(atom37_np_all.shape[0]):
                gen_ca = atom37_np_all[i, :, _CA_INDEX, :]
                n_common = min(len(gt_ca), len(gen_ca))
                if n_common <= 0:
                    continue
                gen_c = gen_ca[:n_common]
                gt_c = gt_ca[:n_common]
                rmsds.append(_ca_rmsd(gen_c, gt_c))
                rmsds_refl.append(_ca_rmsd_with_reflection(gen_c, gt_c))
                chiralities.append(_chirality_sign(gen_c, gt_c))
            if rmsds:
                rmsd = float(np.mean(rmsds))
                rmsd_reflected = float(np.mean(rmsds_refl))
                # Mean over +-1 flags -> real-valued estimator in [-1, +1].
                chirality = float(np.mean(chiralities))

        return pdb_path, rmsd, rmsd_reflected, chirality

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
            logger.info(f"ProteinEvalCallback: generating proteins at step {step}")

            # --- Switch to eval mode for clean generation --------------------
            was_training = pl_module.training
            pl_module.eval()

            try:
                nsteps_base = getattr(pl_module, 'meanflow_nsteps_sample', 1)
                ema_ctx = self._get_ema_context(trainer)

                with ema_ctx:
                    # 1. MeanFlow 1-step (or nsteps_sample)
                    path_mf1, rmsd_mf1, rmsd_refl_mf1, chir_mf1 = self._generate_and_save(
                        pl_module, step, "mf1", nsteps=nsteps_base
                    )
                    # 2. MeanFlow 10x steps
                    path_mf10x, rmsd_mf10x, rmsd_refl_mf10x, chir_mf10x = self._generate_and_save(
                        pl_module, step, "mf10x", nsteps=nsteps_base * 10
                    )
                    # 3. FM-style ODE with 100 Euler steps (h=0)
                    path_fm100, rmsd_fm100, rmsd_refl_fm100, chir_fm100 = self._generate_and_save(
                        pl_module, step, "fm100", nsteps=100, use_fm_euler=True
                    )
                    # 4. FM-loss reflection symmetry diagnostic on GT x_1
                    loss_orig, loss_mirror, loss_diff = self._fm_loss_mirror_diagnostic(
                        pl_module
                    )
            finally:
                # --- Restore training mode -----------------------------------
                pl_module.train(was_training)

            # --- Log to WandB -----------------------------------------------
            log_dict = {
                "eval/generated_mf1": wandb.Molecule(path_mf1),
                "eval/generated_mf10x": wandb.Molecule(path_mf10x),
                "eval/generated_fm100": wandb.Molecule(path_fm100),
            }

            # RMSD (proper rotation only), RMSD with reflection allowed,
            # and chirality flag (+1 correct, -1 mirror image of GT).
            for label, rmsd, rmsd_refl, chir in [
                ("mf1", rmsd_mf1, rmsd_refl_mf1, chir_mf1),
                ("mf10x", rmsd_mf10x, rmsd_refl_mf10x, chir_mf10x),
                ("fm100", rmsd_fm100, rmsd_refl_fm100, chir_fm100),
            ]:
                if rmsd is not None:
                    log_dict[f"eval/rmsd_{label}"] = rmsd
                if rmsd_refl is not None:
                    log_dict[f"eval/rmsd_reflected_{label}"] = rmsd_refl
                if chir is not None:
                    log_dict[f"eval/chirality_{label}"] = chir

            # FM-loss reflection symmetry diagnostic. If losses match, the
            # loss landscape is chirality-degenerate and generation can land
            # on either chirality.
            if loss_orig is not None:
                log_dict["eval/fm_loss_gt"] = loss_orig
                log_dict["eval/fm_loss_gt_mirror"] = loss_mirror
                log_dict["eval/fm_loss_mirror_abs_diff"] = loss_diff

            if not self._gt_logged:
                gt_path = self._prepare_gt_pdb()
                if gt_path:
                    log_dict["eval/ground_truth_protein"] = wandb.Molecule(gt_path)
                self._gt_logged = True

            # Summary: "R" = proper-rotation RMSD, "R*" = reflection allowed,
            # "c" = mean chirality over nsamples generations (float in [-1, +1];
            # +1 = all correct, -1 = all mirrored).
            def _fmt(label, rmsd, rmsd_refl, chir):
                if rmsd is None:
                    return None
                chir_str = f"{chir:+.2f}" if chir is not None else "NA"
                return f"{label}: R={rmsd:.2f} R*={rmsd_refl:.2f} c={chir_str}"

            parts = [f"eval @ step {step}"]
            for line in [
                _fmt("mf1", rmsd_mf1, rmsd_refl_mf1, chir_mf1),
                _fmt("mf10x", rmsd_mf10x, rmsd_refl_mf10x, chir_mf10x),
                _fmt("fm100", rmsd_fm100, rmsd_refl_fm100, chir_fm100),
            ]:
                if line is not None:
                    parts.append(line)
            if loss_orig is not None:
                parts.append(
                    f"FM loss(x1)={loss_orig:.4f} FM loss(Q·x1)={loss_mirror:.4f} "
                    f"|Δ|={loss_diff:.4f}"
                )
            logger.info(" | ".join(parts))

            log_dict["trainer/global_step"] = step
            pl_module.logger.experiment.log(log_dict, commit=False)

        except Exception as e:
            logger.warning(f"ProteinEvalCallback failed at step {step}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return
