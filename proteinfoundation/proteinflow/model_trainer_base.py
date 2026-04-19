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
import random
import re

from abc import abstractmethod
from functools import partial
from typing import Dict, List, Literal

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float
from loguru import logger
from torch import Tensor

from proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level
from proteinfoundation.proteinflow.chirality_loss import chirality_hinge_loss


class ModelTrainerBase(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None):
        super(ModelTrainerBase, self).__init__()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only used for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        self.metric_factory = None

        # Scaling laws stuff
        self.nflops = 0
        self.nparams = None
        self.nsamples_processed = 0

        # Attributes re-written by classes that inherit from this one
        self.nn = None
        self.fm = None
        self.ot_sampler = None  # Overridden by subclasses if OT coupling enabled

        # For autoguidance, overridden in `self.configure_inference`
        self.nn_ag = None
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr
        )

        warmup_steps = self.cfg_exp.opt.get("warmup_steps", 0)
        total_steps = self.cfg_exp.opt.get("lr_decay_total_steps", 0)
        end_lr = self.cfg_exp.opt.get("end_lr", 0.0)
        init_lr = self.cfg_exp.opt.get("init_lr", 0.0)
        base_lr = self.cfg_exp.opt.lr

        if total_steps > 0:
            min_factor = end_lr / base_lr if base_lr > 0 else 0.0
            init_factor = init_lr / base_lr if base_lr > 0 else 0.0

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return init_factor + (1.0 - init_factor) * current_step / max(1, warmup_steps)
                linear = (total_steps - current_step) / max(1, total_steps - warmup_steps)
                return max(min_factor, linear)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        return optimizer

    def _nn_out_to_x_clean(self, nn_out, batch):
        """[LEGACY - not used in MeanFlow training/inference]
        Transforms the output of the nn to a clean sample prediction. The transformation depends on the
        parameterization used. For now we admit x_1 or v.

        Args:
            nn_out: Dictionary, nerual network output
                - "coords_pred": Tensor of shape [b, n, 3], could be the clean sample or the velocity
                - "pair_pred" (Optional): Tensor of shape [b, n, n, num_buckets_predict_pair], could be the clean sample or the velocity
            batch: Dictionary, batch of data

        Returns:
            Clean sample prediction, tensor of shape [b, n, 3].
        """
        nn_pred = nn_out["coors_pred"]
        t = batch["t"]  # [*]
        t_ext = t[..., None, None]  # [*, 1, 1]
        x_t = batch["x_t"]  # [*, n, 3]
        if self.cfg_exp.model.target_pred == "x_1":
            x_1_pred = nn_pred
        elif self.cfg_exp.model.target_pred == "v":
            x_1_pred = x_t + (1.0 - t_ext) * nn_pred
        else:
            raise IOError(
                f"Wrong parameterization chosen: {self.cfg_exp.model.target_pred}"
            )
        return x_1_pred

    def predict_clean(
        self,
        batch: Dict,
    ):
        """[LEGACY - not used in MeanFlow training/inference]
        Predicts clean samples given noisy ones and time.

        Args:
            batch: a batch of data with some additions, including
                - "x_t": Type depends on the mode (see beluw, "returns" part)
                - "t": Time, shape [*]
                - "mask": Binary mask of shape [*, n]
                - "x_sc" (optional): Prediction for self-conditioning
                - Other features from the dataloader.

        Returns:
            Predicted clean sample, depends on the "modality" we're in.
                - For frameflow it returns a dictionary with keys "trans" and "rot", and values
                tensors of shape [*, n, 3] and [*, n, 3, 3] respectively,
                - For CAflow it returns a tensor of shape [*, n, 3].
            Other things predicted by nn (pair_pred for distogram loss)
        """
        nn_out = self.nn(batch)  # [*, n, 3]
        return self._nn_out_to_x_clean(nn_out, batch), nn_out  # [*, n, 3]

    def predict_clean_n_v_w_guidance(
        self,
        batch: Dict,
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
    ):
        """[LEGACY - not used in MeanFlow training/inference]
        Logic for CFG and autoguidance goes here. This computes a clean sample prediction (can be single thing, tuple, etc)
        and the corresponding vector field used to initialize.

        Here if we want to do the different self conditioning for cond / ucond, ag / no ag, we can just return tuples of x_pred and
        modify the batches accordingly every time we call predict clean.

        w: guidance weight
        alpha: autoguidance ratio
        x_pred = w * x_pred + (1 - alpha) * (1 - w) * x_pred_uncond + alpha * (1 - w) * x_pred_auto_guidance

        WARNING: The ag checkpoint needs to rely on the same parameterization of the main model. This can be changed after training
        so no big deal but just in case leaving a note.
        """
        if self.motif_conditioning and ("fixed_structure_mask" not in batch or "x_motif" not in batch):
            batch.update(self.motif_factory(batch, zeroes = True))  # for generation we have to pass conditioning info in. But for validation do the same as training

        nn_out = self.nn(batch)
        x_pred = self._nn_out_to_x_clean(nn_out, batch)

        if guidance_weight != 1.0:
            assert autoguidance_ratio >= 0.0 and autoguidance_ratio <= 1.0
            if autoguidance_ratio > 0.0:  # Use auto-guidance
                nn_out_ag = self.nn_ag(batch)
                x_pred_ag = self._nn_out_to_x_clean(nn_out_ag, batch)
            else:
                x_pred_ag = torch.zeros_like(x_pred)

            if autoguidance_ratio < 1.0:  # Use CFG
                assert (
                    "cath_code" in batch
                ), "Only support CFG when cath_code is provided"
                uncond_batch = batch.copy()
                uncond_batch.pop("cath_code")
                nn_out_uncond = self.nn(uncond_batch)
                x_pred_uncond = self._nn_out_to_x_clean(nn_out_uncond, uncond_batch)
            else:
                x_pred_uncond = torch.zeros_like(x_pred)

            x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (
                autoguidance_ratio * x_pred_ag
                + (1 - autoguidance_ratio) * x_pred_uncond
            )

        v = self.fm.xt_dot(x_pred, batch["x_t"], batch["t"], batch["mask"])
        return x_pred, v

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    @abstractmethod
    def align_wrapper(self, x_0, x_1, mask):
        """Performs Kabsch on the CAs of x_0 and x_1."""

    @abstractmethod
    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """

    def sample_t(self, shape):
        if self.cfg_exp.loss.t_distribution.name == "uniform":
            t_max = self.cfg_exp.loss.t_distribution.p2
            return torch.rand(shape, device=self.device) * t_max  # [*]
        elif self.cfg_exp.loss.t_distribution.name == "logit-normal":
            mean = self.cfg_exp.loss.t_distribution.p1
            std = self.cfg_exp.loss.t_distribution.p2
            noise = torch.randn(shape, device=self.device) * std + mean  # [*]
            return torch.nn.functional.sigmoid(noise)  # [*]
        elif self.cfg_exp.loss.t_distribution.name == "beta":
            p1 = self.cfg_exp.loss.t_distribution.p1
            p2 = self.cfg_exp.loss.t_distribution.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            return dist.sample(shape).to(self.device)
        elif self.cfg_exp.loss.t_distribution.name == "mix_up02_beta":
            p1 = self.cfg_exp.loss.t_distribution.p1
            p2 = self.cfg_exp.loss.t_distribution.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            samples_beta = dist.sample(shape).to(self.device)
            samples_uniform = torch.rand(shape, device=self.device)
            u = torch.rand(shape, device=self.device)
            return torch.where(u < 0.02, samples_uniform, samples_beta)
        else:
            raise NotImplementedError(
                f"Sampling mode for t {self.cfg_exp.loss.t_distribution.name} not implemented"
            )

    # Adaptive weighting (MeanFlow paper Eq. 22)
    def adaptive_loss(self, loss):
        norm_p = self.cfg_exp.training.meanflow.get("norm_p", None)
        norm_eps = self.cfg_exp.training.meanflow.get("norm_eps", None)
        if norm_p is None or norm_eps is None:
            return loss
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        return (loss / adp_wt)

    def _compute_adp_wt(self, loss):
        """Return the adaptive-weight tensor ``(loss.detach() + eps)^p`` for
        the SAME (norm_p, norm_eps) used by ``adaptive_loss``.

        Returns a tensor of ones with ``loss``'s shape/device when the knobs
        are absent (so downstream multipliers are no-ops). Sharing this
        computation between MF and auxiliary losses (e.g. chirality) keeps
        their relative scale invariant under ``norm_p``.
        """
        norm_p = self.cfg_exp.training.meanflow.get("norm_p", None)
        norm_eps = self.cfg_exp.training.meanflow.get("norm_eps", None)
        if norm_p is None or norm_eps is None:
            return torch.ones_like(loss)
        return (loss.detach() + norm_eps) ** norm_p

    def _compute_single_noise_loss(self, x_1, mask, t_ext, r_ext, t, batch, B, *, use_sc=False):
        """
        Runs one noise sample through the MeanFlow + FM loss pipeline.

        Samples x_0, applies OT coupling if enabled, interpolates, computes
        JVP (MeanFlow loss) and FM loss, and applies adaptive weighting.

        Args:
            x_1: [B, n, 3] clean data, already COM-zeroed.
            mask: [B, n] bool residue mask.
            t_ext: [B, 1, 1] noise-side time (extended).
            r_ext: [B, 1, 1] data-side time (extended).
            t: [B] noise-side time (flat, needed for FM loss).
            batch: original batch dict (needed for cath_code).
            B: int batch size.
            use_sc: If True, run one extra no-grad forward pass to obtain
                x_sc and feed it as a detached constant into the JVP and FM
                sub-passes. Requires the NN config to declare x_sc /
                x_sc_pair_dists in its feature lists.

        Returns:
            combined_adp_loss: scalar tensor (requires_grad=True)
            raw_loss_mf: scalar tensor (detached mean, for logging)
            raw_loss_fm: scalar tensor (detached mean, for logging)
            raw_loss_chir: scalar tensor (detached, for logging; zero when disabled)
            raw_adp_wt_mean: scalar tensor (detached mean of per-sample adp_wt,
                for logging the shared MF/chirality scaling knob)
        """
        n = x_1.shape[-2]
        dtype = x_1.dtype
        batch_shape = x_1.shape[:-2]  # (B,)

        # 1. Sample noise
        x_0 = self.fm.sample_reference(
            n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
        )
        # 2. Standard square-batch OT (when noise_samples pool is NOT active)
        if self.ot_sampler is not None:
            ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(x_1, x_0, mask)
            x_0 = x_0[ot_noise_idx]

        # 3. Interpolate: z_t = (1-t)*x_1 + t*x_0 (paper convention)
        z = (1 - t_ext) * x_1 + t_ext * x_0
        v = x_0 - x_1
        z = self.fm._apply_mask(z, mask)
        v = self.fm._apply_mask(v, mask)

        # 3b. Self-conditioning warmup (no-grad, no tangent).
        x_sc = None
        if use_sc:
            with torch.no_grad():
                warmup_batch = {
                    "x_t": z,
                    "t": t_ext.squeeze(-1).squeeze(-1),
                    "h": (t_ext - r_ext).squeeze(-1).squeeze(-1),
                    "mask": mask,
                }
                if "cath_code" in batch:
                    warmup_batch["cath_code"] = [batch["cath_code"][i] for i in range(B)]
                u_warmup = self.nn(warmup_batch)["coors_pred"]
                # Interpret u as avg velocity and recover a clean-coord proxy:
                # x_1 ≈ z - t * u (exact at r=0, approximation elsewhere).
                x_sc = z - t_ext * u_warmup
                x_sc = self.fm._mask_and_zero_com(x_sc, mask)
                x_sc = x_sc.detach()

        # 4. JVP for MeanFlow loss
        def u_func(z_in, t_in, r_in):
            h = (t_in - r_in).squeeze(-1).squeeze(-1)
            t_flat = t_in.squeeze(-1).squeeze(-1)
            batch_nn = {
                "x_t": z_in,
                "t": t_flat,
                "h": h,
                "mask": mask,
            }
            if x_sc is not None:
                batch_nn["x_sc"] = x_sc  # detached, closed-over, no tangent
            if "cath_code" in batch:
                batch_nn["cath_code"] = [batch["cath_code"][i] for i in range(B)]
            nn_out = self.nn(batch_nn)
            return nn_out["coors_pred"]

        dtdt_mf = torch.ones_like(t_ext)
        drdt_mf = torch.zeros_like(r_ext)

        with torch.amp.autocast(self.device.type, enabled=False):
            u_pred_mf, dudt_mf = torch.func.jvp(
                u_func, (z, t_ext, r_ext), (v, dtdt_mf, drdt_mf)
            )

        u_tgt = (v - (t_ext - r_ext) * dudt_mf).detach()

        u_pred = self.fm._mask_and_zero_com(u_pred_mf, mask)
        u_tgt = self.fm._mask_and_zero_com(u_tgt, mask)

        nres = mask.sum(dim=-1) * 3
        error = (u_pred - u_tgt) * mask[..., None]
        loss_mf = (error ** 2).sum(dim=(-1, -2)) / nres
        raw_loss_mf = loss_mf.mean().detach() # use for logging, before adaptive weighting
        # Compute adp_wt ONCE from the per-sample MF loss and reuse it for
        # both the MF-side division (via adaptive_loss) and the chirality
        # scaling below. At norm_p=0, adp_wt is identically 1 → backward
        # compatible with the baseline.
        adp_wt_mf = self._compute_adp_wt(loss_mf)
        raw_adp_wt_mean = adp_wt_mf.mean().detach()
        loss_mf = self.adaptive_loss(loss_mf).mean()

        # 5. FM loss
        mf_ratio = self.cfg_exp.training.meanflow.ratio
        v_pred = u_func(z, t, t)
        v_pred = self.fm._mask_and_zero_com(v_pred, mask)
        loss_fm = (v_pred - v) ** 2 * mask[..., None]
        loss_fm = loss_fm.sum(dim=(-1, -2)) / nres
        raw_loss_fm = loss_fm.mean().detach() # use for logging, before adaptive weighting
        loss_fm = self.adaptive_loss(loss_fm).mean()

        # 6. Combined
        combined_adp_loss = (1 - mf_ratio) * loss_fm + mf_ratio * loss_mf

        # 7. Chirality hinge loss (optional)
        raw_loss_chir = torch.zeros((), device=self.device)
        if self.chirality_loss_enabled and self.chirality_loss_weight > 0:
            # Recover x_1 prediction from the FM sub-pass velocity:
            # z = (1-t)*x_1 + t*x_0  =>  x_1 ≈ z - t * v  (at r=t).
            x_1_pred = z - t_ext * v_pred
            x_1_pred = self.fm._mask_and_zero_com(x_1_pred, mask)
            loss_chir = chirality_hinge_loss(
                x_pred=x_1_pred,
                x_gt=x_1,
                mask=mask,
                margin_alpha=self.chirality_margin_alpha,
                stride=self.chirality_stride,
            )
            raw_loss_chir = loss_chir.detach()
            # Apply the SAME per-step adp_wt used on the MF loss so MF and
            # chirality gradients stay at the same scale under norm_p > 0.
            # chirality_hinge_loss returns a scalar; adp_wt_mf is per-sample
            # ([B]), so we divide by its batch-mean (detached, already
            # detached inside _compute_adp_wt). At norm_p=0 this mean == 1,
            # so the branch is numerically identical to the baseline.
            adp_wt_chir = adp_wt_mf.mean()
            loss_chir_scaled = loss_chir / adp_wt_chir
            combined_adp_loss = combined_adp_loss + self.chirality_loss_weight * loss_chir_scaled

        return combined_adp_loss, raw_loss_mf, raw_loss_fm, raw_loss_chir, raw_adp_wt_mean

    def training_step(self, batch, batch_idx, *, val_step=False):
        """
        MeanFlow training step. Learns average velocity u(z_t, r, t) using JVP.

        PAPER CONVENTION: z_t = (1-t)*data + t*noise. Data at t=0, noise at t=1.
        Proteina convention: x_0=noise, x_1=data. We map between them explicitly.

        Args:
            batch: Data batch.
            val_step: If True, logs under "validation_loss" prefix and skips
                scaling stats. Validation always uses K=1 (single noise sample).

        Returns:
            Training loss (scalar) for K=1/automatic optimization, or None for K>1/manual.
        """
        log_prefix = "validation_loss" if val_step else "train"

        assert not self.motif_conditioning, (
            "Motif conditioning is not yet supported with MeanFlow training. "
            "Set training.motif_conditioning=False in your config."
        )

        # --- 1. Shared setup: extract x_1, mask, sample (t, r), fold conditioning ---
        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        noise_samples = ot_cfg.get("noise_samples", None)

        if noise_samples is not None and self.ot_sampler is not None:
            # OT pool: _build_ot_pool determines which proteins to train on.
            # x_0 from the pool is discarded; each noise pass in the K loop
            # samples fresh x_0 and runs standard OT against the fixed x_1.
            x_1, _x_0_pool, mask, batch_shape, n, dtype = self._build_ot_pool(batch)
        else:
            x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, mask)

        # Sample (t, r) -- shared across all K noise passes
        t, r = self.fm.sample_two_timesteps(
            batch_shape, self.device,
            ratio=self.meanflow_ratio,
            P_mean_t=self.meanflow_P_mean_t,
            P_std_t=self.meanflow_P_std_t,
            P_mean_r=self.meanflow_P_mean_r,
            P_std_r=self.meanflow_P_std_r,
        )
        t_ext = t[..., None, None]   # [B, 1, 1]
        r_ext = r[..., None, None]   # [B, 1, 1]

        B = t.shape[0]

        # --- 1b. Self-conditioning activation (per step, consistent across K passes) ---
        use_sc = (
            not val_step
            and self.cfg_exp.training.get("self_cond", False)
            and random.random() < self.self_cond_prob
        )

        # Fold conditioning -- shared, mutates batch in-place
        if self.cfg_exp.training.fold_cond:
            bs = x_1.shape[0]
            cath_code_list = batch.cath_code
            for i in range(bs):
                cath_code_list[i] = mask_cath_code_by_level(
                    cath_code_list[i], level="H"
                )
                if random.random() < self.cfg_exp.training.mask_T_prob:
                    cath_code_list[i] = mask_cath_code_by_level(
                        cath_code_list[i], level="T"
                    )
                    if random.random() < self.cfg_exp.training.mask_A_prob:
                        cath_code_list[i] = mask_cath_code_by_level(
                            cath_code_list[i], level="A"
                        )
                        if random.random() < self.cfg_exp.training.mask_C_prob:
                            cath_code_list[i] = mask_cath_code_by_level(
                                cath_code_list[i], level="C"
                            )
            batch.cath_code = cath_code_list
        else:
            if "cath_code" in batch:
                batch.pop("cath_code")

        # --- 2. Determine K ---
        # Validation always uses a single noise sample regardless of config.
        K = 1 if val_step else self.loss_accumulation_steps

        # --- 3. K=1 path (automatic optimization, identical to pre-refactor behavior) ---
        if K == 1:
            combined_adp_loss, raw_loss_mf, raw_loss_fm, raw_loss_chir, raw_adp_wt_mean = self._compute_single_noise_loss(
                x_1, mask, t_ext, r_ext, t, batch, B, use_sc=use_sc,
            )

            self.log(
                f"{log_prefix}/combined_adaptive_loss",
                combined_adp_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            if not val_step:
                self.log(
                    f"{log_prefix}/raw_loss_mf",
                    raw_loss_mf,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{log_prefix}/raw_loss_fm",
                    raw_loss_fm,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{log_prefix}/raw_loss_chirality",
                    raw_loss_chir,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{log_prefix}/raw_adp_wt_mean",
                    raw_adp_wt_mean,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                b, n = mask.shape
                self.nsamples_processed = (
                    self.nsamples_processed + b * self.trainer.world_size
                )
                self.log(
                    "scaling/nsamples_processed",
                    self.nsamples_processed * 1.0,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=1,
                    sync_dist=True,
                )
                self.log(
                    "scaling/nparams",
                    self.nparams * 1.0,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=1,
                    sync_dist=True,
                )

            return combined_adp_loss

        # --- 4. K>1 path (manual optimization) ---
        optimizer = self.optimizers()

        total_loss_mf = 0.0
        total_loss_fm = 0.0
        total_loss_chir = 0.0
        total_adp_wt = 0.0

        for _ in range(K):
            loss_k, raw_loss_mf_k, raw_loss_fm_k, raw_loss_chir_k, raw_adp_wt_mean_k = self._compute_single_noise_loss(
                x_1, mask, t_ext, r_ext, t, batch, B, use_sc=use_sc,
            )
            # Backward with 1/K scaling so accumulated gradient == mean gradient
            self.manual_backward(loss_k / K)
            total_loss_mf += raw_loss_mf_k.item()
            total_loss_fm += raw_loss_fm_k.item()
            total_loss_chir += raw_loss_chir_k.item()
            total_adp_wt += raw_adp_wt_mean_k.item()

        avg_loss_mf = total_loss_mf / K
        avg_loss_fm = total_loss_fm / K
        avg_loss_chir = total_loss_chir / K
        avg_adp_wt = total_adp_wt / K
        mf_ratio = self.cfg_exp.training.meanflow.ratio
        avg_combined = (1 - mf_ratio) * avg_loss_fm + mf_ratio * avg_loss_mf

        # Log averaged metrics
        self.log(
            f"{log_prefix}/combined_adaptive_loss", avg_combined,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_loss_mf", avg_loss_mf,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_loss_fm", avg_loss_fm,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_loss_chirality", avg_loss_chir,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/raw_adp_wt_mean", avg_adp_wt,
            on_step=True, on_epoch=True, prog_bar=False, logger=True,
            batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/loss_accumulation_steps", float(K),
            on_step=True, on_epoch=False, prog_bar=False, logger=True,
            batch_size=1, sync_dist=False, add_dataloader_idx=False,
        )

        b, _n = mask.shape
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed", self.nsamples_processed * 1.0,
            on_step=True, on_epoch=False, prog_bar=False, logger=True,
            batch_size=1, sync_dist=True,
        )
        self.log(
            "scaling/nparams", self.nparams * 1.0,
            on_step=True, on_epoch=False, prog_bar=False, logger=True,
            batch_size=1, sync_dist=True,
        )

        # --- 5. Manual optimizer step (respects _accum_grad_batches) ---
        self._manual_step_count += 1
        if self._manual_step_count % self._accum_grad_batches == 0:
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()
            optimizer.zero_grad()
            # Step LR scheduler if present
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        return None  # manual optimization: Lightning doesn't need a loss value

    def compute_fm_loss(
        self, x_1, x_1_pred, x_t, t: Float[Tensor, "*"], mask: Bool[Tensor, "* nres"], **kwargs
    ):
        """
        Computes and logs flow matching loss(es).

        Args:
            x_1: True clean sample.
            x_1_pred: Predicted clean sample.
            x_t: Sample at interpolation time t (used as input to predict clean sample).
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss per sample in the batch.
        """
        raise NotImplementedError("Subclass must override compute_fm_loss if used")

    def compute_auxiliary_loss(
        self, x_1, x_1_pred, x_t, t: Float[Tensor, "*"], mask: Bool[Tensor, "* nres"], batch = None, **kwargs
    ):
        """
        Computes and logs auxiliary losses.

        Args:
            x_1: True clean sample.
            x_1_pred: Predicted clean sample.
            x_t: Sample at interpolation time t (used as input to predict clean sample).
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Auxiliary loss per sample in the batch.
        """
        raise NotImplementedError("Subclass must override compute_auxiliary_loss if used")

    def detach_gradients(self, x):
        """Detaches gradients from sample x"""
        raise NotImplementedError("Subclass must override detach_gradients if used")

    def validation_step(self, batch, batch_idx):
        """
        This is the validation step for both when generating proteins (dataloader_idx_1) and when evaluating the training
        loss on some validation data (dataloader_idx_2).

        dataloader_idx_1: The batch comes from the length dataset
        dataloader_idx_2: The batch contains actual data

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
            dataloader_idx: 0 or 1.
                0 means the batch comes from the length dataloader, contains no data, but the info of the samples to generate (nsamples, nres, dt)
                1 means the batch comes from the data dataloader, contains data from the dataset, we compute normal training loss
        """
        self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch, batch_idx):
        """
        Evaluates the training loss, without auxiliary loss nor logging.
        """
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, val_step=True)
            self.validation_output_data.append(loss.item())

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def predict_step(self, batch, batch_idx):
        """
        MeanFlow prediction step. Uses average velocity field for generation.

        Args:
            batch: data batch with nsamples, nres, and optionally mask.

        Returns:
            Samples generated in atom 37 format.
        """
        # Inference config overrides training defaults when available
        if self.inf_cfg is not None:
            nsteps = self.inf_cfg.get("nsteps", self.meanflow_nsteps_sample)
            fold_cond = self.inf_cfg.get("fold_cond", self.cfg_exp.training.get("fold_cond", False))
        else:
            nsteps = self.meanflow_nsteps_sample
            fold_cond = self.cfg_exp.training.get("fold_cond", False)

        mask = batch['mask'].squeeze(0) if 'mask' in batch else None
        cath_code = _extract_cath_code(batch) if fold_cond else None

        x = self.generate(
            nsamples=batch["nsamples"],
            n=batch["nres"],
            nsteps=nsteps,
            dtype=torch.float32,
            mask=mask,
            cath_code=cath_code,
        )
        return self.samples_to_atom37(x)  # [b, n, 37, 3]

    def generate(
        self,
        nsamples: int,
        n: int,
        nsteps: int = 1,
        dtype: torch.dtype = None,
        mask=None,
        cath_code=None,
        **kwargs,
    ) -> Tensor:
        """
        Generates samples using MeanFlow average velocity field.
        1-step: z_0 = z_1 - u(z_1, r=0, t=1)
        Multi-step: iterate over intervals from t=1 to t=0.
        """
        if mask is None:
            mask = torch.ones(nsamples, n).long().bool().to(self.device)

        def predict_u(z, t, r, mask):
            # PAPER CONVENTION: t and r are scalars broadcast to batch
            batch_nn = {
                "x_t": z,
                "t": t,
                "h": t - r,
                "mask": mask,
            }
            if cath_code is not None:
                batch_nn["cath_code"] = cath_code
            nn_out = self.nn(batch_nn)
            return nn_out["coors_pred"]

        return self.fm.meanflow_sample(
            predict_u, nsamples, n, self.device, mask, nsteps, dtype
        )

    def generate_fm_euler(
        self,
        nsamples: int,
        n: int,
        nsteps: int = 100,
        dtype: torch.dtype = None,
        mask=None,
        cath_code=None,
    ) -> Tensor:
        """
        Generates samples using standard FM-style Euler ODE integration.

        Calls the network with h=0 (instantaneous velocity mode, i.e. r=t)
        and integrates from t=1 (noise) to t=0 (data) using Euler steps.
        This serves as a diagnostic to compare MeanFlow 1-step generation
        against the standard flow matching ODE solver.

        CONVENTION: z_t = (1-t)*data + t*noise. v(z_t,t) = dz/dt = noise - data.
        Integration: z_{t-dt} = z_t - dt * v(z_t, t).
        """
        if mask is None:
            mask = torch.ones(nsamples, n).long().bool().to(self.device)

        def predict_v_instantaneous(z, t, _r, mask):
            # Force h=0 so the network predicts instantaneous velocity v(z_t, t)
            batch_nn = {
                "x_t": z,
                "t": t,
                "h": torch.zeros_like(t),
                "mask": mask,
            }
            if cath_code is not None:
                batch_nn["cath_code"] = cath_code
            nn_out = self.nn(batch_nn)
            return nn_out["coors_pred"]

        # Reuse meanflow_sample infrastructure: with h=0, multi-step Euler
        # becomes standard FM ODE integration from t=1 to t=0.
        return self.fm.meanflow_sample(
            predict_v_instantaneous, nsamples, n, self.device, mask, nsteps, dtype
        )


def _extract_cath_code(batch):
    cath_code = batch.get("cath_code", None)
    if cath_code:
        # Remove the additional tuple layer introduced during collate
        _cath_code = []
        for codes in cath_code:
            _cath_code.append(
                [code[0] if isinstance(code, tuple) else code for code in codes]
            )
        cath_code = _cath_code
    return cath_code
