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
import warnings
from math import prod
from typing import Dict

import torch
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from scipy.spatial.transform import Rotation
from torch import Tensor

from proteinfoundation.flow_matching.ot_sampler import MaskedOTPlanSampler
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
from proteinfoundation.utils.align_utils.align_utils import kabsch_align
from proteinfoundation.utils.coors_utils import ang_to_nm, trans_nm_to_atom37
from proteinfoundation.nn.motif_factory import SingleMotifFactory


@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> Float[Tensor, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class Proteina(ModelTrainerBase):
    def __init__(self, cfg_exp, store_dir=None):
        super(Proteina, self).__init__(cfg_exp=cfg_exp, store_dir=store_dir)
        self.save_hyperparameters()

        # Define flow matcher
       
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)
        self.fm = R3NFlowMatcher(zero_com= not self.motif_conditioning, scale_ref=1.0)  # Work in nm

        # Optimal transport coupling
        ot_cfg = cfg_exp.training.get("ot_coupling", {})
        if ot_cfg.get("enabled", False):
            self.ot_sampler = MaskedOTPlanSampler(
                method=ot_cfg.get("method", "exact"),
                reg=ot_cfg.get("reg", 0.05),
                reg_m=ot_cfg.get("reg_m", 1.0),
                normalize_cost=ot_cfg.get("normalize_cost", False),
            )

        # Loss accumulation
        K = cfg_exp.training.get("loss_accumulation_steps", 1)
        self.loss_accumulation_steps = K
        if K > 1:
            # Manual optimization: Lightning won't call optimizer.step() for us.
            # We reimplement accumulate_grad_batches manually below.
            self.automatic_optimization = False
        self._accum_grad_batches = cfg_exp.opt.get("accumulate_grad_batches", 1)
        self._manual_step_count = 0

        if self.motif_conditioning:
            self.motif_conditioning_sequence_rep = cfg_exp.training.get("motif_conditioning_sequence_rep", False)
            if self.motif_conditioning_sequence_rep:
                if "motif_sequence_mask" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_sequence_mask")
                if "motif_x1" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_x1")
                
            if "motif_structure_mask" not in cfg_exp.model.nn.feats_pair_repr:
                cfg_exp.model.nn.feats_pair_repr.append("motif_structure_mask")
            if "motif_x1_pair_dists" not in cfg_exp.model.nn.feats_pair_repr:
                cfg_exp.model.nn.feats_pair_repr.append("motif_x1_pair_dists")
            self.motif_factory = SingleMotifFactory(motif_prob=cfg_exp.training.get("motif_prob", 1.0))

        # MeanFlow hyperparams
        mf_cfg = cfg_exp.training.get("meanflow", {})
        self.meanflow_norm_p = mf_cfg.get("norm_p", 1.0)
        self.meanflow_norm_eps = mf_cfg.get("norm_eps", 1e-3)
        self.meanflow_ratio = mf_cfg.get("ratio", 0.25)
        self.meanflow_P_mean_t = mf_cfg.get("P_mean_t", mf_cfg.get("P_mean", -0.4))
        self.meanflow_P_std_t = mf_cfg.get("P_std_t", mf_cfg.get("P_std", 1.0))
        self.meanflow_P_mean_r = mf_cfg.get("P_mean_r", mf_cfg.get("P_mean", -0.4))
        self.meanflow_P_std_r = mf_cfg.get("P_std_r", mf_cfg.get("P_std", 1.0))
        self.meanflow_nsteps_sample = mf_cfg.get("nsteps_sample", 1)

        if cfg_exp.loss.get("use_aux_loss", False):
            warnings.warn(
                "use_aux_loss=True has no effect in MeanFlow training.",
                UserWarning, stacklevel=2,
            )
        if cfg_exp.training.get("self_cond", False):
            warnings.warn(
                "self_cond=True has no effect in MeanFlow training.",
                UserWarning, stacklevel=2,
            )

        # Neural network
        # JVP incompatible with torch.utils.checkpoint (used in PairReprUpdate)
        assert not cfg_exp.model.nn.get(
            "update_pair_repr", False
        ), "update_pair_repr must be False for MeanFlow (torch.func.jvp incompatible with torch.utils.checkpoint)"
        self.nn = ProteinTransformerAF3(**cfg_exp.model.nn)

        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        create_dir(self.val_path_tmp)

    def on_train_start(self):
        """Store reference to training dataset for OT pool sampling."""
        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        noise_samples = ot_cfg.get("noise_samples", None)
        if noise_samples is not None and self.ot_sampler is not None:
            dm = self.trainer.datamodule
            if dm.train_ds is None:
                dm.setup("fit")
            self._ot_dataset = dm.train_ds
            assert noise_samples >= self.trainer.datamodule.batch_size, (
                f"noise_samples ({noise_samples}) must be >= batch_size "
                f"({self.trainer.datamodule.batch_size})"
            )

    def _build_ot_pool(self, batch):
        """Build a square OT pool on CPU and return B selected pairs on GPU.

        Samples extra proteins from the training dataset to build a pool of
        `noise_samples` data/noise pairs, computes square OT assignment via
        Hungarian algorithm, then randomly selects B pairs for training.

        Args:
            batch: Dataloader batch (used to extract the initial B proteins).

        Returns:
            Tuple (x_1, x_0, mask, batch_shape, n, dtype) with B selected
            pairs on self.device (GPU).
        """
        import scipy.optimize
        from proteinfoundation.utils.dense_padding_data_loader import (
            dense_padded_from_data_list,
        )

        ot_cfg = self.cfg_exp.training.get("ot_coupling", {})
        K = ot_cfg["noise_samples"]  # total pool size

        # --- 1. Extract clean data from the dataloader batch (on CPU) ---
        x_1_batch, mask_batch, batch_shape_orig, n_batch, dtype = (
            self.extract_clean_sample(batch)
        )
        x_1_batch = self.fm._mask_and_zero_com(x_1_batch, mask_batch)
        B = batch_shape_orig[0]

        # Ensure tensors are on CPU for OT computation
        x_1_batch = x_1_batch.detach().cpu()
        mask_batch = mask_batch.cpu()

        # --- 2. Sample extra proteins to fill the pool ---
        n_extra = K - B

        if n_extra > 0:
            extra_indices = torch.randint(0, len(self._ot_dataset), (n_extra,))
            extra_data_list = [self._ot_dataset[int(idx)] for idx in extra_indices]
            extra_batch = dense_padded_from_data_list(extra_data_list)

            # Process extra proteins identically to the main batch
            x_1_extra, mask_extra, _, _, _ = self.extract_clean_sample(extra_batch)
            x_1_extra = self.fm._mask_and_zero_com(x_1_extra, mask_extra)
            x_1_extra = x_1_extra.detach().cpu()
            mask_extra = mask_extra.cpu()

            # --- 3. Pad both to the same sequence length ---
            N_batch = x_1_batch.shape[1]
            N_extra = x_1_extra.shape[1]
            N_pool = max(N_batch, N_extra)

            if N_batch < N_pool:
                pad_size = N_pool - N_batch
                x_1_batch = torch.cat(
                    [x_1_batch, torch.zeros(B, pad_size, 3, dtype=dtype)], dim=1
                )
                mask_batch = torch.cat(
                    [mask_batch, torch.zeros(B, pad_size, dtype=torch.bool)], dim=1
                )

            if N_extra < N_pool:
                pad_size = N_pool - N_extra
                x_1_extra = torch.cat(
                    [x_1_extra, torch.zeros(n_extra, pad_size, 3, dtype=dtype)],
                    dim=1,
                )
                mask_extra = torch.cat(
                    [mask_extra, torch.zeros(n_extra, pad_size, dtype=torch.bool)],
                    dim=1,
                )

            x_1_pool = torch.cat([x_1_batch, x_1_extra], dim=0)  # [K, N_pool, 3]
            mask_pool = torch.cat([mask_batch, mask_extra], dim=0)  # [K, N_pool]
        else:
            # K == B: no extra samples needed
            x_1_pool = x_1_batch
            mask_pool = mask_batch
            N_pool = x_1_batch.shape[1]

        # --- 4. Sample noise on CPU ---
        x_0_pool = (
            torch.randn(K, N_pool, self.fm.dim, dtype=dtype) * self.fm.scale_ref
        )
        x_0_pool = self.fm._mask_and_zero_com(x_0_pool, mask_pool)

        # --- 5. Build cost matrix [K, K] ---
        mask_3d = mask_pool[..., None]  # [K, N_pool, 1] bool
        x_1_flat = (x_1_pool * mask_3d).reshape(K, -1)  # [K, N_pool*3]
        x_0_flat = (x_0_pool * mask_3d).reshape(K, -1)  # [K, N_pool*3]
        M = torch.cdist(x_1_flat, x_0_flat) ** 2  # [K, K]

        # --- 6. Hungarian algorithm for optimal assignment ---
        _, sigma = scipy.optimize.linear_sum_assignment(M.numpy())

        # --- 7. Randomly select B pairs from the OT plan ---
        sel = torch.randperm(K)[:B].numpy()
        x_1_sel = x_1_pool[sel]  # [B, N_pool, 3]
        x_0_sel = x_0_pool[sigma[sel]]  # [B, N_pool, 3]
        mask_sel = mask_pool[sel]  # [B, N_pool]

        # --- 8. Trim to max actual length of selected proteins ---
        max_len = int(mask_sel.sum(dim=1).max().item())
        if max_len == 0:
            max_len = N_pool  # safety fallback
        x_1_sel = x_1_sel[:, :max_len, :]
        x_0_sel = x_0_sel[:, :max_len, :]
        mask_sel = mask_sel[:, :max_len]

        # Re-apply mask and zero COM after trimming
        x_1_sel = self.fm._mask_and_zero_com(x_1_sel, mask_sel)
        x_0_sel = self.fm._mask_and_zero_com(x_0_sel, mask_sel)

        # --- 9. Move to GPU ---
        x_1_out = x_1_sel.to(device=self.device)
        x_0_out = x_0_sel.to(device=self.device)
        mask_out = mask_sel.to(device=self.device)

        # Clean up large CPU tensors
        del x_1_pool, x_0_pool, mask_pool, M, x_1_flat, x_0_flat

        return x_1_out, x_0_out, mask_out, torch.Size([B]), max_len, dtype

    def align_wrapper(self, x_0, x_1, mask):
        """Performs Kabsch on the translation component of x_0 and x_1."""
        return kabsch_align(mobile=x_0, target=x_1, mask=mask)

    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.
        Applies augmentations if those are required.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """
        x_1 = batch["coords"][:,:,1,:]  # [b, n, 3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        if self.cfg_exp.model.augmentation.global_rotation:
            # CAREFUL: If naug_rot is > 1 this increases "batch size"
            x_1, mask = self.apply_random_rotation(
                x_1, mask, naug=self.cfg_exp.model.augmentation.naug_rot
            )
        batch_shape = x_1.shape[:-2]
        n = x_1.shape[-2]
        return (
            ang_to_nm(x_1),
            mask,
            batch_shape,
            n,
            x_1.dtype,
        )  # Since we work in nm throughout

    def apply_random_rotation(self, x, mask, naug=1):
        """
        Applies random rotation augmentation. Each sample in the batch may receive more than one augmentation,
        specified by the parameters naug. If naug > 1 this is basically increaseing the batch size from b to
        naug * b. This should likely be implemented in the dataloaders.

        Args:
            - x: Data batch, shape [b, n, 3]
            - mask: Binary, shape [b, n]
            - naug: Number of augmentations to apply to each sample, effectively increasing batch size if >1.

        Returns:
            Augmented samples and mask, shapes [b * naug, n, 3] and [B * naug, n].
        """
        assert (
            x.ndim == 3
        ), f"Augmetations can only be used for simple (x_1) batches [b, n, 3], current shape is {x.shape}"
        assert (
            mask.ndim == 2
        ), f"Augmetations can only be used for simple (mask) batches [b, n], current shape is {mask.shape}"
        assert naug >= 1, f"Number of augmentations (int) should >= 1, currently {naug}"

        # Repeat for multiple augmentations per sample
        x = x.repeat([naug, 1, 1])  # [naug * b, n, 3]
        mask = mask.repeat([naug, 1])  # [naug * b, n]

        # Sample and apply rotations
        rots = sample_uniform_rotation(
            shape=x.shape[:-2], dtype=x.dtype, device=x.device
        )  # [naug * b, 3, 3]
        x_rot = torch.matmul(x, rots)
        return self.fm._mask_and_zero_com(x_rot, mask), mask

    def compute_loss_weight(
        self, t: Float[Tensor, "*"], eps: float = 1e-3
    ) -> Float[Tensor, "*"]:
        t = t.clamp(min=eps, max=1.0 - eps)  # For safety
        return t / (
            1.0 - t
        )

    def compute_fm_loss(
        self,
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* nres"],
        log_prefix: str,
    ) -> Float[Tensor, "*"]:
        """[LEGACY - not used in MeanFlow training/inference]
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        nres = torch.sum(mask, dim=-1) * 3  # [*]

        err = (x_1 - x_1_pred) * mask[..., None]  # [*, n, 3]
        loss = torch.sum(err**2, dim=(-1, -2)) / nres  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        if log_prefix:
            self.log(
                f"{log_prefix}/trans_loss",
                torch.mean(loss),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        return loss

    def compute_auxiliary_loss(
        self,
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: Dict[str, Tensor],
        log_prefix: str,
        batch: Dict[str, Tensor] = None,
    ) -> Float[Tensor, ""]:
        """[LEGACY - not used in MeanFlow training/inference]
        Computes and logs auxiliary losses.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, n].
            nn_out: Dictionary of output from neural network

        Returns:
            Auxiliary loss.
        """
        bs = x_1.shape[0]
        n = x_1.shape[1]
        nres = mask.sum(-1)  # [*]

        gt_ca_coors = x_1 * mask[..., None]  # [*, n, 3]
        pred_ca_coors = x_1_pred * mask[..., None]  # [*, n, 3]
        pair_mask = mask[..., None, :] * mask[..., None]  # [*, n, n]

        # Pairwise distances
        gt_pair_dists = torch.linalg.norm(
            gt_ca_coors[:, :, None, :] - gt_ca_coors[:, None, :, :], dim=-1
        )  # [*, n, n]
        pred_pair_dists = torch.linalg.norm(
            pred_ca_coors[:, :, None, :] - pred_ca_coors[:, None, :, :], dim=-1
        )  # [*, n, n]
        gt_pair_dists = gt_pair_dists * pair_mask  # [*, n, n]
        pred_pair_dists = pred_pair_dists * pair_mask  # [*, n, n]

        # Add mask to only account for pairs that are closer than thr in ground truth
        max_dist = self.cfg_exp.loss.thres_aux_2d_loss
        if max_dist is None:
            max_dist = 1e10
        pair_mask_thr = gt_pair_dists < max_dist  # [*, n, n]
        total_pair_mask = pair_mask * pair_mask_thr  # [*, n, n]

        # Compute loss
        den = torch.sum(total_pair_mask, dim=(-1, -2)) - nres
        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * total_pair_mask, dim=(-1, -2)
        )  # [*]
        dist_mat_loss = dist_mat_loss / den  # [*]

        # Distogram loss
        num_dist_buckets = self.cfg_exp.loss.get("num_dist_buckets", 64)
        pair_pred = nn_out.get("pair_pred", None)
        if num_dist_buckets and pair_pred is not None:
            assert (
                num_dist_buckets == pair_pred.shape[-1]
            ), "The number of distance buckets should be equal with the output dim of pair pred head"
            assert num_dist_buckets > 1, "Need more than one bucket for distogram loss"

            # Bucketize pair distance
            max_dist_boundary = self.cfg_exp.loss.get("max_dist_boundary", 1.0)
            boundaries = torch.linspace(
                0.0, max_dist_boundary, num_dist_buckets - 1, device=pair_pred.device
            )
            gt_pair_dist_bucket = torch.bucketize(
                gt_pair_dists, boundaries
            )  # [*, n, n], each value in [0, num_dist_buckets)

            # Distogram loss
            pair_pred = pair_pred.view(bs * n * n, num_dist_buckets)
            gt_pair_dist_bucket = gt_pair_dist_bucket.view(bs * n * n)
            distogram_loss = torch.nn.functional.cross_entropy(
                pair_pred, gt_pair_dist_bucket, reduction="none"
            )  # [bs * n * n]
            distogram_loss = distogram_loss.view(bs, n, n)
            distogram_loss = torch.sum(distogram_loss * pair_mask, dim=(-1, -2))  # [*]
            distogram_loss = distogram_loss / (
                pair_mask.sum(dim=(-1, -2)) + 1e-10
            )  # [*]
        else:
            distogram_loss = dist_mat_loss * 0

        auxiliary_loss = (
            distogram_loss
            * (t > self.cfg_exp.loss.aux_loss_t_lim)
            * self.cfg_exp.loss.aux_loss_weight
        )
        auxiliary_loss_no_w = distogram_loss * (t > self.cfg_exp.loss.aux_loss_t_lim)
        motif_aux_loss_weight = self.cfg_exp.loss.get("motif_aux_loss_weight", 0)
        scaffold_aux_loss_weight = self.cfg_exp.loss.get("scaffold_aux_loss_weight", 0)
        if scaffold_aux_loss_weight > 0:
            scaffold_loss = scaffold_aux_loss_weight * self.compute_fm_loss(
                        x_1=x_1,
                        x_1_pred=x_1_pred,
                        x_t=x_t,
                        mask=~batch["fixed_sequence_mask"]*batch["mask"],
                        t=t,
                        log_prefix=None
                    )
            auxiliary_loss += scaffold_loss
            self.log(
                f"{log_prefix}/scaffold_loss",
                torch.mean(scaffold_loss),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        elif motif_aux_loss_weight:
            mask_to_use = batch["fixed_sequence_mask"] * batch["mask"]
            check_weight = 1.0
            if not batch["fixed_sequence_mask"].any():
                check_weight = 0
                mask_to_use = batch["mask"]
            motif_loss = motif_aux_loss_weight * self.compute_fm_loss(
                x_1=x_1,
                x_1_pred=x_1_pred,
                x_t=x_t,
                mask=mask_to_use,
                t=t,
                log_prefix=None,
            )
            auxiliary_loss += check_weight * motif_loss
            self.log(
                f"{log_prefix}/motif_loss",
                torch.mean(motif_loss * check_weight),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )

        self.log(
            f"{log_prefix}/distogram_loss",
            torch.mean(distogram_loss),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/dist_mat_loss",
            torch.mean(dist_mat_loss),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/auxiliary_loss",
            torch.mean(auxiliary_loss),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/auxiliary_loss_no_w",
            torch.mean(auxiliary_loss_no_w),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return auxiliary_loss

    def detach_gradients(self, x):
        """Detaches gradients from sample x"""
        return x.detach()

    def samples_to_atom37(self, samples):
        """
        Transforms samples to atom37 representation.

        Args:
            samples: Tensor of shape [b, n, 3]

        Returns:
            Samples in atom37 representation, shape [b, n, 37, 3].
        """
        return trans_nm_to_atom37(samples)  # [b, n, 37, 3]
