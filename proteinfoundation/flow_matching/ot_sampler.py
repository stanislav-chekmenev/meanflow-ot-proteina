"""
Mini-batch optimal transport coupling for flow matching on protein structures.

OTPlanSampler is adapted from the torchcfm library by Tong et al.:
https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

Reference: Tong et al., "Improving and Generalizing Flow-Based Generative Models
with Minibatch Optimal Transport" (ICLR 2024).

MaskedOTPlanSampler is a thin wrapper that handles masked/variable-length protein
data by zeroing out padded positions before computing the OT cost matrix.
"""

import warnings
from functools import partial
from typing import Tuple, Union

import numpy as np
import ot as pot
import torch
from torch import Tensor


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan
    (wrt squared Euclidean cost) with different implementations of the plan
    calculation.

    Adapted from torchcfm (https://github.com/atong01/conditional-flow-matching).
    """

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(
                pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m
            )
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source
        and a target minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi $(x,z) \sim \pi$.

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        batch_size : int
            number of samples to draw
        replace : bool
            sampling with or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays
            indices of source and target data samples from pi
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan and draw source and target samples from it.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        replace : bool
            sampling with or without replacement

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            source minibatch drawn from OT plan
        x1[j] : Tensor, shape (bs, *dim)
            target minibatch drawn from OT plan
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_scipy(self, x0, x1):
        r"""Compute deterministic OT assignment using scipy's Hungarian algorithm.

        Returns column indices j such that x0[j] is optimally paired with x1.
        Only permutes x0; x1 order is preserved by construction.
        Lower variance than sample_plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        j : numpy array, shape (bs,)
            permutation indices for x0 that minimize transport cost to x1
        """
        import scipy

        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0.detach(), x1.detach()) ** 2
        if self.normalize_cost:
            M = M / M.max()
        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        return j


class MaskedOTPlanSampler:
    """Wrapper around OTPlanSampler that handles masked/variable-length
    protein data.

    Before computing the OT cost matrix, padded positions (where mask=False)
    are zeroed out so they don't affect the transport cost. The original
    [B, N, 3] tensor shapes are preserved in the output (unlike base
    OTPlanSampler which flattens).

    Uses scipy's linear_sum_assignment for deterministic OT (lower variance).
    Only x_0 (noise) is permuted; x_1 (data) order is preserved.
    """

    def __init__(
        self,
        method: str = "exact",
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        self.ot_plan_sampler = OTPlanSampler(
            method=method,
            reg=reg,
            reg_m=reg_m,
            normalize_cost=normalize_cost,
            num_threads=num_threads,
            warn=warn,
        )

    def sample_plan(
        self,
        x_0: Tensor,
        x_1: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Re-pair noise with data via OT, handling variable-length masks.

        Zeroes out padded positions before computing OT cost, then applies
        deterministic Hungarian assignment. Returns permuted noise in original
        [B, N, 3] shape.

        Args:
            x_0: Noise samples, shape [B, N, 3]
            x_1: Data samples, shape [B, N, 3]
            mask: Boolean residue mask, shape [B, N]. True = valid position.

        Returns:
            Tuple of (x_0_permuted, x_1_unchanged), both shape [B, N, 3].
        """
        import scipy

        B, N, D = x_0.shape

        # Zero out padded positions so they don't affect OT cost
        mask_3d = mask[..., None]  # [B, N, 1]
        x_0_masked = (x_0 * mask_3d).reshape(B, -1)  # [B, N*3]
        x_1_masked = (x_1 * mask_3d).reshape(B, -1)  # [B, N*3]

        # Compute cost matrix
        M = torch.cdist(x_0_masked, x_1_masked) ** 2  # [B, B]
        if self.ot_plan_sampler.normalize_cost:
            M = M / M.max()

        # Solve assignment (Hungarian algorithm)
        _, j = scipy.optimize.linear_sum_assignment(M.detach().cpu().numpy())

        # Permute x_0 using original (non-flattened) tensor
        j_tensor = torch.tensor(j, dtype=torch.long, device=x_0.device)
        x_0_permuted = x_0[j_tensor]

        return x_0_permuted, x_1

    def sample_plan_with_scipy(
        self,
        x_0: Tensor,
        x_1: Tensor,
        mask: Tensor,
    ) -> np.ndarray:
        """Compute deterministic OT assignment with masking, return indices.

        Same as sample_plan but returns raw permutation indices instead of
        permuted tensors. Matches the interface used in training_step:
            ot_noise_idx = self.ot_sampler.sample_plan_with_scipy(x_0, x_1, mask)
            x_0 = x_0[ot_noise_idx]

        Args:
            x_0: Noise samples, shape [B, N, 3]
            x_1: Data samples, shape [B, N, 3]
            mask: Boolean residue mask, shape [B, N]. True = valid position.

        Returns:
            j: numpy array of shape (B,), permutation indices for x_0.
        """
        import scipy

        B = x_0.shape[0]

        # Zero out padded positions so they don't affect OT cost
        mask_3d = mask[..., None]  # [B, N, 1]
        x_0_masked = (x_0 * mask_3d).reshape(B, -1)  # [B, N*3]
        x_1_masked = (x_1 * mask_3d).reshape(B, -1)  # [B, N*3]

        # Compute cost matrix
        M = torch.cdist(x_0_masked, x_1_masked) ** 2  # [B, B]
        if self.ot_plan_sampler.normalize_cost:
            M = M / M.max()

        # Solve assignment (Hungarian algorithm)
        _, j = scipy.optimize.linear_sum_assignment(M.detach().cpu().numpy())

        return j
