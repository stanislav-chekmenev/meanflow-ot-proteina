"""Chirality-sensitive loss helpers for MeanFlow training.

The signed triple product of four sequential CA atoms is a scalar that flips
sign under reflection, so it provides a handedness signal that the raw FM
loss (which is O(3)-invariant on a single protein) lacks.
"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def triple_products(
    x: Float[Tensor, "b n 3"],
    stride: int = 1,
) -> Float[Tensor, "b m"]:
    """Per-residue signed volume of (x_i, x_{i+k}, x_{i+2k}, x_{i+3k}).

    For each i in [0, n - 3k), builds three edge vectors rooted at x_i and
    computes the determinant of the 3x3 matrix formed by stacking them.
    Flips sign under reflection of ``x``.

    Args:
        x: CA coordinates, shape [B, n, 3].
        stride: Spacing k between the four points used per triple (default 1).

    Returns:
        Signed volumes, shape [B, n - 3*stride].
    """
    k = stride
    n = x.shape[-2]
    if n <= 3 * k:
        raise ValueError(f"Need n > 3*stride; got n={n}, stride={k}")

    # Edge vectors [B, n - 3k, 3]
    e1 = x[:, k : n - 2 * k] - x[:, : n - 3 * k]
    e2 = x[:, 2 * k : n - k] - x[:, : n - 3 * k]
    e3 = x[:, 3 * k : n] - x[:, : n - 3 * k]

    # Stack into [B, n - 3k, 3, 3] where rows are edges.
    M = torch.stack([e1, e2, e3], dim=-2)
    return torch.linalg.det(M)


def _triple_mask(mask: Bool[Tensor, "b n"], stride: int) -> Bool[Tensor, "b m"]:
    """Validity mask for triple-product indices: all four residues must be valid."""
    k = stride
    n = mask.shape[-1]
    return (
        mask[:, : n - 3 * k]
        & mask[:, k : n - 2 * k]
        & mask[:, 2 * k : n - k]
        & mask[:, 3 * k : n]
    )


def chirality_hinge_loss_per_sample(
    x_pred: Float[Tensor, "b n 3"],
    x_gt: Float[Tensor, "b n 3"],
    mask: Bool[Tensor, "b n"],
    margin_alpha: float = 0.1,
    stride: int = 1,
) -> Float[Tensor, "b"]:
    """Per-sample chirality hinge loss (sum of per-element hinges / per-sample
    valid count). Same math as :func:`chirality_hinge_loss`, but without the
    cross-batch reduction — lets callers apply per-sample weights (e.g., a
    t-gate) before averaging.

    Returns:
        Tensor of shape [B] with per-sample mean hinge loss.
    """
    T_gt = triple_products(x_gt, stride=stride)
    T_pred = triple_products(x_pred, stride=stride)
    valid = _triple_mask(mask, stride=stride)

    m_T = margin_alpha * T_gt.abs()
    signed_agreement = torch.sign(T_gt) * T_pred
    hinge = torch.relu(m_T - signed_agreement) * valid  # [B, m]

    per_sample_count = valid.sum(dim=-1).clamp(min=1).to(T_gt.dtype)  # [B]
    return hinge.sum(dim=-1) / per_sample_count  # [B]


def chirality_hinge_loss(
    x_pred: Float[Tensor, "b n 3"],
    x_gt: Float[Tensor, "b n 3"],
    mask: Bool[Tensor, "b n"],
    margin_alpha: float = 0.1,
    stride: int = 1,
) -> Float[Tensor, ""]:
    """Hinge loss penalizing wrong-handedness predictions.

    Computes signed triple products for GT and prediction, and applies a
    per-element hinge: for each valid index i,

    ``relu(alpha * |T_gt_i| - sign(T_gt_i) * T_pred_i)``

    The loss is the sum of per-element hinges divided by the number of valid
    indices. Zero when predictions match GT handedness with triple-product
    magnitude at least ``alpha * |T_gt_i|``; linear in shortfall otherwise.

    Args:
        x_pred: Predicted CA coordinates, shape [B, n, 3], requires grad.
        x_gt: Ground-truth CA coordinates, shape [B, n, 3].
        mask: Per-residue validity, shape [B, n].
        margin_alpha: Per-element margin scale; margin at index i is
            ``alpha * |T_gt_i|`` (default 0.1).
        stride: Stride k for the triple-product window.

    Returns:
        Scalar loss.
    """
    T_gt = triple_products(x_gt, stride=stride)
    T_pred = triple_products(x_pred, stride=stride)
    valid = _triple_mask(mask, stride=stride)

    valid_count = valid.sum().clamp(min=1).to(T_gt.dtype)
    # Per-element margin: alpha * |T_gt_i|. Ensures relu is zero when pred == gt
    # (relu((alpha - 1) * |T_gt_i|) = 0 for alpha < 1) and positive at mirror.
    m_T = margin_alpha * T_gt.abs()

    signed_agreement = torch.sign(T_gt) * T_pred  # >0 when predicted same handedness
    hinge = torch.relu(m_T - signed_agreement) * valid
    return hinge.sum() / valid_count
