"""Tests for OTPlanSampler and MaskedOTPlanSampler.

Covers shape correctness, plan validity, permutation properties,
transport cost reduction, and variable-length masking behavior.
"""

import pytest
import torch

from proteinfoundation.flow_matching.ot_sampler import (
    MaskedOTPlanSampler,
    OTPlanSampler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
N_RESIDUES = 16
DIM = 3
SEED = 42


@pytest.fixture()
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


@pytest.fixture()
def x0_2d(rng: torch.Generator) -> torch.Tensor:
    """Flat [B, D] noise tensor."""
    return torch.randn(BATCH_SIZE, DIM, generator=rng)


@pytest.fixture()
def x1_2d(rng: torch.Generator) -> torch.Tensor:
    """Flat [B, D] data tensor."""
    return torch.randn(BATCH_SIZE, DIM, generator=rng) + 5.0


@pytest.fixture()
def x0_3d(rng: torch.Generator) -> torch.Tensor:
    """Protein-shaped [B, N, 3] noise tensor."""
    return torch.randn(BATCH_SIZE, N_RESIDUES, DIM, generator=rng)


@pytest.fixture()
def x1_3d(rng: torch.Generator) -> torch.Tensor:
    """Protein-shaped [B, N, 3] data tensor."""
    return torch.randn(BATCH_SIZE, N_RESIDUES, DIM, generator=rng) + 10.0


@pytest.fixture()
def full_mask() -> torch.Tensor:
    """All-true boolean mask [B, N]."""
    return torch.ones(BATCH_SIZE, N_RESIDUES, dtype=torch.bool)


# ===========================================================================
# TestOTPlanSampler
# ===========================================================================


class TestOTPlanSampler:
    """Tests for the base OTPlanSampler class."""

    def test_get_map_shape(self, x0_2d: torch.Tensor, x1_2d: torch.Tensor) -> None:
        """OT plan matrix should be [B, B]."""
        sampler = OTPlanSampler(method="exact")
        pi = sampler.get_map(x0_2d, x1_2d)
        assert pi.shape == (BATCH_SIZE, BATCH_SIZE)

    def test_get_map_is_valid_plan(
        self, x0_2d: torch.Tensor, x1_2d: torch.Tensor
    ) -> None:
        """Rows and columns of the plan should each sum to 1/B."""
        sampler = OTPlanSampler(method="exact")
        pi = sampler.get_map(x0_2d, x1_2d)
        expected_marginal = 1.0 / BATCH_SIZE
        torch.testing.assert_close(
            pi.sum(dim=1),
            torch.full((BATCH_SIZE,), expected_marginal),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            pi.sum(dim=0),
            torch.full((BATCH_SIZE,), expected_marginal),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_sample_plan_output_shapes(
        self, x0_2d: torch.Tensor, x1_2d: torch.Tensor
    ) -> None:
        """sample_plan should return tensors with the same shape as inputs."""
        sampler = OTPlanSampler(method="exact")
        x0_paired, x1_paired = sampler.sample_plan(x0_2d, x1_2d)
        assert x0_paired.shape == x0_2d.shape
        assert x1_paired.shape == x1_2d.shape

    def test_sample_plan_with_3d_tensors(
        self, x0_3d: torch.Tensor, x1_3d: torch.Tensor
    ) -> None:
        """sample_plan should work with [B, N, 3] protein tensors."""
        sampler = OTPlanSampler(method="exact")
        x0_paired, x1_paired = sampler.sample_plan(x0_3d, x1_3d)
        assert x0_paired.shape == x0_3d.shape
        assert x1_paired.shape == x1_3d.shape

    def test_sample_plan_with_scipy_preserves_x1_order(
        self, x0_3d: torch.Tensor, x1_3d: torch.Tensor
    ) -> None:
        """sample_plan_with_scipy should return x1 unchanged."""
        sampler = OTPlanSampler(method="exact")
        _, x1_out = sampler.sample_plan_with_scipy(x0_3d, x1_3d)
        torch.testing.assert_close(x1_out, x1_3d)


# ===========================================================================
# TestMaskedOTPlanSampler
# ===========================================================================


class TestMaskedOTPlanSampler:
    """Tests for the MaskedOTPlanSampler wrapper for variable-length proteins."""

    def test_output_shape_preserved(
        self,
        x0_3d: torch.Tensor,
        x1_3d: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> None:
        """Output tensors should keep the [B, N, 3] shape."""
        sampler = MaskedOTPlanSampler()
        x0_out, x1_out = sampler.sample_plan(x0_3d, x1_3d, full_mask)
        assert x0_out.shape == (BATCH_SIZE, N_RESIDUES, DIM)
        assert x1_out.shape == (BATCH_SIZE, N_RESIDUES, DIM)

    def test_x1_order_preserved(
        self,
        x0_3d: torch.Tensor,
        x1_3d: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> None:
        """x1 should be returned without permutation."""
        sampler = MaskedOTPlanSampler()
        _, x1_out = sampler.sample_plan(x0_3d, x1_3d, full_mask)
        torch.testing.assert_close(x1_out, x1_3d)

    def test_output_is_permutation_of_input(
        self,
        x0_3d: torch.Tensor,
        x1_3d: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> None:
        """Each row in x0_out should be a row that existed in x0_3d."""
        sampler = MaskedOTPlanSampler()
        x0_out, _ = sampler.sample_plan(x0_3d, x1_3d, full_mask)
        # Every row of x0_out must match some row of x0_3d
        for i in range(BATCH_SIZE):
            diffs = (x0_3d - x0_out[i : i + 1]).abs().sum(dim=(1, 2))  # [B]
            assert diffs.min().item() < 1e-6, (
                f"Row {i} of x0_out does not match any row in x0_3d"
            )

    def test_ot_reduces_total_transport_cost(
        self,
        x0_3d: torch.Tensor,
        x1_3d: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> None:
        """OT-paired cost should be <= identity pairing cost."""
        sampler = MaskedOTPlanSampler()
        x0_out, x1_out = sampler.sample_plan(x0_3d, x1_3d, full_mask)

        # Identity pairing cost: sum of squared distances with original order
        identity_cost = ((x0_3d - x1_3d) ** 2).sum().item()
        # OT pairing cost
        ot_cost = ((x0_out - x1_out) ** 2).sum().item()

        assert ot_cost <= identity_cost + 1e-6, (
            f"OT cost ({ot_cost:.4f}) should be <= identity cost ({identity_cost:.4f})"
        )

    def test_masking_affects_assignment(self) -> None:
        """Different masks should (in general) produce different assignments."""
        torch.manual_seed(123)
        B, N = 8, 20
        x0 = torch.randn(B, N, 3)
        x1 = torch.randn(B, N, 3) + 10.0

        sampler = MaskedOTPlanSampler()

        # Mask 1: all residues valid
        mask_full = torch.ones(B, N, dtype=torch.bool)
        x0_full, _ = sampler.sample_plan(x0, x1, mask_full)

        # Mask 2: only first half valid
        mask_half = torch.zeros(B, N, dtype=torch.bool)
        mask_half[:, : N // 2] = True
        x0_half, _ = sampler.sample_plan(x0, x1, mask_half)

        # The permutations should differ (with high probability for these inputs)
        assert not torch.allclose(x0_full, x0_half), (
            "Different masks should produce different OT assignments"
        )

    def test_variable_length_proteins(self) -> None:
        """Handle a batch where proteins have different valid lengths."""
        torch.manual_seed(456)
        B, N = 6, 24
        x0 = torch.randn(B, N, 3)
        x1 = torch.randn(B, N, 3) + 8.0

        # Each protein has a different length
        lengths = [8, 12, 16, 20, 24, 10]
        mask = torch.zeros(B, N, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        sampler = MaskedOTPlanSampler()
        x0_out, x1_out = sampler.sample_plan(x0, x1, mask)

        assert x0_out.shape == (B, N, 3)
        assert x1_out.shape == (B, N, 3)
        # x1 should be unchanged
        torch.testing.assert_close(x1_out, x1)
        # x0_out rows should be permutation of x0 rows
        for i in range(B):
            diffs = (x0 - x0_out[i : i + 1]).abs().sum(dim=(1, 2))
            assert diffs.min().item() < 1e-6
