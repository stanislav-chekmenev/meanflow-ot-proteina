"""Integration tests for OT coupling with the flow matching pipeline."""

import torch
import pytest

from proteinfoundation.flow_matching.ot_sampler import MaskedOTPlanSampler
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher


class TestOTIntegrationWithFlowMatcher:
    """Test OT coupling works correctly within the flow matching pipeline."""

    def test_ot_coupled_interpolation_same_shape(self):
        """OT-coupled interpolation should produce same shapes as standard."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 30
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)
        t = torch.rand(B)

        # Standard interpolation
        x_t_standard = fm.interpolate(x_0, x_1, t, mask)

        # OT-coupled interpolation
        x_0_ot, _ = ot_sampler.sample_plan(x_0, x_1, mask)
        x_t_ot = fm.interpolate(x_0_ot, x_1, t, mask)

        assert x_t_standard.shape == x_t_ot.shape == (B, N, 3)

    def test_ot_reduces_average_transport_distance(self):
        """OT-paired (x_0, x_1) should have lower avg distance than random."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 32, 50
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)

        random_dist = ((x_0 - x_1) ** 2).sum(dim=(-1, -2)).mean()
        x_0_ot, _ = ot_sampler.sample_plan(x_0, x_1, mask)
        ot_dist = ((x_0_ot - x_1) ** 2).sum(dim=(-1, -2)).mean()

        assert ot_dist <= random_dist

    def test_ot_with_variable_length_masks(self):
        """OT coupling should work with variable-length proteins."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 40
        mask = torch.ones(B, N, dtype=torch.bool)
        lengths = [15, 20, 25, 30, 35, 40, 18, 22]
        for i, length in enumerate(lengths):
            mask[i, length:] = False

        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)
        x_0_ot, x_1_ot = ot_sampler.sample_plan(x_0, x_1, mask)

        assert x_0_ot.shape == (B, N, 3)
        assert torch.equal(x_1_ot, x_1)

    def test_ot_preserves_zero_com(self):
        """OT coupling should preserve zero center-of-mass."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 30
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)

        x_0_ot, _ = ot_sampler.sample_plan(x_0, x_1, mask)
        # OT only permutes rows -- each row was already zero-COM
        com = x_0_ot.mean(dim=1)  # [B, 3]
        assert torch.allclose(com, torch.zeros_like(com), atol=1e-5)

    def test_full_training_step_shapes(self):
        """Simulate the full training step flow with OT coupling."""
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        ot_sampler = MaskedOTPlanSampler(method="exact")

        B, N = 8, 30
        mask = torch.ones(B, N, dtype=torch.bool)
        x_1 = fm._mask_and_zero_com(torch.randn(B, N, 3), mask)
        t = torch.rand(B)
        x_0 = fm.sample_reference(n=N, shape=(B,), mask=mask)

        # OT coupling (the new step)
        x_0, _ = ot_sampler.sample_plan(x_0, x_1, mask)

        # Interpolation (existing step)
        x_t = fm.interpolate(x_0, x_1, t, mask)

        # Target (existing step)
        x_t_dot = fm.xt_dot(x_1, x_t, t, mask)

        assert x_t.shape == (B, N, 3)
        assert x_t_dot.shape == (B, N, 3)
