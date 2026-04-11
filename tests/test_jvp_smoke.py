"""
JVP smoke test for MeanFlow training with ProteinTransformerAF3.

Verifies that torch.func.jvp works end-to-end with the actual network
architecture used in training. This is the highest-risk item in the
MeanFlow integration — if JVP fails here, the training_step won't work.

Works on CPU (stubs out CUDA-only deps like torch_scatter).
"""

import sys
import types
import unittest.mock as mock

# Stub torch_scatter before any proteinfoundation import — the CUDA .so
# fails to load on CPU-only nodes, but scatter_mean is unused by our test.
_ts_stub = types.ModuleType("torch_scatter")
_ts_stub.scatter_mean = None
sys.modules["torch_scatter"] = _ts_stub

import torch


def build_small_nn():
    """Build a small ProteinTransformerAF3 matching the ca_af3_60M_notri config
    but with reduced dimensions for fast testing."""
    from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3

    kwargs = dict(
        name="ca_af3",
        token_dim=64,
        nlayers=2,
        nheads=4,
        residual_mha=True,
        residual_transition=True,
        parallel_mha_transition=False,
        use_attn_pair_bias=True,
        strict_feats=False,
        feats_init_seq=["res_seq_pdb_idx", "chain_break_per_res"],
        feats_cond_seq=["time_emb", "delta_t_emb"],  # MeanFlow: includes delta_t_emb
        t_emb_dim=32,
        dim_cond=64,
        idx_emb_dim=32,
        fold_emb_dim=32,
        feats_pair_repr=["rel_seq_sep", "xt_pair_dists"],
        feats_pair_cond=["time_emb", "delta_t_emb"],  # MeanFlow: includes delta_t_emb
        xt_pair_dist_dim=16,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3,
        x_sc_pair_dist_dim=16,
        x_sc_pair_dist_min=0.1,
        x_sc_pair_dist_max=3,
        x_motif_pair_dist_dim=16,
        x_motif_pair_dist_min=0.1,
        x_motif_pair_dist_max=3,
        seq_sep_dim=127,
        pair_repr_dim=32,
        update_pair_repr=False,
        update_pair_repr_every_n=2,
        use_tri_mult=False,
        num_registers=4,
        use_qkln=True,
        num_buckets_predict_pair=16,
        multilabel_mode="sample",
        cath_code_dir=".",
    )
    return ProteinTransformerAF3(**kwargs)


def test_jvp_forward():
    """Test that torch.func.jvp produces finite outputs with ProteinTransformerAF3."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    nn = build_small_nn().to(device).float()
    nn.eval()  # Avoid dropout variability

    B, N = 2, 16
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    # Inputs matching the training_step u_func closure
    z = torch.randn(B, N, 3, device=device, dtype=torch.float32)
    t_ext = torch.rand(B, 1, 1, device=device, dtype=torch.float32)
    r_ext = t_ext * torch.rand(B, 1, 1, device=device, dtype=torch.float32)  # r <= t

    # Tangent vectors
    v = torch.randn_like(z)  # dz/dt = conditional velocity
    dtdt = torch.ones_like(t_ext)
    drdt = torch.zeros_like(r_ext)

    def u_func(z_in, t_in, r_in):
        h = (t_in - r_in).squeeze(-1).squeeze(-1)
        t_flat = t_in.squeeze(-1).squeeze(-1)
        batch_nn = {
            "x_t": z_in,
            "t": t_flat,
            "h": h,
            "mask": mask,
        }
        nn_out = nn(batch_nn)
        return nn_out["coors_pred"]

    # Run JVP (autocast disabled in training_step; on CPU it's a no-op)
    u_pred, dudt = torch.func.jvp(u_func, (z, t_ext, r_ext), (v, dtdt, drdt))

    # Check outputs
    assert u_pred.shape == (B, N, 3), f"u_pred shape mismatch: {u_pred.shape}"
    assert dudt.shape == (B, N, 3), f"dudt shape mismatch: {dudt.shape}"
    assert torch.isfinite(u_pred).all(), f"u_pred has non-finite values: {u_pred}"
    assert torch.isfinite(dudt).all(), f"dudt has non-finite values: {dudt}"

    print(f"u_pred shape: {u_pred.shape}, mean: {u_pred.mean():.6f}, std: {u_pred.std():.6f}")
    print(f"dudt   shape: {dudt.shape}, mean: {dudt.mean():.6f}, std: {dudt.std():.6f}")
    print("JVP forward pass: OK")


def test_jvp_backward():
    """Test that gradients flow back through the JVP-based MeanFlow loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nn = build_small_nn().to(device).float()
    nn.train()

    B, N = 2, 16
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    z = torch.randn(B, N, 3, device=device, dtype=torch.float32)
    t_ext = torch.rand(B, 1, 1, device=device, dtype=torch.float32)
    r_ext = t_ext * torch.rand(B, 1, 1, device=device, dtype=torch.float32)
    v = torch.randn_like(z)

    dtdt = torch.ones_like(t_ext)
    drdt = torch.zeros_like(r_ext)

    def u_func(z_in, t_in, r_in):
        h = (t_in - r_in).squeeze(-1).squeeze(-1)
        t_flat = t_in.squeeze(-1).squeeze(-1)
        batch_nn = {
            "x_t": z_in,
            "t": t_flat,
            "h": h,
            "mask": mask,
        }
        nn_out = nn(batch_nn)
        return nn_out["coors_pred"]

    u_pred, dudt = torch.func.jvp(u_func, (z, t_ext, r_ext), (v, dtdt, drdt))

    # MeanFlow loss (same as training_step)
    u_tgt = (v - (t_ext - r_ext) * dudt).detach()
    error = (u_pred - u_tgt) * mask[..., None]
    nres = mask.sum(dim=-1) * 3
    loss = ((error ** 2).sum(dim=(-1, -2)) / nres).mean()

    loss.backward()

    # Verify gradients exist on network parameters
    n_params_with_grad = 0
    n_params_total = 0
    for name, p in nn.named_parameters():
        if p.requires_grad:
            n_params_total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                n_params_with_grad += 1

    print(f"Loss: {loss.item():.6f}")
    print(f"Parameters with nonzero grad: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad > 0, "No parameters received gradients!"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # Check no NaN grads
    for name, p in nn.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

    print("JVP backward pass: OK")


if __name__ == "__main__":
    test_jvp_forward()
    print()
    test_jvp_backward()
    print("\nAll JVP smoke tests passed!")
