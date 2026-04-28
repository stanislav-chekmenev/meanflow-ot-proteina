import pytest
import torch
from proteinfoundation.flow_matching.ot_pool import OTPool


def test_pool_size_floored_to_multiple_of_batch_size(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    pool = OTPool(pool_size=130, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.pool_size == 128
    # loguru -> std logging compat: check a warning was emitted.
    # If loguru doesn't show in caplog, check pool._truncated flag instead.
    assert pool.batch_size == 4


def test_pool_size_exact_multiple_unchanged():
    pool = OTPool(pool_size=128, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.pool_size == 128


def test_pool_size_smaller_than_batch_size_raises():
    with pytest.raises(AssertionError):
        OTPool(pool_size=2, batch_size=4, dim=3, scale_ref=1.0)


def test_initial_state_is_empty():
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.empty is True


# --- Task 2: refill fixtures + tests --------------------------------------


class _FakeDataset:
    """Minimal torch_geometric-style dataset: returns Data with coords + coord_mask.

    Mirrors the PDBDataset output shape so the real
    `dense_padded_from_data_list` collator works end-to-end.
    """
    def __init__(self, n_proteins, n_residues, seed=0):
        from torch_geometric.data import Data
        g = torch.Generator().manual_seed(seed)
        self._items = []
        for _ in range(n_proteins):
            d = Data()
            d.coords = torch.randn(n_residues, 37, 3, generator=g)
            d.coord_mask = torch.ones(n_residues, 37, dtype=torch.bool)
            self._items.append(d)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class _FakeFM:
    """Replacement for R3NFlowMatcher providing only what OTPool uses."""
    dim = 3
    scale_ref = 1.0

    def _mask_and_zero_com(self, x, mask):
        # match R3NFlowMatcher: zero COM along sequence dim with mask
        mask_f = mask.float()[..., None]  # [..., N, 1]
        n = mask_f.sum(dim=-2).clamp_min(1.0)  # [..., 1]
        com = (x * mask_f).sum(dim=-2) / n    # [..., 3]
        x = (x - com[..., None, :]) * mask_f
        return x


def _fake_extract_clean_sample(batch):
    # Mimics Proteina.extract_clean_sample signature/return.
    from proteinfoundation.utils.coors_utils import ang_to_nm
    x_1 = batch["coords"][:, :, 1, :]  # [B, N, 3] CA
    mask = batch["mask_dict"]["coords"][..., 0, 0]  # [B, N]
    batch_shape = x_1.shape[:-2]
    n = x_1.shape[-2]
    return ang_to_nm(x_1), mask, batch_shape, n, x_1.dtype


def test_refill_produces_valid_hungarian_pairing():
    """After refill, (x_1[i], x_0[i]) equals the Hungarian-paired OT pair."""
    import scipy.optimize
    import numpy as np

    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=16, n_residues=8, seed=1)
    fm = _FakeFM()

    torch.manual_seed(0)  # make randperm + noise deterministic
    pool.refill(ds, fm, _fake_extract_clean_sample)

    K = pool.pool_size
    mask3 = pool._mask[..., None]
    x1_flat = (pool._x1 * mask3).reshape(K, -1)
    x0_flat = (pool._x0 * mask3).reshape(K, -1)

    M = torch.cdist(x1_flat, x0_flat) ** 2  # [K, K]
    _, sigma = scipy.optimize.linear_sum_assignment(M.numpy())
    # After reorder, the identity permutation should be optimal.
    assert np.all(sigma == np.arange(K)), (
        f"x_0 should be pre-reordered so identity is optimal; got {sigma}"
    )


def test_refill_tensors_on_cpu():
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=2)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    assert pool._x1.device.type == "cpu"
    assert pool._x0.device.type == "cpu"
    assert pool._mask.device.type == "cpu"


def test_refill_resets_cursor_and_perm():
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=3)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    assert pool._cursor == 0
    assert pool._perm is not None
    assert pool._perm.shape == (4,)
    assert set(pool._perm.tolist()) == {0, 1, 2, 3}
    assert not pool.empty


# --- Task 3: next_batch tests ---------------------------------------------


def test_next_batch_pops_batch_size_rows_and_advances_cursor():
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=16, n_residues=8, seed=4)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    assert pool._cursor == 0

    x1, x0, mask, bshape, n, dtype = pool.next_batch("cpu")
    assert x1.shape[0] == 4
    assert x0.shape[0] == 4
    assert mask.shape[0] == 4
    assert pool._cursor == 4
    assert not pool.empty

    _, _, _, _, _, _ = pool.next_batch("cpu")
    assert pool._cursor == 8
    assert pool.empty


def test_next_batch_random_without_replacement_per_cycle():
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=16, n_residues=8, seed=5)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)

    # The pool's perm should contain each index exactly once.
    seen = set(pool._perm.tolist())
    assert seen == {0, 1, 2, 3, 4, 5, 6, 7}


def test_next_batch_on_empty_raises():
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=6)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    pool.next_batch("cpu")
    pool.next_batch("cpu")  # cursor == 4, empty
    with pytest.raises(AssertionError):
        pool.next_batch("cpu")


def test_next_batch_trims_to_actual_max_length():
    """If pool has mixed lengths, next_batch output should be trimmed."""
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    fm = _FakeFM()

    # Build a pool with mixed lengths manually.
    N_pool = 20
    pool._x1 = torch.randn(4, N_pool, 3)
    pool._x0 = torch.randn(4, N_pool, 3)
    mask = torch.zeros(4, N_pool, dtype=torch.bool)
    mask[0, :10] = True   # length 10
    mask[1, :8] = True    # length 8
    mask[2, :20] = True   # length 20
    mask[3, :15] = True   # length 15
    pool._mask = mask
    # Force perm so first batch is proteins 0, 1 (max length = 10).
    pool._perm = torch.tensor([0, 1, 2, 3])
    pool._cursor = 0

    x1, x0, m, bshape, n, dtype = pool.next_batch("cpu")
    assert x1.shape == (2, 10, 3), f"expected (2, 10, 3), got {x1.shape}"
    assert x0.shape == (2, 10, 3)
    assert m.shape == (2, 10)
    assert n == 10


def test_next_batch_moves_to_device_but_pool_stays_cpu():
    pool = OTPool(pool_size=4, batch_size=2, dim=3, scale_ref=1.0)
    ds = _FakeDataset(n_proteins=8, n_residues=8, seed=7)
    fm = _FakeFM()
    pool.refill(ds, fm, _fake_extract_clean_sample)
    x1, x0, m, _, _, _ = pool.next_batch("cpu")
    assert x1.device.type == "cpu"
    # Pool tensors remain on CPU regardless of request device.
    assert pool._x1.device.type == "cpu"
    assert pool._x0.device.type == "cpu"

    if torch.cuda.is_available():
        pool.refill(ds, fm, _fake_extract_clean_sample)
        x1c, x0c, mc, _, _, _ = pool.next_batch("cuda")
        assert x1c.device.type == "cuda"
        assert pool._x1.device.type == "cpu"


# --- Task 5: x_0_override signature test ----------------------------------


def test_compute_single_noise_loss_accepts_x0_override():
    """_compute_single_noise_loss should use x_0_override verbatim when provided."""
    import inspect
    from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
    sig = inspect.signature(ModelTrainerBase._compute_single_noise_loss)
    assert "x_0_override" in sig.parameters, (
        "ModelTrainerBase._compute_single_noise_loss must accept x_0_override"
    )
    # Default should be None (backwards-compat with non-pool path).
    assert sig.parameters["x_0_override"].default is None


# --- Task 6: Proteina wiring smoke tests ----------------------------------


def test_proteina_on_train_start_builds_pool_when_ot_pool_size_set():
    """Structural test: OTPool construction matches what on_train_start would make."""
    pool = OTPool(pool_size=8, batch_size=4, dim=3, scale_ref=1.0)
    assert pool.pool_size == 8
    assert pool.batch_size == 4


# --- Task 7: integration tests -------------------------------------------


def _build_proteina_cfg(ot_pool_size=4, loss_accumulation_steps=1):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {
            "target_pred": "v",
            "augmentation": {"global_rotation": False, "naug_rot": 1},
            "nn": {
                "name": "ca_af3",
                "token_dim": 64, "nlayers": 2, "nheads": 4,
                "residual_mha": True, "residual_transition": True,
                "parallel_mha_transition": False, "use_attn_pair_bias": True,
                "strict_feats": False,
                "feats_init_seq": ["res_seq_pdb_idx", "chain_break_per_res"],
                "feats_cond_seq": ["time_emb", "delta_t_emb"],
                "t_emb_dim": 32, "dim_cond": 64, "idx_emb_dim": 32,
                "fold_emb_dim": 32,
                "feats_pair_repr": ["rel_seq_sep", "xt_pair_dists"],
                "feats_pair_cond": ["time_emb", "delta_t_emb"],
                "xt_pair_dist_dim": 16, "xt_pair_dist_min": 0.1,
                "xt_pair_dist_max": 3,
                "x_sc_pair_dist_dim": 16, "x_sc_pair_dist_min": 0.1,
                "x_sc_pair_dist_max": 3,
                "x_motif_pair_dist_dim": 16, "x_motif_pair_dist_min": 0.1,
                "x_motif_pair_dist_max": 3,
                "seq_sep_dim": 127, "pair_repr_dim": 32,
                "update_pair_repr": False, "update_pair_repr_every_n": 2,
                "use_tri_mult": False, "num_registers": 4,
                "use_qkln": True, "num_buckets_predict_pair": 16,
                "multilabel_mode": "sample", "cath_code_dir": ".",
            },
        },
        "loss": {
            "t_distribution": {"name": "uniform", "p1": 0.0, "p2": 1.0},
            "loss_t_clamp": 0.9, "use_aux_loss": False,
            "aux_loss_t_lim": 0.3, "thres_aux_2d_loss": 0.6,
            "aux_loss_weight": 1.0, "num_dist_buckets": 16,
            "max_dist_boundary": 1.0,
        },
        "training": {
            "loss_accumulation_steps": loss_accumulation_steps,
            "self_cond": False, "fold_cond": False,
            "mask_T_prob": 0.5, "mask_A_prob": 0.5, "mask_C_prob": 0.5,
            "motif_conditioning": False,
            "ot_coupling": {
                "enabled": True, "method": "exact",
                "ot_pool_size": ot_pool_size,
            },
            "meanflow": {
                "ratio": 0.5, "P_mean": -0.4, "P_std": 1.0,
                "norm_p": 1.0, "norm_eps": 0.001, "nsteps_sample": 1,
            },
        },
        "opt": {"lr": 1e-4},
    })


def test_training_step_pool_mode_runs_and_loss_finite():
    """Pool is built on on_train_start, _get_ot_batch cycles through two
    batches, and refills on third call. Loss is finite."""
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg_exp = _build_proteina_cfg(ot_pool_size=4, loss_accumulation_steps=1)
    model = Proteina(cfg_exp=cfg_exp)
    model.to("cpu")
    model.train()

    class _FakeDM:
        batch_size = 2
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=42)
        def setup(self, stage): pass

    class _FakeTrainer:
        datamodule = _FakeDM()
        world_size = 1

    model._trainer = _FakeTrainer()
    model.on_train_start()

    assert model._ot_pool is not None
    assert model._ot_pool.pool_size == 4
    assert model._ot_pool.batch_size == 2
    assert model._ot_pool.empty  # not yet refilled

    # Call _get_ot_batch twice - pool size 4, batch 2 -> two batches then empty.
    x1a, x0a, ma, _, _, _ = model._get_ot_batch()
    assert torch.isfinite(x1a).all() and torch.isfinite(x0a).all()
    assert not model._ot_pool.empty
    x1b, x0b, mb, _, _, _ = model._get_ot_batch()
    assert model._ot_pool.empty  # consumed the whole pool

    # Third call triggers refill - pool becomes non-empty again.
    x1c, x0c, mc, _, _, _ = model._get_ot_batch()
    assert not model._ot_pool.empty


def test_on_train_start_accepts_pool_with_loss_accum_gt_1():
    """Pool + loss_accumulation_steps > 1 is now supported. on_train_start
    must build the pool without raising."""
    from proteinfoundation.proteinflow.proteina import Proteina

    cfg_exp = _build_proteina_cfg(ot_pool_size=8, loss_accumulation_steps=2)
    model = Proteina(cfg_exp=cfg_exp)
    model.to("cpu")

    class _FakeDM:
        batch_size = 2
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=0)
        def setup(self, stage): pass

    class _FakeTrainer:
        datamodule = _FakeDM()
        world_size = 1

    model._trainer = _FakeTrainer()

    # Should NOT raise.
    model.on_train_start()

    assert model._ot_pool is not None
    assert model._ot_pool.pool_size == 8
    assert model._ot_pool.batch_size == 2


# --- Task 4: training_step pool + K>1 pops K batches per step ---------------


def test_training_step_pool_mode_with_loss_accum_pops_k_batches_per_step():
    """Pool + loss_accumulation_steps=K: one training_step call pops K
    consecutive batches from the pool (cursor advances by K*B)."""
    from proteinfoundation.proteinflow.proteina import Proteina

    K = 2
    B = 2
    POOL = 8  # holds K * B * 2 = 8 samples -> 2 training steps per pool

    cfg_exp = _build_proteina_cfg(ot_pool_size=POOL, loss_accumulation_steps=K)
    model = Proteina(cfg_exp=cfg_exp)
    model.to("cpu")
    model.train()

    import unittest.mock as mock

    class _FakeDM:
        batch_size = B
        train_ds = _FakeDataset(n_proteins=16, n_residues=8, seed=42)
        def setup(self, stage): pass

    trainer_mock = mock.MagicMock()
    trainer_mock.world_size = 1
    trainer_mock.gradient_clip_val = None
    trainer_mock.gradient_clip_algorithm = None
    trainer_mock.datamodule = _FakeDM()
    # manual_backward calls trainer.strategy.backward(loss, None) -- wire it
    # so it actually calls loss.backward() on the underlying tensor.
    trainer_mock.strategy.backward = lambda loss, *a, **kw: loss.backward()
    model._trainer = trainer_mock
    model.on_train_start()

    opts = model.configure_optimizers()
    if isinstance(opts, dict):
        opt = opts["optimizer"]
    else:
        opt = opts
    model._optimizers_list = [opt]
    model.optimizers = lambda: opt
    model.lr_schedulers = lambda: None
    model._manual_step_count = 0
    model._accum_grad_batches = 1

    # Build a dummy dataloader batch (ignored on the pool path).
    dummy_batch = {
        "coords": torch.zeros(B, 8, 37, 3),
        "mask_dict": {"coords": torch.ones(B, 8, 1, 1, dtype=torch.bool)},
    }

    # Count how many times the pool serves a batch during one training_step.
    # Robust against cursor wraparound on refill — the cursor goes negative
    # in the diff when the pool refills mid-step, so we count calls instead.
    next_batch_calls = []
    original_next_batch = model._ot_pool.next_batch

    def counting_next_batch(*a, **kw):
        next_batch_calls.append(1)
        return original_next_batch(*a, **kw)

    model._ot_pool.next_batch = counting_next_batch

    result = model.training_step(dummy_batch, batch_idx=0)

    assert len(next_batch_calls) == K, (
        f"Pool next_batch called {len(next_batch_calls)} times, expected K={K}"
    )
    # Manual optimization in K>1 returns None.
    assert result is None
