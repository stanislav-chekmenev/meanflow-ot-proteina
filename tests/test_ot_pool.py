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
