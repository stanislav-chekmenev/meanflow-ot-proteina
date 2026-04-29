"""CPU-only tests for scripts/dump_pdb_eval_ca_features.py helpers.

These exercise the non-GPU pieces — file loading and batch iteration. The
GearNet inference / metric_factory pieces are integration-only and require
CUDA + the user's data; they are not covered here.
"""
from __future__ import annotations

import os

import pytest
import torch
from torch_geometric.data import Batch, Data

from scripts.dump_pdb_eval_ca_features import (
    _load_data,
    iter_real_pyg_batches,
)


def _write_synthetic_pt(path: str, n_res: int, *, mask_dtype=torch.bool) -> None:
    """Persist a tiny PyG Data with the same attribute contract as the
    pre-processed training set: coords [L,37,3] float32, coord_mask [L,37] bool."""
    coords = torch.randn(n_res, 37, 3, dtype=torch.float32)
    coord_mask = torch.ones(n_res, 37, dtype=mask_dtype)
    data = Data(coords=coords, coord_mask=coord_mask)
    torch.save(data, path)


# -------------------------- _load_data ---------------------------------


def test_load_data_builds_minimal_pyg_data(tmp_path):
    pt = tmp_path / "x.pt"
    _write_synthetic_pt(str(pt), n_res=42)

    d = _load_data(str(pt))

    assert isinstance(d, Data)
    assert d.coords.shape == (42, 37, 3)
    assert d.coord_mask.shape == (42, 37)
    assert d.coord_mask.dtype == torch.bool
    assert d.node_id.shape == (42, 1)
    # node_id should be a sequential index [0..L-1]
    assert torch.equal(d.node_id.squeeze(-1), torch.arange(42))


# -------------------------- iter_real_pyg_batches -----------------------


def test_iter_real_pyg_batches_yields_correct_batches(tmp_path):
    # 5 files with varying length; batch_size=2 -> 2 full + 1 partial = 3 batches
    lengths = [10, 12, 50, 70, 80]
    for i, L in enumerate(lengths):
        _write_synthetic_pt(str(tmp_path / f"prot_{i}.pt"), n_res=L)

    batches = list(iter_real_pyg_batches(str(tmp_path), batch_size=2))
    assert len(batches) == 3
    total = 0
    for batch, stats in batches:
        assert isinstance(batch, Batch)
        assert hasattr(batch, "batch")
        # Each batch reports cumulative stats; per-batch sample count derivable
        # from the Batch object's num_graphs.
        total += batch.num_graphs
    assert total == len(lengths)

    # Final stats dict reflects all files processed
    _, last_stats = batches[-1]
    assert last_stats["processed"] == len(lengths)
    assert last_stats["skipped_len"] == 0
    assert last_stats["skipped_err"] == 0


def test_iter_real_pyg_batches_max_len_filter(tmp_path):
    # Two short, two long. With max_len=20 only the short two survive.
    lengths = [10, 15, 100, 200]
    for i, L in enumerate(lengths):
        _write_synthetic_pt(str(tmp_path / f"prot_{i}.pt"), n_res=L)

    batches = list(
        iter_real_pyg_batches(str(tmp_path), batch_size=4, max_len=20)
    )
    # All survivors fit in one partial batch
    assert len(batches) == 1
    batch, stats = batches[0]
    assert batch.num_graphs == 2
    assert stats["processed"] == 2
    assert stats["skipped_len"] == 2
    assert stats["skipped_err"] == 0


def test_iter_real_pyg_batches_skips_unloadable(tmp_path):
    # Mix one corrupt file with two valid ones.
    _write_synthetic_pt(str(tmp_path / "good_0.pt"), n_res=10)
    _write_synthetic_pt(str(tmp_path / "good_1.pt"), n_res=20)
    # Empty file -> torch.load will raise.
    (tmp_path / "broken.pt").write_bytes(b"")

    batches = list(iter_real_pyg_batches(str(tmp_path), batch_size=4))
    # The two good files end up in a single partial batch.
    assert len(batches) == 1
    batch, stats = batches[0]
    assert batch.num_graphs == 2
    assert stats["processed"] == 2
    assert stats["skipped_err"] == 1
    assert stats["skipped_len"] == 0


def test_iter_real_pyg_batches_empty_dir_raises(tmp_path):
    with pytest.raises(AssertionError):
        # The iterator must materialize the assertion eagerly — wrap in list()
        # to force evaluation in case it's a generator.
        list(iter_real_pyg_batches(str(tmp_path), batch_size=4))


def test_iter_real_pyg_batches_single_file_raises(tmp_path):
    _load_dir = tmp_path / "single"
    _load_dir.mkdir()
    _write_synthetic_pt(str(_load_dir / "lonely.pt"), n_res=10)
    with pytest.raises(AssertionError):
        list(iter_real_pyg_batches(str(_load_dir), batch_size=4))
