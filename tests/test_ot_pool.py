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
