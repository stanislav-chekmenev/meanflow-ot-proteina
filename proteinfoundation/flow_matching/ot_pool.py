"""CPU-resident OT pool: compute Hungarian pairing once per K/B training
steps, stream B-sized batches out of it random-without-replacement.

The pool owns (x_1, x_0, mask) tensors on CPU. Only next_batch() transfers
to GPU. refill() rebuilds the pool by sampling K dataset proteins + K
noise vectors and running scipy's Hungarian algorithm on the K x K
masked-coord cost matrix.
"""
import torch
from loguru import logger


class OTPool:
    def __init__(self, pool_size: int, batch_size: int, dim: int, scale_ref: float):
        effective = (pool_size // batch_size) * batch_size
        if effective != pool_size:
            logger.warning(
                "ot_pool_size={} not divisible by batch_size={}; flooring to {}",
                pool_size, batch_size, effective,
            )
        assert effective >= batch_size, (
            f"pool_size ({pool_size}) must be >= batch_size ({batch_size})"
        )
        self.pool_size = effective
        self.batch_size = batch_size
        self.dim = dim
        self.scale_ref = scale_ref
        self._x1 = None
        self._x0 = None
        self._mask = None
        self._perm = None
        self._cursor = self.pool_size  # trigger refill on first next_batch

    @property
    def empty(self) -> bool:
        return self._cursor >= self.pool_size

    def refill(self, dataset, fm, extract_clean_sample) -> None:
        raise NotImplementedError  # Task 2

    def next_batch(self, device):
        raise NotImplementedError  # Task 3
