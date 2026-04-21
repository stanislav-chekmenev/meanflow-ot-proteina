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
        """Sample K proteins + K noise, run Hungarian, store pool on CPU.

        Args:
            dataset: a dataset with __len__ and __getitem__ returning batch dicts.
            fm: flow matcher with _mask_and_zero_com(x, mask).
            extract_clean_sample: callable(batch) -> (x_1, mask, bshape, n, dtype).
        """
        import scipy.optimize
        from proteinfoundation.utils.dense_padding_data_loader import (
            dense_padded_from_data_list,
        )

        K = self.pool_size

        # 1. Sample K random dataset indices -> load and collate into a batch.
        idx = torch.randint(0, len(dataset), (K,))
        data_list = [dataset[int(i)] for i in idx]
        batch = dense_padded_from_data_list(data_list)

        # 2. Extract CA coords + masks via the same path Proteina uses.
        x_1_pool, mask_pool, _, _, dtype = extract_clean_sample(batch)
        x_1_pool = fm._mask_and_zero_com(x_1_pool, mask_pool)
        x_1_pool = x_1_pool.detach().cpu()
        mask_pool = mask_pool.cpu()

        N_pool = x_1_pool.shape[1]

        # 3. Sample noise on CPU, mask + zero COM.
        x_0_pool = torch.randn(K, N_pool, self.dim, dtype=dtype) * self.scale_ref
        x_0_pool = fm._mask_and_zero_com(x_0_pool, mask_pool)

        # 4. Build K x K cost matrix on flat masked coords.
        mask_3d = mask_pool[..., None]
        x_1_flat = (x_1_pool * mask_3d).reshape(K, -1)
        x_0_flat = (x_0_pool * mask_3d).reshape(K, -1)
        M = torch.cdist(x_1_flat, x_0_flat) ** 2  # [K, K]

        # 5. Hungarian.
        _, sigma = scipy.optimize.linear_sum_assignment(M.numpy())

        # 6. Reorder x_0 so (x_1[i], x_0[i]) is the OT pair.
        sigma_t = torch.as_tensor(sigma, dtype=torch.long)
        x_0_pool = x_0_pool[sigma_t]

        # 7. Store pool state (all CPU).
        self._x1 = x_1_pool.contiguous()
        self._x0 = x_0_pool.contiguous()
        self._mask = mask_pool.contiguous()

        # 8. Reset cursor + random serving order.
        self._perm = torch.randperm(self.pool_size)
        self._cursor = 0

        # 9. Free intermediates.
        del M, x_1_flat, x_0_flat

    def next_batch(self, device):
        """Pop B indices from the permutation, trim, and ship to device.

        Returns:
            (x_1, x_0, mask, batch_shape, n, dtype) on `device`.
        """
        assert not self.empty, "OTPool is empty - call refill() first."
        idx = self._perm[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size

        x_1_sel = self._x1[idx]      # [B, N_pool, 3] CPU
        x_0_sel = self._x0[idx]      # [B, N_pool, 3] CPU
        mask_sel = self._mask[idx]   # [B, N_pool]    CPU

        # Trim to max actual length among selected rows.
        n_sel = int(mask_sel.sum(dim=1).max().item())
        if n_sel == 0:
            n_sel = x_1_sel.shape[1]
        x_1_sel = x_1_sel[:, :n_sel, :]
        x_0_sel = x_0_sel[:, :n_sel, :]
        mask_sel = mask_sel[:, :n_sel]

        dtype = self._x1.dtype
        batch_shape = torch.Size([self.batch_size])

        return (
            x_1_sel.to(device),
            x_0_sel.to(device),
            mask_sel.to(device),
            batch_shape,
            n_sel,
            dtype,
        )
