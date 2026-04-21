"""Pytest conftest — runs before any test module collection.

Pre-imports torch_sparse (which itself imports from torch_scatter) so the
symbols are resolved against the real torch_scatter package. Without this,
test modules like tests/test_training_step.py and
tests/test_loss_accumulation.py stub `torch_scatter` at module-import time,
which later breaks `from torch_scatter import scatter_add, segment_csr`
inside torch_sparse for any subsequent test that uses torch_geometric.
"""
import torch_sparse  # noqa: F401
import torch_geometric  # noqa: F401
