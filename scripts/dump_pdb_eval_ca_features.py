"""Dump a Proteina-FID reference feature .pth from a directory of pre-processed .pt files.

Builds the same `{FID_real_features_sum, FID_real_features_cov_sum,
FID_real_features_num_samples}` schema as
`data/metric_factory/features/pdb_eval_ca_features.pth`, but accumulated from
an arbitrary training subset (e.g. the train_frac0.1 monomer pre-processed
directory). Useful when the FID reference shipped in the repo doesn't reflect
the train distribution and we want to point `fid.real_features_path` at a
matched reference instead.

Single-GPU only — `GenerationMetricFactory.dump_real_dataset_features`
asserts `world_size == 1`. We do not initialize DDP here.

DATA_PATH is NOT required by this script; we never load a Proteina checkpoint.

Usage
-----
    source .venv/bin/activate
    PYTHONPATH=. python scripts/dump_pdb_eval_ca_features.py \
        --data-dir /mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/monomers_minl50_maxl256_ds_frac_0.1/processed \
        --output data/metric_factory/features/pdb_eval_ca_features_monomers_frac0.1.pth \
        --batch-size 16 \
        --max-len 256
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time
from typing import Iterator, Tuple

import torch
from torch_geometric.data import Batch, Data

DEFAULT_DATA_DIR = (
    "/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/"
    "monomers_minl50_maxl256_ds_frac_0.1/processed"
)
DEFAULT_OUTPUT = (
    "data/metric_factory/features/pdb_eval_ca_features_monomers_frac0.1.pth"
)
DEFAULT_GEARNET_CKPT = "data/metric_factory/model_weights/gearnet_ca.pth"


def _load_data(pt_path: str) -> Data:
    """Load a pre-processed .pt file and return a minimal PyG `Data` carrying
    only what GearNet-CA needs: `coords`, `coord_mask` (bool), `node_id`.

    Mirrors the contract used by `scripts/fid_floor_probe.py::_load_data` and
    `proteinfoundation/metrics/metric_factory.py::DatasetWrapper.__getitem__`.
    """
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    coords = raw.coords  # [n_res, 37, 3]
    coord_mask = raw.coord_mask.bool()  # [n_res, 37]
    n_res = coords.shape[0]
    return Data(
        coords=coords,
        coord_mask=coord_mask,
        node_id=torch.arange(n_res).unsqueeze(-1),
    )


def iter_real_pyg_batches(
    data_dir: str,
    batch_size: int,
    max_len: int | None = None,
) -> Iterator[Tuple[Batch, dict]]:
    """Yield `(pyg_batch, stats)` tuples by loading every `*.pt` in `data_dir`
    in sorted (deterministic) order.

    Skips files that fail to load (counted in `stats["skipped_err"]`) and
    files longer than `max_len` (counted in `stats["skipped_len"]`). Flushes
    the partial last batch.

    Asserts at least 2 valid `.pt` files exist in `data_dir` (FID needs >=2).
    """
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    assert len(pt_files) >= 2, (
        f"Need at least 2 .pt files in {data_dir} to build a FID reference, "
        f"found {len(pt_files)}."
    )

    stats = {"processed": 0, "skipped_len": 0, "skipped_err": 0}
    buf: list[Data] = []
    for pt_path in pt_files:
        try:
            d = _load_data(pt_path)
        except Exception:
            stats["skipped_err"] += 1
            continue
        if max_len is not None and d.coords.shape[0] > max_len:
            stats["skipped_len"] += 1
            continue
        buf.append(d)
        if len(buf) == batch_size:
            stats["processed"] += len(buf)
            yield Batch.from_data_list(buf), dict(stats)
            buf = []
    if buf:
        stats["processed"] += len(buf)
        yield Batch.from_data_list(buf), dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory of pre-processed .pt files (one per protein).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination .pth path for the dumped real features.",
    )
    parser.add_argument("--gearnet-ckpt", default=DEFAULT_GEARNET_CKPT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="If set, skip structures longer than this (e.g. 256 to match the training filter).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Need a GPU for GearNet inference."

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Imported here so the helper module is importable on a CPU-only test box
    # without paying GearNet/torchdrug import cost.
    from proteinfoundation.metrics.metric_factory import GenerationMetricFactory

    print(
        f"[dump-features] Loading GenerationMetricFactory ckpt={args.gearnet_ckpt}"
    )
    mf = GenerationMetricFactory(
        metrics=["FID"],
        ckpt_path=args.gearnet_ckpt,
        ca_only=True,
        reset_real_features=False,
        real_features_path=None,  # critical: build from scratch, not load
    ).cuda()

    device = torch.device("cuda")
    start = time.time()
    last_logged = 0
    final_stats = {"processed": 0, "skipped_len": 0, "skipped_err": 0}

    for pyg_batch, stats in iter_real_pyg_batches(
        args.data_dir, batch_size=args.batch_size, max_len=args.max_len
    ):
        pyg_batch = pyg_batch.to(device)
        mf.update(pyg_batch, real=True)
        final_stats = stats
        if final_stats["processed"] - last_logged >= 500:
            elapsed = time.time() - start
            rate = final_stats["processed"] / elapsed if elapsed > 0 else 0.0
            print(
                f"[dump-features] processed={final_stats['processed']} "
                f"skipped_len={final_stats['skipped_len']} "
                f"skipped_err={final_stats['skipped_err']} "
                f"rate={rate:.1f} struct/s"
            )
            last_logged = final_stats["processed"]

    if final_stats["processed"] < 2:
        raise RuntimeError(
            f"Need at least 2 structures to build a FID reference, "
            f"got {final_stats['processed']}."
        )

    mf.dump_real_dataset_features(args.output)

    elapsed = time.time() - start
    print(
        f"[dump-features] Done. processed={final_stats['processed']} "
        f"skipped_len={final_stats['skipped_len']} "
        f"skipped_err={final_stats['skipped_err']} "
        f"elapsed={elapsed:.1f}s output_path={args.output}"
    )


if __name__ == "__main__":
    main()
