"""FID floor probe: feed training-subset PDBs as "fake" and compute FID against the eval reference.

If the resulting FID is close to the values we observe during training (10K-150K),
the floor is a train-vs-eval distribution gap, not a generator quality issue —
and the fix is to dump a train-subset reference via
`GenerationMetricFactory.dump_real_dataset_features` and point
`fid.real_features_path` at that instead.

If it is near zero (paper-scale, <100), the reference and train distributions
are comparable, and the large FIDs observed during training are truly generator
quality. In that case the next diagnostic is multi-step FM Euler generation
via `Proteina.generate_fm_euler`.

Usage
-----
    .venv/bin/activate
    PYTHONPATH=. python scripts/fid_floor_probe.py \
        --train-dir /netscratch/schekmenev/ot_mf_proteina_ds/processed \
        --n-samples 5000 \
        --batch-size 16

Runs on 1 GPU.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time

import torch
from torch_geometric.data import Batch, Data

from proteinfoundation.metrics.metric_factory import GenerationMetricFactory

DEFAULT_TRAIN_DIR = "/netscratch/schekmenev/ot_mf_proteina_ds/processed"
DEFAULT_GEARNET_CKPT = "./data/metric_factory/model_weights/gearnet_ca.pth"
DEFAULT_REAL_FEATURES = "./data/metric_factory/features/pdb_eval_ca_features.pth"


def _load_data(pt_path: str) -> Data:
    """Load a pre-processed .pt file and return a minimal PyG Data with the
    attributes GearNet-CA needs: coords, coord_mask, node_id.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR,
                        help="Directory of pre-processed .pt files (one per protein).")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of training structures to push through GearNet as 'fake'.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="PyG batch size for GearNet inference.")
    parser.add_argument("--gearnet-ckpt", default=DEFAULT_GEARNET_CKPT)
    parser.add_argument("--real-features", default=DEFAULT_REAL_FEATURES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=None,
                        help="If set, skip structures longer than this (e.g. 256 to match training filter).")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Need a GPU for GearNet inference."

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Gather .pt files
    pt_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pt")))
    print(f"[floor-probe] Found {len(pt_files)} .pt files in {args.train_dir}")
    assert pt_files, f"No .pt files under {args.train_dir}"

    random.shuffle(pt_files)

    # Build metric factory (CA-only, loads the same real-features the training callback uses)
    print(f"[floor-probe] Loading GenerationMetricFactory "
          f"ckpt={args.gearnet_ckpt} real={args.real_features}")
    mf = GenerationMetricFactory(
        metrics=["FID"],
        ckpt_path=args.gearnet_ckpt,
        ca_only=True,
        reset_real_features=False,
        real_features_path=args.real_features,
    ).cuda()

    # Iterate training structures, push through metric_factory.update(..., real=False)
    device = torch.device("cuda")
    processed = 0
    skipped_len = 0
    skipped_err = 0
    batch_buf: list[Data] = []
    start = time.time()

    for pt_path in pt_files:
        if processed >= args.n_samples:
            break
        try:
            d = _load_data(pt_path)
        except Exception as e:
            skipped_err += 1
            continue
        if args.max_len is not None and d.coords.shape[0] > args.max_len:
            skipped_len += 1
            continue
        batch_buf.append(d)
        if len(batch_buf) == args.batch_size:
            pyg_batch = Batch.from_data_list(batch_buf).to(device)
            mf.update(pyg_batch, real=False)
            processed += len(batch_buf)
            batch_buf = []
            if processed % 500 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[floor-probe] processed={processed} "
                      f"skipped_len={skipped_len} skipped_err={skipped_err} "
                      f"rate={rate:.1f} struct/s")

    # Flush remainder
    if batch_buf and processed < args.n_samples:
        pyg_batch = Batch.from_data_list(batch_buf).to(device)
        mf.update(pyg_batch, real=False)
        processed += len(batch_buf)

    elapsed = time.time() - start
    print(f"[floor-probe] Done. processed={processed} "
          f"skipped_len={skipped_len} skipped_err={skipped_err} "
          f"elapsed={elapsed:.1f}s")

    if processed < 2:
        raise RuntimeError("Need at least 2 structures to compute FID.")

    # Compute FID
    results = mf.compute()
    fid_value = results["FID"].item()
    print(f"[floor-probe] FID(train_subset -> pdb_eval_ref) = {fid_value:.4f} "
          f"over {processed} train structures")
    print(
        "\nInterpretation:\n"
        "  FID in the 5K-30K range -> training FIDs are dominated by the\n"
        "    train/eval distribution gap, not generator quality. Action: dump\n"
        "    a train-subset reference via GenerationMetricFactory.dump_real_dataset_features\n"
        "    and point fid.real_features_path at it.\n"
        "  FID < 100 -> reference and train distribution are comparable; the large\n"
        "    training-time FIDs are genuine generator quality. Next diagnostic:\n"
        "    multi-step FM Euler via Proteina.generate_fm_euler(nsteps=100..200).\n"
    )


if __name__ == "__main__":
    main()
