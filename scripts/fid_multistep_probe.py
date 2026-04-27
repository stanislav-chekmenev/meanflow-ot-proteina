# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Multi-step FM-Euler FID diagnostic.

Loads a Proteina checkpoint, generates the same pool of samples with
(a) the training-time 1-step MeanFlow sampler (``generate``) and
(b) the multi-step FM-Euler ODE sampler (``generate_fm_euler`` with h=0),
encodes both via GearNet-CA, and computes FID against the same reference
features the training callback uses (``pdb_eval_ca_features.pth`` by default).

Reuses the exact pipeline from proteinfoundation/callbacks/fid_callback.py:
    generate(..)  ->  samples_to_atom37(..)  ->  _build_pyg_batch(..)
                  ->  metric_factory.update(real=False)  ->  compute().

Interpretation
--------------
If the multi-step FID is drastically lower than 1-step (e.g. 1-step >= 10K
but multi-step < 1K), the 1-step MeanFlow generator has not converged to a
good average velocity and that is what drives the huge training-time FIDs.
The learned velocity field itself is OK; only the h != 0 MeanFlow prediction
is off.

If both are similarly high, the model's velocity field is genuinely off
the data manifold and no amount of integration steps will save it. Look
upstream (training loss saturation, OT coupling, etc.).

Usage
-----
    source .venv/bin/activate
    PYTHONPATH=. python scripts/fid_multistep_probe.py \\
        --ckpt ./store/meanflow-base-0.5-monomers-maxl256/checkpoints/last.ckpt \\
        --n-samples 1000 \\
        --nsteps-multistep 200 \\
        --generation-batch-size 4

Runs on 1 GPU. For n_samples=1000 x nsteps=200 expect ~5-10 min of wall
time on a single 80 GB GPU. Scale n_samples up if you want tighter FID.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import List

import torch
from loguru import logger
from torch_geometric.data import Batch, Data

from proteinfoundation.metrics.metric_factory import GenerationMetricFactory
from proteinfoundation.proteinflow.proteina import Proteina

DEFAULT_CKPT = "./store/meanflow-base-0.5-monomers-maxl256/checkpoints/last.ckpt"
DEFAULT_GEARNET_CKPT = "./data/metric_factory/model_weights/gearnet_ca.pth"
DEFAULT_REAL_FEATURES = "./data/metric_factory/features/pdb_eval_ca_features.pth"
DEFAULT_LENGTHS: List[int] = [64, 100, 128, 150, 192, 200, 250, 256]


def _configure_logging() -> None:
    """Loguru sinks: INFO+ to stdout, WARNING+ to stderr (so SLURM .err captures
    warnings and tracebacks, while stdout carries normal progress)."""
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stdout, level="INFO", format=fmt, enqueue=False)
    logger.add(sys.stderr, level="WARNING", format=fmt, enqueue=False, backtrace=True, diagnose=False)


def _build_pyg_batch(atom37_batch: torch.Tensor) -> Batch:
    """Mirror of FIDCallback._build_pyg_batch (CA-only mask, openfold index 1)."""
    B, n_res = atom37_batch.shape[0], atom37_batch.shape[1]
    mask = torch.zeros((n_res, 37), dtype=torch.bool, device=atom37_batch.device)
    mask[:, 1] = True  # CA

    data_list = []
    for b in range(B):
        data_list.append(
            Data(
                coords=atom37_batch[b],
                coord_mask=mask.clone(),
                node_id=torch.arange(n_res, device=atom37_batch.device).unsqueeze(-1),
            )
        )
    return Batch.from_data_list(data_list)


def _distribute_lengths(n_samples: int, lengths: List[int]) -> List[int]:
    """Round-robin assignment of lengths to sample slots (matches FIDCallback)."""
    return [lengths[i % len(lengths)] for i in range(n_samples)]


def _generate_and_update_metric(
    model: Proteina,
    metric_factory: GenerationMetricFactory,
    mode: str,
    n_samples: int,
    lengths: List[int],
    nsteps: int,
    generation_batch_size: int,
    device: torch.device,
) -> float:
    """Generate n_samples proteins with the specified sampler, push them
    through GearNet via metric_factory, and return FID.

    Args:
        mode: "meanflow_1step" uses ``model.generate(nsteps=nsteps)``;
              "fm_euler"       uses ``model.generate_fm_euler(nsteps=nsteps)``.
    """
    assert mode in {"meanflow_1step", "fm_euler"}, mode
    sampler = model.generate if mode == "meanflow_1step" else model.generate_fm_euler

    plan = _distribute_lengths(n_samples, lengths)
    length_counts: dict = {}
    for ln in plan:
        length_counts[ln] = length_counts.get(ln, 0) + 1

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    generated = 0

    with torch.no_grad():
        for length, count in length_counts.items():
            processed = 0
            while processed < count:
                chunk = min(generation_batch_size, count - processed)
                samples = sampler(nsamples=chunk, n=length, nsteps=nsteps)  # [chunk, length, 3] nm
                atom37 = model.samples_to_atom37(samples)                   # [chunk, length, 37, 3] Å
                pyg_batch = _build_pyg_batch(atom37).to(device)
                metric_factory.update(pyg_batch, real=False)
                processed += chunk
                generated += chunk
                if generated % max(1, (n_samples // 20)) == 0:
                    elapsed = time.time() - t0
                    rate = generated / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"[{mode}] generated={generated}/{n_samples} "
                        f"rate={rate:.2f} samples/s  len={length}"
                    )

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    elapsed = time.time() - t0
    logger.info(
        f"[{mode}] done: n_samples={generated} nsteps={nsteps} "
        f"elapsed={elapsed:.1f}s peak_vram={peak_gb:.2f}GB"
    )

    results = metric_factory.compute()
    fid = results["FID"].item()
    metric_factory.reset()  # clears fake-side accumulators; reset_real_features=False keeps real
    return fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help="Path to the .ckpt to load (non-EMA).")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Samples per sampler. 5000 matches training cfg; 1000 is cheaper.")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                        help="Residue lengths to sample round-robin, matches training cfg.")
    parser.add_argument("--nsteps-meanflow", type=int, default=1,
                        help="Integration steps for the MeanFlow sampler (training cfg uses 1).")
    parser.add_argument("--nsteps-multistep", type=int, default=200,
                        help="Integration steps for the FM-Euler sampler (diagnostic, 100-200).")
    parser.add_argument("--generation-batch-size", type=int, default=4,
                        help="Batch size per generate() call. Same knob as fid.generation_batch_size.")
    parser.add_argument("--gearnet-ckpt", default=DEFAULT_GEARNET_CKPT)
    parser.add_argument("--real-features", default=DEFAULT_REAL_FEATURES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-meanflow", action="store_true",
                        help="Skip 1-step MeanFlow and only run FM-Euler.")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Need a GPU."
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    logger.info(f"[multistep-probe] Loading model from {args.ckpt}")
    model = Proteina.load_from_checkpoint(args.ckpt, map_location=device)
    model = model.to(device)
    model.eval()
    logger.info(f"[multistep-probe] Model loaded. nparams={sum(p.numel() for p in model.parameters()):,}")

    logger.info(
        f"[multistep-probe] Building metric factory "
        f"gearnet={args.gearnet_ckpt} real={args.real_features}"
    )
    metric_factory = GenerationMetricFactory(
        metrics=["FID"],
        ckpt_path=args.gearnet_ckpt,
        ca_only=True,
        reset_real_features=False,
        real_features_path=args.real_features,
    ).to(device)

    results = {}

    if not args.skip_meanflow:
        logger.info("=== 1-step MeanFlow (matches training-time FID callback) ===")
        results["meanflow_1step"] = _generate_and_update_metric(
            model=model,
            metric_factory=metric_factory,
            mode="meanflow_1step",
            n_samples=args.n_samples,
            lengths=args.lengths,
            nsteps=args.nsteps_meanflow,
            generation_batch_size=args.generation_batch_size,
            device=device,
        )

    logger.info(f"=== Multi-step FM-Euler (nsteps={args.nsteps_multistep}) ===")
    results["fm_euler"] = _generate_and_update_metric(
        model=model,
        metric_factory=metric_factory,
        mode="fm_euler",
        n_samples=args.n_samples,
        lengths=args.lengths,
        nsteps=args.nsteps_multistep,
        generation_batch_size=args.generation_batch_size,
        device=device,
    )

    logger.info("[multistep-probe] Summary")
    logger.info("-" * 60)
    for mode, fid in results.items():
        logger.info(f"  FID({mode}) = {fid:.4f}")
    logger.info("-" * 60)

    if "meanflow_1step" in results:
        ratio = results["meanflow_1step"] / max(results["fm_euler"], 1e-9)
        logger.info(f"  ratio  FID(1-step) / FID(multi-step) = {ratio:.2f}x")
        logger.info(
            "Interpretation: "
            "ratio >> 1 (e.g. >10x) -> 1-step MeanFlow has not converged to a "
            "good average velocity; velocity field itself may be fine. "
            "Next: look at meanflow_nsteps_sample, P_mean/P_std(r), loss weighting. "
            "ratio ~= 1 -> model is genuinely off-manifold regardless of sampler. "
            "Next: examine training dynamics (OT coupling, loss saturation, data pipeline)."
        )


if __name__ == "__main__":
    _configure_logging()
    try:
        main()
    except Exception:
        logger.error(
            "[multistep-probe] Unhandled exception:\n{}", traceback.format_exc()
        )
        sys.exit(1)
