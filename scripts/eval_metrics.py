"""Evaluate a Proteina-style checkpoint with the standard sample-quality
metrics: designability, diversity (TM-score and cluster), and FID.

Pipeline
--------
1. Load a Proteina checkpoint and configure single-step inference defaults
   (sampling_caflow.sampling_mode='sc', sc_scale_noise=0.4, dt=0.0025) — same
   shape as `configs/experiment_config/inference_base.yaml` +
   `inference_ucond_200m_notri.yaml`.
2. Sample N proteins per length (default lengths=[50,100,150,200,250],
   N=100) using `lightning.Trainer.predict`. Each prediction is a
   `[b, n, 37, 3]` atom37 tensor; we write each entry to a `.pdb` via
   `proteinfoundation.utils.ff_utils.pdb_utils.write_prot_to_pdb`.
3. Designability: `% samples whose min(scRMSD over 8 ProteinMPNN seqs) < 2 A`,
   using `proteinfoundation.metrics.designability.scRMSD(..., ret_min=False)`.
4. Diversity (TM-score): pairwise TM-align over **designable** samples,
   averaged per-length, then averaged across lengths. We shell out to
   `TMalign` and parse the chain_1-normalized score.
5. Diversity (cluster): foldseek easy-cluster with the paper-§F.1 flags;
   ratio = #clusters / #designable.
6. FID via `GenerationMetricFactory.generation_metric_from_list` against
   the precomputed real-features `.pth`.
7. Novelty: stub. Logs a placeholder line and returns None. TODO: implement
   via `foldseek easy-search` against PDB100 (paper §F.1).

CLI
---
See `--help`. Key flags:
  --ckpt-path  required
  --num-samples-per-length, --lengths
  --output-dir
  --gearnet-ckpt, --fid-real-features
  --foldseek-bin, --tmalign-bin
  --quick-test  (smoke: lengths=[50,100], 4 samples each)
  --skip-designability / --skip-fid / --skip-tm-diversity / --skip-cluster-diversity

The pure-Python helpers (`tm_score_ca`, `aggregate_pairwise_diversity`,
`parse_foldseek_cluster_tsv`, `cluster_diversity_from_dir`,
`designability_rate`, `parse_tmalign_output`, `compute_novelty`) are
unit-tested in `tests/scripts/test_eval_metrics.py`. Integration pieces
(model loading, sampling, ESMFold, GearNet, foldseek/TMalign subprocess)
are exercised end-to-end via `scripts/eval_metrics.sbatch`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import torch
from loguru import logger

# Loguru -> stdlib logging bridge so pytest's caplog can see logger.warning(...).
def _add_loguru_to_stdlib_propagation() -> None:
    """Add a loguru sink that re-emits every record on stdlib logging so that
    pytest's caplog (and any external `logging` consumer) sees our messages.
    Idempotent — adding the same sink twice is harmless (loguru allocates a
    fresh id each time, but our public callers only invoke this once)."""
    py_logger = logging.getLogger("eval_metrics")
    py_logger.setLevel(logging.DEBUG)

    def _sink(message):
        record = message.record
        # Map loguru levels to stdlib numeric levels.
        levelname = record["level"].name
        levelno = getattr(logging, levelname, logging.INFO)
        py_logger.log(levelno, record["message"])

    logger.add(_sink, level="DEBUG")


_add_loguru_to_stdlib_propagation()


# ---------------------------------------------------------------------------
# Pure-Python helpers (unit-tested)
# ---------------------------------------------------------------------------


def _kabsch_align_ca(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Align `a` onto `b` (both `[L, 3]`) via Kabsch. Returns aligned `a`.

    Pure-python so this module stays import-cheap. Mirrors the math in
    `proteinfoundation.utils.align_utils.align_utils.kabsch_align`.
    """
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    H = a_c.T @ b_c
    U, _S, Vt = torch.linalg.svd(H)
    R = (Vt.T @ U.T)
    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1] = -Vt[-1]
        R = (Vt.T @ U.T)
    return (a_c @ R.T) + b.mean(dim=0, keepdim=True)


def tm_score_ca(coors_a: torch.Tensor, coors_b: torch.Tensor) -> float:
    """TM-score between two CA point clouds of shape `[L, 3]`.

    This is a pure-python *Kabsch + TM-score* implementation: we do a single
    rigid alignment via Kabsch (no rotational search), then evaluate the
    standard TM-score functional with `d_0 = 1.24 * (L-15)^(1/3) - 1.8` for
    L>=19 (with a 0.5 floor for shorter chains, matching the published
    short-chain convention).

    For the actual diversity reported in `results.json` the main flow shells
    out to TMalign — see `tm_align_pair` — which runs the full rotational
    search. This helper exists for cheap unit tests and as a sanity-check
    fallback when the bundled TMalign binary is missing.

    Args:
        coors_a: `[L, 3]` torch tensor of CA coordinates.
        coors_b: `[L, 3]` torch tensor of CA coordinates.

    Returns:
        TM-score in `[0, 1]`. 1.0 for identical / rigid-transformed inputs.

    Raises:
        ValueError: if the two inputs have mismatched lengths.
    """
    if coors_a.shape != coors_b.shape:
        raise ValueError(
            f"tm_score_ca: shape mismatch {tuple(coors_a.shape)} vs {tuple(coors_b.shape)}"
        )
    if coors_a.ndim != 2 or coors_a.shape[-1] != 3:
        raise ValueError(
            f"tm_score_ca: expected [L, 3] tensors, got {tuple(coors_a.shape)}"
        )
    L = coors_a.shape[0]
    if L == 0:
        raise ValueError("tm_score_ca: empty coordinates")

    # d_0 normalization. The classical formula is undefined for L < 19; we
    # use a 0.5 floor in that regime, which matches the published short-chain
    # convention used by tools like TM-align.
    if L >= 19:
        d0 = 1.24 * ((L - 15) ** (1.0 / 3.0)) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)

    aligned = _kabsch_align_ca(coors_a, coors_b)
    diffs = aligned - coors_b.to(torch.float64)
    d = torch.linalg.vector_norm(diffs, dim=-1)
    score = (1.0 / (1.0 + (d / d0) ** 2)).mean().item()
    return float(score)


def aggregate_pairwise_diversity(
    by_length: Dict[int, Sequence[torch.Tensor]],
    *,
    tm_score_fn: Callable[[torch.Tensor, torch.Tensor], float] = tm_score_ca,
) -> Dict[str, object]:
    """Average pairwise TM-score among designable samples.

    Per the Proteina paper §F: per-length mean of all pairwise TM-scores,
    then average across lengths.

    Args:
        by_length: dict `{L: [coors, ...]}` of designable CA tensors per length.
            Each tensor is `[L, 3]`. Lengths with <2 samples are skipped (no
            pairs).
        tm_score_fn: function that consumes two CA tensors and returns a
            scalar TM-score. Pluggable so the main flow can pass a TMalign
            shell-out wrapper while the unit tests pass a constant fake.

    Returns:
        ``{"per_length_mean_tm": {L: mean}, "pairs_per_length": {L: count},
           "mean_pairwise_tm": float | None}``.
    """
    per_length_mean: Dict[int, float] = {}
    pairs_per_length: Dict[int, int] = {}
    for L, samples in by_length.items():
        if len(samples) < 2:
            continue
        scores: List[float] = []
        for a, b in combinations(samples, 2):
            scores.append(float(tm_score_fn(a, b)))
        per_length_mean[L] = sum(scores) / len(scores)
        pairs_per_length[L] = len(scores)

    if not per_length_mean:
        mean_pairwise = None
    else:
        mean_pairwise = sum(per_length_mean.values()) / len(per_length_mean)
    return {
        "per_length_mean_tm": per_length_mean,
        "pairs_per_length": pairs_per_length,
        "mean_pairwise_tm": mean_pairwise,
    }


def parse_foldseek_cluster_tsv(tsv_path: str) -> int:
    """Count unique cluster representatives in a foldseek `*_cluster.tsv`.

    Foldseek `easy-cluster` writes a TSV with columns `cluster_rep \\t member`.
    Number of clusters equals the number of distinct `cluster_rep` entries.

    Args:
        tsv_path: path to `<prefix>_cluster.tsv`.

    Returns:
        The number of unique cluster representatives.
    """
    reps = set()
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if not parts or not parts[0].strip():
                continue
            reps.add(parts[0].strip())
    return len(reps)


def cluster_diversity_from_dir(
    samples_dir: str,
    *,
    foldseek_bin: str,
    tmscore_threshold: float = 0.5,
    cov_mode: int = 0,
    min_seq_id: float = 0.0,
    alignment_type: int = 1,
    tmp_root: Optional[str] = None,
) -> Dict[str, object]:
    """Run foldseek easy-cluster on a directory of designable PDBs.

    Replicates the paper §F.1 invocation:
        foldseek easy-cluster <samples_dir> <tmp>/res <tmp> \\
            --alignment-type 1 --cov-mode 0 --min-seq-id 0 \\
            --tmscore-threshold 0.5

    Args:
        samples_dir: directory of `*.pdb` files (designable samples).
        foldseek_bin: absolute path to the bundled foldseek binary.
        tmscore_threshold: foldseek `--tmscore-threshold` (default 0.5 per
            paper).
        cov_mode: foldseek `--cov-mode`.
        min_seq_id: foldseek `--min-seq-id`.
        alignment_type: foldseek `--alignment-type`.
        tmp_root: optional parent dir for the foldseek scratch dir.

    Returns:
        ``{"n_clusters": int, "n_designable": int,
           "cluster_diversity": float | None}``. `cluster_diversity` is
        None when there are no designable samples.
    """
    pdb_paths = sorted(glob.glob(os.path.join(samples_dir, "*.pdb")))
    n_designable = len(pdb_paths)
    if n_designable == 0:
        return {
            "n_clusters": 0,
            "n_designable": 0,
            "cluster_diversity": None,
        }

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        prefix_res = os.path.join(tmp, "res")
        cmd = [
            foldseek_bin,
            "easy-cluster",
            samples_dir,
            prefix_res,
            tmp,
            "--alignment-type",
            str(alignment_type),
            "--cov-mode",
            str(cov_mode),
            "--min-seq-id",
            str(min_seq_id),
            "--tmscore-threshold",
            str(tmscore_threshold),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        cluster_tsv = prefix_res + "_cluster.tsv"
        n_clusters = parse_foldseek_cluster_tsv(cluster_tsv)

    diversity = n_clusters / n_designable
    return {
        "n_clusters": n_clusters,
        "n_designable": n_designable,
        "cluster_diversity": diversity,
    }


def designability_rate(
    sample_rmsds: Sequence[Sequence[float]],
    *,
    threshold: float = 2.0,
) -> Dict[str, object]:
    """Designability rate: percentage of samples whose minimum scRMSD over
    the ProteinMPNN sequences is **strictly** less than `threshold` Å.

    Args:
        sample_rmsds: per-sample list of scRMSDs (one entry per ProteinMPNN
            sequence, typically 8). Order is preserved.
        threshold: scRMSD cutoff in Å. Standard value is 2.0.

    Returns:
        ``{"n_total": int, "n_designable": int, "designability": float | None,
           "designable_indices": [int, ...]}``.
    """
    n_total = len(sample_rmsds)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_designable": 0,
            "designability": None,
            "designable_indices": [],
        }
    designable_indices: List[int] = []
    for i, rmsds in enumerate(sample_rmsds):
        if not rmsds:
            continue
        if min(rmsds) < threshold:
            designable_indices.append(i)
    return {
        "n_total": n_total,
        "n_designable": len(designable_indices),
        "designability": len(designable_indices) / n_total,
        "designable_indices": designable_indices,
    }


def parse_tmalign_output(text: str, *, normalize_by: str = "chain1") -> float:
    """Parse a TM-align stdout block and return the requested TM-score.

    TMalign emits two ``TM-score=`` lines:
        TM-score= 0.7234 (if normalized by length of Chain_1, ...)
        TM-score= 0.7100 (if normalized by length of Chain_2, ...)

    For protein diversity the convention is to compare same-length pairs and
    use the chain_1 normalization (default).

    Args:
        text: TMalign stdout.
        normalize_by: ``"chain1"`` (default) or ``"chain2"``.

    Returns:
        The requested TM-score as a float.

    Raises:
        ValueError: if the requested line cannot be located, or if
            `normalize_by` is not one of {"chain1", "chain2"}.
    """
    if normalize_by not in {"chain1", "chain2"}:
        raise ValueError(
            f"parse_tmalign_output: normalize_by must be 'chain1' or 'chain2', "
            f"got {normalize_by!r}"
        )
    needle = "Chain_1" if normalize_by == "chain1" else "Chain_2"
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("TM-score="):
            continue
        if needle not in line:
            continue
        # Layout: "TM-score= 0.72340 (if normalized by length of Chain_1, ...)"
        try:
            after_eq = line.split("=", 1)[1].strip()
            number = after_eq.split()[0]
            return float(number)
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"parse_tmalign_output: could not parse TM-score from line {line!r}"
            ) from e
    raise ValueError(
        f"parse_tmalign_output: no TM-score line for {needle} in TMalign output"
    )


def compute_novelty(
    designable_pdbs: Optional[Sequence[str]] = None,
    **_kwargs,
) -> None:
    """Novelty placeholder.

    The Proteina paper §F.1 defines novelty as
        ``mean over designable samples of [max TM-score against the train dataset via
          foldseek easy-search, alignment-type 1, --tmscore-threshold 0.0]``
    i.e. for each designable sample find the closest train_ds hit and average
    the TM-scores.

    TODO(eval-novelty): implement via
        foldseek easy-search <designable_dir> <train_ds> <out>.m8 <tmp> \\
            --alignment-type 1 --tmscore-threshold 0.0 --format-output \\
            "query,target,alntmscore,..."
    Then per-query take max(alntmscore), aggregate to a mean. Skipped here
    because we do not currently have a staged train_ds foldseek index.

    Always logs a single warning-level line so that the gap is visible in
    every run.
    """
    n = len(designable_pdbs) if designable_pdbs is not None else 0
    logger.warning(
        f"Novelty metric not yet implemented (placeholder). "
        f"Would have evaluated {n} designable PDBs against the train dataset. "
        f"See compute_novelty() docstring for the foldseek easy-search recipe."
    )
    return None


# ---------------------------------------------------------------------------
# Integration helpers (NOT unit-tested; exercised by the sbatch end-to-end)
# ---------------------------------------------------------------------------


_TMALIGN_BUNDLE_HINT = (
    "TMalign binary not found at {bin_path!r}. Bundle it once via:\n"
    "    mkdir -p {parent}\n"
    "    curl -fsSL https://zhanggroup.org/TM-align/TMalign_cpp.gz -o /tmp/TMalign_cpp.gz\n"
    "    gunzip -c /tmp/TMalign_cpp.gz > {bin_path}\n"
    "    chmod +x {bin_path}\n"
)


def assert_tmalign_bin_available(bin_path: str) -> None:
    if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
        raise FileNotFoundError(
            _TMALIGN_BUNDLE_HINT.format(
                bin_path=bin_path, parent=os.path.dirname(bin_path) or "."
            )
        )


def tm_align_pair(pdb_a: str, pdb_b: str, *, tmalign_bin: str) -> float:
    """Run TMalign on two PDB files; return chain_1-normalized TM-score."""
    proc = subprocess.run(
        [tmalign_bin, pdb_a, pdb_b],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_tmalign_output(proc.stdout, normalize_by="chain1")


# ---------------------------------------------------------------------------
# Sampling + write PDB
# ---------------------------------------------------------------------------


def _build_inference_cfg(dt: float, nsteps: int, sc_scale_noise: float = 0.4):
    """Build a minimal OmegaConf-style cfg with the keys
    `proteina.predict_step` / `configure_inference` look up.

    `nsteps` overrides `Proteina.meanflow_nsteps_sample` (which is loaded
    from the checkpoint's training config and defaults to 1). Setting it
    to >1 switches to MeanFlow multi-step generation.
    """
    from omegaconf import OmegaConf

    cfg_dict = {
        "nsteps": nsteps,
        "self_cond": True,
        "fold_cond": False,
        "dt": dt,
        "sampling_caflow": {
            "sampling_mode": "sc",
            "sc_scale_noise": sc_scale_noise,
            "sc_scale_score": 1.0,
            "gt_mode": "1/t",
            "gt_p": 1.0,
            "gt_clamp_val": None,
        },
        "schedule": {
            "schedule_mode": "log",
            "schedule_p": 2.0,
        },
        "guidance_weight": 1.0,
        "autoguidance_ratio": 0.0,
        "autoguidance_ckpt_path": None,
    }
    return OmegaConf.create(cfg_dict)


def _generate_samples(
    *,
    ckpt_path: str,
    lengths: Sequence[int],
    n_per_length: int,
    output_pdb_dir: str,
    dt: float,
    nsteps: int,
    sc_scale_noise: float,
    seed: int,
    max_nsamples_per_batch: int = 8,
) -> Dict[int, List[str]]:
    """Sample `n_per_length` proteins for each L in `lengths`, write each as
    a PDB into `output_pdb_dir/L_{L}/sample_{i}.pdb`, and return the per-length
    list of PDB paths.

    Each `predict_step` generates `nsamples` proteins in one batched forward
    pass, so we chunk `n_per_length` into pieces of `max_nsamples_per_batch`
    rather than running n_per_length separate forward passes.

    `nsteps` controls MeanFlow integration: 1 = single-step (default trained
    behavior), >1 = multi-step Euler-style integration via
    `Proteina.generate -> fm.meanflow_sample`.
    """
    import lightning as L_pl
    from torch.utils.data import DataLoader

    # Imported here so the test module (which imports this script) doesn't pay
    # the cost at collection time.
    from proteinfoundation.inference import GenDataset
    from proteinfoundation.proteinflow.proteina import Proteina
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

    L_pl.seed_everything(seed)

    logger.info(f"Loading Proteina checkpoint: {ckpt_path}")
    model = Proteina.load_from_checkpoint(ckpt_path)

    inf_cfg = _build_inference_cfg(
        dt=dt, nsteps=nsteps, sc_scale_noise=sc_scale_noise
    )
    model.configure_inference(inf_cfg, nn_ag=None)
    logger.info(
        f"MeanFlow generation: nsteps={nsteps} dt={dt} sc_scale_noise={sc_scale_noise}"
    )

    nres_list: List[int] = []
    nsamples_list: List[int] = []
    for L in lengths:
        remaining = n_per_length
        while remaining > 0:
            chunk = min(remaining, max_nsamples_per_batch)
            nres_list.append(int(L))
            nsamples_list.append(chunk)
            remaining -= chunk

    dataset = GenDataset(nres=nres_list, nsamples=nsamples_list, dt=dt)
    dataloader = DataLoader(dataset, batch_size=1)

    trainer = L_pl.Trainer(accelerator="gpu", devices=1, logger=False)
    predictions = trainer.predict(model, dataloader)

    by_length: Dict[int, List[str]] = {int(L): [] for L in lengths}
    for L in lengths:
        os.makedirs(os.path.join(output_pdb_dir, f"L_{L}"), exist_ok=True)

    counter: Dict[int, int] = {int(L): 0 for L in lengths}
    for pred in predictions:
        coors_atom37 = pred  # [b, n, 37, 3]
        n = int(coors_atom37.shape[-3])
        for i in range(coors_atom37.shape[0]):
            idx = counter[n]
            pdb_path = os.path.join(
                output_pdb_dir, f"L_{n}", f"sample_{idx:04d}.pdb"
            )
            write_prot_to_pdb(
                coors_atom37[i].cpu().numpy(),
                pdb_path,
                overwrite=True,
                no_indexing=True,
            )
            by_length[n].append(pdb_path)
            counter[n] = idx + 1

    for L, paths in by_length.items():
        logger.info(f"  L={L}: wrote {len(paths)} PDBs to {output_pdb_dir}/L_{L}/")
    return by_length


# ---------------------------------------------------------------------------
# Designability driver (calls scRMSD per PDB)
# ---------------------------------------------------------------------------


def _run_designability(
    by_length: Dict[int, List[str]],
    *,
    designability_tmp_root: str,
    threshold: float = 2.0,
) -> Dict[str, object]:
    """Run scRMSD on every sample (8 ProteinMPNN seqs each), collate."""
    from proteinfoundation.metrics.designability import scRMSD

    os.makedirs(designability_tmp_root, exist_ok=True)
    all_rmsds: List[List[float]] = []
    flat_paths: List[str] = []
    by_length_indices: Dict[int, List[int]] = {L: [] for L in by_length}

    flat_idx = 0
    for L, pdb_paths in by_length.items():
        logger.info(f"Designability for L={L}: {len(pdb_paths)} samples")
        for pdb_path in pdb_paths:
            tmp_for_sample = os.path.join(
                designability_tmp_root,
                f"L_{L}",
                os.path.splitext(os.path.basename(pdb_path))[0],
            )
            os.makedirs(tmp_for_sample, exist_ok=True)
            try:
                rmsds = scRMSD(pdb_path, ret_min=False, tmp_path=tmp_for_sample)
            except Exception as e:
                logger.warning(
                    f"  scRMSD failed on {pdb_path}: {e}; treating as non-designable"
                )
                rmsds = [float("inf")] * 8
            all_rmsds.append(list(rmsds))
            flat_paths.append(pdb_path)
            by_length_indices[L].append(flat_idx)
            flat_idx += 1

    summary = designability_rate(all_rmsds, threshold=threshold)
    summary["pdb_paths"] = flat_paths
    summary["by_length_flat_indices"] = by_length_indices
    summary["per_sample_rmsds"] = all_rmsds
    return summary


# ---------------------------------------------------------------------------
# FID driver
# ---------------------------------------------------------------------------


def _run_fid(
    pdb_paths: Sequence[str],
    *,
    gearnet_ckpt: str,
    real_features_path: str,
) -> float:
    """Run FID against the precomputed real-features `.pth`.

    Mirrors the `compute_fid` block of `proteinfoundation/inference.py`.
    """
    from proteinfoundation.metrics.metric_factory import (
        GenerationMetricFactory,
        generation_metric_from_list,
    )

    factory = GenerationMetricFactory(
        metrics=["FID"],
        ckpt_path=gearnet_ckpt,
        ca_only=True,
        reset_real_features=False,
        real_features_path=real_features_path,
    ).cuda()
    metrics = generation_metric_from_list(list(pdb_paths), factory)
    fid = float(metrics["FID"].cpu().item())
    return fid


# ---------------------------------------------------------------------------
# TM-diversity driver
# ---------------------------------------------------------------------------


def _run_tm_diversity(
    designable_by_length: Dict[int, List[str]],
    *,
    tmalign_bin: str,
) -> Dict[str, object]:
    """Run TMalign pairwise on the designable samples, per length."""
    assert_tmalign_bin_available(tmalign_bin)

    per_length_mean: Dict[int, float] = {}
    pairs_per_length: Dict[int, int] = {}
    for L, pdb_paths in designable_by_length.items():
        if len(pdb_paths) < 2:
            logger.info(f"  L={L}: <2 designable samples — skipping TM diversity")
            continue
        scores: List[float] = []
        for a, b in combinations(pdb_paths, 2):
            try:
                scores.append(tm_align_pair(a, b, tmalign_bin=tmalign_bin))
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"  TMalign failed on ({a}, {b}): exit={e.returncode}, "
                    f"stderr={e.stderr[:200]!r}"
                )
        if scores:
            per_length_mean[L] = sum(scores) / len(scores)
            pairs_per_length[L] = len(scores)
            logger.info(
                f"  L={L}: n_pairs={len(scores)} mean_pairwise_tm={per_length_mean[L]:.4f}"
            )

    if per_length_mean:
        mean_pairwise = sum(per_length_mean.values()) / len(per_length_mean)
    else:
        mean_pairwise = None
    return {
        "per_length_mean_tm": per_length_mean,
        "pairs_per_length": pairs_per_length,
        "mean_pairwise_tm": mean_pairwise,
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _stage_designable_pdbs(
    designable_pdbs: Sequence[str], dest_dir: str
) -> List[str]:
    """Hardlink (fallback: copy) each designable PDB into `dest_dir` so
    foldseek easy-cluster sees them as a single directory."""
    os.makedirs(dest_dir, exist_ok=True)
    staged: List[str] = []
    for src in designable_pdbs:
        dst = os.path.join(dest_dir, os.path.basename(src))
        # Disambiguate by parent dir if names collide across lengths.
        if os.path.exists(dst):
            parent = os.path.basename(os.path.dirname(src))
            dst = os.path.join(
                dest_dir, f"{parent}__{os.path.basename(src)}"
            )
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-path", required=True, help="Proteina .ckpt to evaluate")
    p.add_argument("--num-samples-per-length", type=int, default=100)
    p.add_argument(
        "--lengths",
        default="50,100,150,200,250",
        help="Comma-separated protein lengths to sample at.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: ./inference/eval_metrics_<timestamp>",
    )
    p.add_argument(
        "--gearnet-ckpt",
        default="data/metric_factory/model_weights/gearnet_ca.pth",
    )
    p.add_argument(
        "--fid-real-features",
        default=(
            "data/metric_factory/features/pdb_eval_ca_features_monomers_frac0.1.pth"
        ),
    )
    p.add_argument("--foldseek-bin", default="./foldseek/bin/foldseek")
    p.add_argument("--tmalign-bin", default="./tmalign/TMalign")
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Smoke test: lengths=[50,100], 4 samples each.",
    )
    p.add_argument("--skip-designability", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--skip-tm-diversity", action="store_true")
    p.add_argument("--skip-cluster-diversity", action="store_true")
    p.add_argument("--dt", type=float, default=0.0025)
    p.add_argument(
        "--nsteps",
        type=int,
        default=4,
        help=(
            "MeanFlow integration steps. 1 = single-step (model's trained "
            "default); >1 = multi-step Euler integration of the average "
            "velocity field. Overrides Proteina.meanflow_nsteps_sample at "
            "inference."
        ),
    )
    p.add_argument("--sc-scale-noise", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--designability-threshold", type=float, default=2.0)
    p.add_argument(
        "--max-nsamples-per-batch",
        type=int,
        default=8,
        help="Per-length samples are chunked into batches of this size for one model forward pass.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.quick_test:
        lengths = [50, 100]
        n_per_length = 4
        logger.warning("--quick-test active: lengths=[50,100], 4 samples each")
    else:
        lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
        n_per_length = args.num_samples_per_length

    if args.output_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("inference", f"eval_metrics_{ts}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pdb_dir = os.path.join(output_dir, "samples")
    designability_tmp_dir = os.path.join(output_dir, "designability_tmp")
    designable_dir = os.path.join(output_dir, "designable")

    logger.info("=" * 72)
    logger.info("Eval metrics run")
    logger.info(f"  ckpt:                {args.ckpt_path}")
    logger.info(f"  lengths:             {lengths}")
    logger.info(f"  samples/length:      {n_per_length}")
    logger.info(f"  nsteps (meanflow):   {args.nsteps}")
    logger.info(f"  dt:                  {args.dt}")
    logger.info(f"  sc_scale_noise:      {args.sc_scale_noise}")
    logger.info(f"  seed:                {args.seed}")
    logger.info(f"  output_dir:          {output_dir}")
    logger.info(f"  gearnet_ckpt:        {args.gearnet_ckpt}")
    logger.info(f"  fid_real_features:   {args.fid_real_features}")
    logger.info(f"  foldseek_bin:        {args.foldseek_bin}")
    logger.info(f"  tmalign_bin:         {args.tmalign_bin}")
    logger.info(f"  skip_designability:  {args.skip_designability}")
    logger.info(f"  skip_fid:            {args.skip_fid}")
    logger.info(f"  skip_tm_diversity:   {args.skip_tm_diversity}")
    logger.info(f"  skip_cluster_div:    {args.skip_cluster_diversity}")
    logger.info("=" * 72)

    # Step 1: sample.
    by_length = _generate_samples(
        ckpt_path=args.ckpt_path,
        lengths=lengths,
        n_per_length=n_per_length,
        output_pdb_dir=pdb_dir,
        dt=args.dt,
        nsteps=args.nsteps,
        sc_scale_noise=args.sc_scale_noise,
        seed=args.seed,
        max_nsamples_per_batch=args.max_nsamples_per_batch,
    )
    flat_pdb_paths = [p for paths in by_length.values() for p in paths]

    results: Dict[str, object] = {
        "config": {
            "ckpt_path": args.ckpt_path,
            "lengths": lengths,
            "num_samples_per_length": n_per_length,
            "nsteps": args.nsteps,
            "dt": args.dt,
            "sc_scale_noise": args.sc_scale_noise,
            "seed": args.seed,
            "designability_threshold": args.designability_threshold,
            "quick_test": args.quick_test,
        },
        "n_samples": len(flat_pdb_paths),
    }

    # Step 2: designability.
    designable_by_length: Dict[int, List[str]] = {L: [] for L in lengths}
    if args.skip_designability:
        logger.info("Skipping designability (--skip-designability)")
        # Without designability we can't know which subset is "designable" —
        # downstream diversity metrics will operate on ALL samples in that case.
        designable_by_length = {L: list(paths) for L, paths in by_length.items()}
    else:
        logger.info("=" * 72)
        logger.info("Computing designability")
        logger.info("=" * 72)
        d_summary = _run_designability(
            by_length,
            designability_tmp_root=designability_tmp_dir,
            threshold=args.designability_threshold,
        )
        # Build per-length list of designable PDB paths.
        flat_paths = d_summary["pdb_paths"]
        designable_set = set(d_summary["designable_indices"])
        for L, indices in d_summary["by_length_flat_indices"].items():
            designable_by_length[L] = [
                flat_paths[i] for i in indices if i in designable_set
            ]
        results["designability"] = {
            "n_total": d_summary["n_total"],
            "n_designable": d_summary["n_designable"],
            "designability": d_summary["designability"],
            "threshold": args.designability_threshold,
            "per_length_n_designable": {
                L: len(paths) for L, paths in designable_by_length.items()
            },
        }
        logger.info(
            f"Designability: {d_summary['n_designable']}/{d_summary['n_total']} "
            f"= {d_summary['designability']}"
        )

    designable_flat = [
        p for paths in designable_by_length.values() for p in paths
    ]

    # Step 3: TM diversity (TMalign pairwise on designables).
    if args.skip_tm_diversity:
        logger.info("Skipping TM diversity (--skip-tm-diversity)")
    else:
        logger.info("=" * 72)
        logger.info("Computing TM-score diversity (TMalign pairwise)")
        logger.info("=" * 72)
        tm_div = _run_tm_diversity(
            designable_by_length, tmalign_bin=args.tmalign_bin
        )
        results["tm_diversity"] = tm_div
        logger.info(f"TM diversity (lower is more diverse): {tm_div['mean_pairwise_tm']}")

    # Step 4: cluster diversity (foldseek easy-cluster on designables).
    if args.skip_cluster_diversity:
        logger.info("Skipping cluster diversity (--skip-cluster-diversity)")
    else:
        logger.info("=" * 72)
        logger.info("Computing cluster diversity (foldseek easy-cluster)")
        logger.info("=" * 72)
        if not designable_flat:
            logger.warning(
                "No designable samples — cluster diversity is undefined."
            )
            results["cluster_diversity"] = {
                "n_clusters": 0,
                "n_designable": 0,
                "cluster_diversity": None,
            }
        else:
            _stage_designable_pdbs(designable_flat, designable_dir)
            cd = cluster_diversity_from_dir(
                designable_dir, foldseek_bin=args.foldseek_bin
            )
            results["cluster_diversity"] = cd
            logger.info(
                f"Cluster diversity: {cd['cluster_diversity']} "
                f"({cd['n_clusters']} clusters of {cd['n_designable']} designables)"
            )

    # Step 5: FID over ALL samples (paper convention — FID is computed over
    # the full sampled set, not just designables).
    if args.skip_fid:
        logger.info("Skipping FID (--skip-fid)")
    else:
        logger.info("=" * 72)
        logger.info("Computing FID")
        logger.info("=" * 72)
        fid = _run_fid(
            flat_pdb_paths,
            gearnet_ckpt=args.gearnet_ckpt,
            real_features_path=args.fid_real_features,
        )
        results["fid"] = fid
        logger.info(f"FID: {fid:.4f}")

    # Step 6: novelty placeholder.
    logger.info("=" * 72)
    logger.info("Novelty (placeholder)")
    logger.info("=" * 72)
    novelty = compute_novelty(designable_pdbs=designable_flat)
    results["novelty"] = novelty

    # Persist + summary banner.
    out_json = os.path.join(output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Wrote results to {out_json}")

    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)
    if "designability" in results:
        d = results["designability"]
        logger.info(
            f"  Designability:    {d['n_designable']}/{d['n_total']} = {d['designability']}"
        )
    if "tm_diversity" in results:
        logger.info(
            f"  TM diversity:     mean_pairwise_tm={results['tm_diversity']['mean_pairwise_tm']}"
        )
    if "cluster_diversity" in results:
        c = results["cluster_diversity"]
        logger.info(
            f"  Cluster diversity: {c['cluster_diversity']} "
            f"({c['n_clusters']} clusters / {c['n_designable']} designables)"
        )
    if "fid" in results:
        logger.info(f"  FID:              {results['fid']:.4f}")
    logger.info("  Novelty:          placeholder (None)")
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
