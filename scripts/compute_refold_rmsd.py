"""Post-hoc per-length and total mean RMSD between generated structures and
their ESMFold-refolded counterparts.

Reads an `EVAL_OUTPUT_DIR` produced by `scripts/eval_metrics.py`:

    <EVAL_OUTPUT_DIR>/
        samples/L_<L>/sample_<NNNN>.pdb           # generated structures
        designability_tmp/L_<L>/sample_<NNNN>/
            esm_<k>.pdb_esm                       # ESMFold refold of MPNN seq k

For every (sample, refold) pair we compute the Kabsch-aligned CA RMSD via
`proteinfoundation.metrics.designability.rmsd_metric` — exactly the same
function `scRMSD` uses internally — and aggregate via the shared helper
`scripts.eval_metrics.aggregate_refold_rmsds`.

By default the script writes the result block back into
`<EVAL_OUTPUT_DIR>/results.json` under the key ``refold_rmsd`` (creating
the file if it does not exist; merging with the existing top-level dict
otherwise). Pass ``--no-update-results-json`` to skip that step.

Examples
--------
    python scripts/compute_refold_rmsd.py \\
        --eval-output-dir /netscratch/schekmenev/mf_proteina_inference/eval_metrics_44982

CLI
---
See ``--help``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import torch
from loguru import logger

# Ensure repo-root is importable when invoked as `python scripts/compute_refold_rmsd.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.eval_metrics import aggregate_refold_rmsds  # noqa: E402


_LENGTH_DIR_RE = re.compile(r"^L_(\d+)$")
_SAMPLE_FILE_RE = re.compile(r"^sample_(\d+)\.pdb$")
_SAMPLE_DIR_RE = re.compile(r"^sample_(\d+)$")


def discover_length_dirs(samples_root: str) -> Dict[int, str]:
    """Return ``{L: path-to-samples/L_<L>}`` for every L_<L> subdir found."""
    out: Dict[int, str] = {}
    if not os.path.isdir(samples_root):
        return out
    for name in sorted(os.listdir(samples_root)):
        m = _LENGTH_DIR_RE.match(name)
        if not m:
            continue
        full = os.path.join(samples_root, name)
        if os.path.isdir(full):
            out[int(m.group(1))] = full
    return out


def discover_sample_pdbs(length_dir: str) -> Dict[str, str]:
    """Return ``{sample_id: path-to-sample.pdb}`` inside `length_dir`.

    `sample_id` is the base name without the `.pdb` extension (e.g.
    ``sample_0007``) so we can match it to the sibling
    `designability_tmp/L_<L>/sample_0007/` directory.
    """
    out: Dict[str, str] = {}
    for name in sorted(os.listdir(length_dir)):
        m = _SAMPLE_FILE_RE.match(name)
        if not m:
            continue
        sample_id = name[:-4]  # strip `.pdb`
        out[sample_id] = os.path.join(length_dir, name)
    return out


def discover_refold_pdbs(sample_tmp_dir: str) -> List[str]:
    """Return sorted paths to `esm_*.pdb_esm` in `sample_tmp_dir`."""
    return sorted(glob.glob(os.path.join(sample_tmp_dir, "esm_*.pdb_esm")))


def _load_atom37(pdb_path: str) -> torch.Tensor:
    """Load a PDB and return the `[n, 37, 3]` atom positions tensor.

    Uses the same `from_pdb_string` / `Protein.atom_positions` path that
    `proteinfoundation.metrics.designability.scRMSD` uses.
    """
    from proteinfoundation.metrics.designability import load_pdb

    prot = load_pdb(pdb_path)
    return torch.as_tensor(prot.atom_positions, dtype=torch.float32)


def compute_pair_rmsd(gen_pdb: str, refold_pdb: str) -> float:
    """Kabsch-aligned CA RMSD between two PDB files (Å)."""
    from proteinfoundation.metrics.designability import rmsd_metric

    gen = _load_atom37(gen_pdb)
    rec = _load_atom37(refold_pdb)
    return float(rmsd_metric(gen, rec))


def collect_rmsds(
    eval_output_dir: str,
    *,
    samples_subdir: str = "samples",
    designability_tmp_subdir: str = "designability_tmp",
    expected_n_refolds: Optional[int] = None,
    strict: bool = False,
) -> Dict[int, List[List[float]]]:
    """Walk `eval_output_dir` and compute per-(sample, refold) RMSDs.

    Args:
        eval_output_dir: root produced by ``scripts/eval_metrics.py``.
        samples_subdir: name of the generated-PDB subdir (default ``samples``).
        designability_tmp_subdir: name of the refold subdir (default
            ``designability_tmp``).
        expected_n_refolds: if not None, log a warning when a sample has a
            different count.
        strict: if True, raise on missing refold dir / mismatched counts;
            otherwise log a warning and skip the sample.

    Returns:
        ``{L: [[r0, r1, ...], ...]}`` per-sample RMSD lists, suitable for
        ``aggregate_refold_rmsds``.
    """
    samples_root = os.path.join(eval_output_dir, samples_subdir)
    tmp_root = os.path.join(eval_output_dir, designability_tmp_subdir)

    length_dirs = discover_length_dirs(samples_root)
    if not length_dirs:
        raise FileNotFoundError(
            f"No L_<L> subdirs under {samples_root} — is this an eval output dir?"
        )

    rmsds_by_length: Dict[int, List[List[float]]] = {}
    for L in sorted(length_dirs):
        sample_paths = discover_sample_pdbs(length_dirs[L])
        per_length: List[List[float]] = []
        for sample_id, gen_pdb in sample_paths.items():
            sample_tmp = os.path.join(tmp_root, f"L_{L}", sample_id)
            if not os.path.isdir(sample_tmp):
                msg = (
                    f"L={L} {sample_id}: missing refold dir {sample_tmp} — "
                    f"skipping (no refolded structures available)"
                )
                if strict:
                    raise FileNotFoundError(msg)
                logger.warning(msg)
                continue
            refolds = discover_refold_pdbs(sample_tmp)
            if not refolds:
                msg = (
                    f"L={L} {sample_id}: no esm_*.pdb_esm files in "
                    f"{sample_tmp} — skipping"
                )
                if strict:
                    raise FileNotFoundError(msg)
                logger.warning(msg)
                continue
            if expected_n_refolds is not None and len(refolds) != expected_n_refolds:
                msg = (
                    f"L={L} {sample_id}: expected {expected_n_refolds} refolds, "
                    f"found {len(refolds)}"
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)

            per_sample: List[float] = []
            for r in refolds:
                try:
                    per_sample.append(compute_pair_rmsd(gen_pdb, r))
                except Exception as e:  # noqa: BLE001
                    msg = (
                        f"L={L} {sample_id}: RMSD failed for "
                        f"{os.path.basename(r)}: {e}"
                    )
                    if strict:
                        raise
                    logger.warning(msg)
            if per_sample:
                per_length.append(per_sample)
        rmsds_by_length[L] = per_length
        logger.info(
            f"L={L}: collected RMSDs for {len(per_length)} samples "
            f"({sum(len(s) for s in per_length)} pairs)"
        )
    return rmsds_by_length


def update_results_json(
    results_json_path: str,
    refold_summary: Dict[str, object],
    *,
    rmsds_by_length: Optional[Dict[int, List[List[float]]]] = None,
    include_per_sample: bool = False,
) -> None:
    """Merge `refold_summary` into the existing results.json (or create one).

    The merged block is stored under the top-level key ``refold_rmsd``.
    If `include_per_sample` is True, the raw per-sample RMSD lists are stored
    under ``refold_rmsd.per_sample_rmsds`` for downstream inspection.
    """
    if os.path.isfile(results_json_path):
        with open(results_json_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{results_json_path} is not a JSON object — refusing to merge."
            )
    else:
        data = {}
        os.makedirs(os.path.dirname(results_json_path) or ".", exist_ok=True)

    block = dict(refold_summary)
    if include_per_sample and rmsds_by_length is not None:
        block["per_sample_rmsds"] = {
            int(L): [list(s) for s in samples]
            for L, samples in rmsds_by_length.items()
        }
    data["refold_rmsd"] = block

    with open(results_json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-output-dir",
        required=True,
        help="EVAL_OUTPUT_DIR produced by scripts/eval_metrics.py.",
    )
    p.add_argument(
        "--samples-subdir",
        default="samples",
        help="Subdirectory holding L_<L>/sample_<NNNN>.pdb (default: samples).",
    )
    p.add_argument(
        "--designability-tmp-subdir",
        default="designability_tmp",
        help=(
            "Subdirectory holding L_<L>/sample_<NNNN>/esm_*.pdb_esm "
            "(default: designability_tmp)."
        ),
    )
    p.add_argument(
        "--expected-n-refolds",
        type=int,
        default=8,
        help="Expected refold count per sample; emits warning on mismatch (default: 8).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing refolds / count mismatch / RMSD failure instead of warning.",
    )
    p.add_argument(
        "--results-json",
        default=None,
        help=(
            "Path to results.json to merge into. Defaults to "
            "<eval-output-dir>/results.json."
        ),
    )
    p.add_argument(
        "--no-update-results-json",
        action="store_true",
        help="Just print the summary; do not modify results.json.",
    )
    p.add_argument(
        "--include-per-sample",
        action="store_true",
        help="Store the raw per-sample RMSD lists in the merged block.",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help=(
            "Optional standalone JSON to write the summary to (in addition to "
            "the results.json merge). Useful for sharing without exposing the "
            "full eval results.json."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    eval_dir = os.path.abspath(args.eval_output_dir)
    if not os.path.isdir(eval_dir):
        logger.error(f"--eval-output-dir does not exist: {eval_dir}")
        return 2

    logger.info("=" * 72)
    logger.info("Refold-RMSD post-hoc")
    logger.info(f"  eval_output_dir: {eval_dir}")
    logger.info(f"  samples_subdir:  {args.samples_subdir}")
    logger.info(f"  refold_subdir:   {args.designability_tmp_subdir}")
    logger.info("=" * 72)

    rmsds_by_length = collect_rmsds(
        eval_dir,
        samples_subdir=args.samples_subdir,
        designability_tmp_subdir=args.designability_tmp_subdir,
        expected_n_refolds=args.expected_n_refolds,
        strict=args.strict,
    )

    summary = aggregate_refold_rmsds(rmsds_by_length)

    logger.info("=" * 72)
    logger.info("Per-length and total mean RMSD (gen <-> refold)")
    logger.info("=" * 72)
    for L in sorted(summary["per_length_mean_all"]):
        ma = summary["per_length_mean_all"][L]
        mm = summary["per_length_mean_min"][L]
        ns = summary["per_length_n_samples"].get(L, 0)
        np_ = summary["per_length_n_pairs"].get(L, 0)
        logger.info(
            f"  L={L:>4d}: mean_all={ma} mean_min={mm} "
            f"(n_samples={ns}, n_pairs={np_})"
        )
    logger.info(
        f"  TOTAL : mean_all={summary['mean_all']} mean_min={summary['mean_min']} "
        f"(n_samples={summary['total_n_samples']}, n_pairs={summary['total_n_pairs']})"
    )

    results_json_path = args.results_json or os.path.join(eval_dir, "results.json")
    if not args.no_update_results_json:
        update_results_json(
            results_json_path,
            summary,
            rmsds_by_length=rmsds_by_length,
            include_per_sample=args.include_per_sample,
        )
        logger.info(f"Merged refold_rmsd block into {results_json_path}")
    else:
        logger.info("Skipping results.json update (--no-update-results-json)")

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = dict(summary)
        if args.include_per_sample:
            payload["per_sample_rmsds"] = {
                int(L): [list(s) for s in samples]
                for L, samples in rmsds_by_length.items()
            }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Wrote standalone summary JSON to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
