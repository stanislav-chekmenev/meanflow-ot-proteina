"""CPU-only tests for scripts/eval_metrics.py helpers.

These cover the pure-Python pieces:
  * TM-score helper (Kabsch-aligned, simplified TM-score).
  * Pairwise diversity aggregation (mocked TM-score).
  * Cluster diversity parsing of a Foldseek `*_cluster.tsv` (mocked subprocess).
  * Designability rate aggregation from a list of scRMSDs.
  * TMalign output parser.
  * Novelty placeholder.

Expensive pieces (ProteinMPNN, ESMFold, GearNet, Proteina sampling, real
TMalign / foldseek subprocesses) are integration-only; they are not exercised
here.
"""
from __future__ import annotations

import logging
import math
import subprocess
from unittest import mock

import pytest
import torch

from scripts.eval_metrics import (
    aggregate_pairwise_diversity,
    cluster_diversity_from_dir,
    compute_novelty,
    designability_rate,
    parse_foldseek_cluster_tsv,
    parse_tmalign_output,
    tm_score_ca,
)


# -------------------------- tm_score_ca ---------------------------------


def test_tm_score_identical_structures_is_one():
    coors = torch.randn(80, 3)
    score = tm_score_ca(coors, coors.clone())
    assert math.isclose(score, 1.0, abs_tol=1e-5)


def test_tm_score_invariant_to_rigid_transform():
    n = 64
    coors = torch.randn(n, 3) * 5.0
    # Rigid transform: random rotation + translation.
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = torch.tensor([1.5, -2.3, 0.7])
    coors_b = coors @ R.T + t
    score = tm_score_ca(coors, coors_b)
    assert math.isclose(score, 1.0, abs_tol=1e-4)


def test_tm_score_decreases_with_noise():
    n = 64
    coors_a = torch.randn(n, 3) * 5.0
    # Heavy gaussian perturbation so TM-score must drop materially.
    coors_b = coors_a + torch.randn(n, 3) * 5.0
    score = tm_score_ca(coors_a, coors_b)
    assert 0.0 <= score <= 1.0
    assert score < 0.6


def test_tm_score_short_chain_uses_fallback_d0():
    # L < 19 should use the d_0 = 0.5 floor.
    n = 10
    coors = torch.randn(n, 3)
    score_same = tm_score_ca(coors, coors.clone())
    assert math.isclose(score_same, 1.0, abs_tol=1e-5)


def test_tm_score_mismatched_lengths_raises():
    a = torch.randn(40, 3)
    b = torch.randn(50, 3)
    with pytest.raises((AssertionError, ValueError)):
        tm_score_ca(a, b)


# -------------------------- aggregate_pairwise_diversity ---------------


def _const_tm_score(value):
    """Return a fake tm_score_ca that always returns `value`."""
    def _fn(_a, _b):
        return value
    return _fn


def test_aggregate_pairwise_diversity_single_length_two_samples():
    # Two designable samples -> one pair.
    coors = [torch.zeros(20, 3), torch.zeros(20, 3)]
    by_length = {50: coors}
    result = aggregate_pairwise_diversity(
        by_length, tm_score_fn=_const_tm_score(0.42)
    )
    assert math.isclose(result["per_length_mean_tm"][50], 0.42, abs_tol=1e-9)
    assert math.isclose(result["mean_pairwise_tm"], 0.42, abs_tol=1e-9)
    assert result["pairs_per_length"][50] == 1


def test_aggregate_pairwise_diversity_aggregates_across_lengths():
    # Length 50: TM=0.2 always; length 100: TM=0.8 always.
    # Per-length means: {50: 0.2, 100: 0.8}; across-length mean: 0.5.
    samples_50 = [torch.zeros(20, 3) for _ in range(3)]
    samples_100 = [torch.zeros(30, 3) for _ in range(3)]
    by_length = {50: samples_50, 100: samples_100}

    def _fn(a, b):
        # Distinguish by length.
        return 0.2 if a.shape[0] == 20 else 0.8

    result = aggregate_pairwise_diversity(by_length, tm_score_fn=_fn)
    assert math.isclose(result["per_length_mean_tm"][50], 0.2, abs_tol=1e-9)
    assert math.isclose(result["per_length_mean_tm"][100], 0.8, abs_tol=1e-9)
    assert math.isclose(result["mean_pairwise_tm"], 0.5, abs_tol=1e-9)


def test_aggregate_pairwise_diversity_skips_lengths_with_lt2_samples():
    # Length 50: 1 sample (skipped). Length 100: 2 samples (used).
    by_length = {50: [torch.zeros(20, 3)], 100: [torch.zeros(30, 3), torch.zeros(30, 3)]}
    result = aggregate_pairwise_diversity(
        by_length, tm_score_fn=_const_tm_score(0.5)
    )
    assert 50 not in result["per_length_mean_tm"]
    assert math.isclose(result["per_length_mean_tm"][100], 0.5, abs_tol=1e-9)
    assert math.isclose(result["mean_pairwise_tm"], 0.5, abs_tol=1e-9)


def test_aggregate_pairwise_diversity_returns_none_when_no_pairs():
    # All lengths have <2 designable samples -> nothing to compute.
    by_length = {50: [torch.zeros(20, 3)], 100: []}
    result = aggregate_pairwise_diversity(
        by_length, tm_score_fn=_const_tm_score(0.5)
    )
    assert result["mean_pairwise_tm"] is None
    assert result["per_length_mean_tm"] == {}


# -------------------------- parse_foldseek_cluster_tsv ------------------


def test_parse_foldseek_cluster_tsv_counts_unique_reps(tmp_path):
    tsv = tmp_path / "res_cluster.tsv"
    # cluster_rep \t member rows. 3 unique reps over 5 members.
    tsv.write_text(
        "rep_a\trep_a\n"
        "rep_a\tmemb_1\n"
        "rep_b\trep_b\n"
        "rep_c\trep_c\n"
        "rep_c\tmemb_2\n"
    )
    n_clusters = parse_foldseek_cluster_tsv(str(tsv))
    assert n_clusters == 3


def test_parse_foldseek_cluster_tsv_blank_lines_ignored(tmp_path):
    tsv = tmp_path / "res_cluster.tsv"
    tsv.write_text(
        "rep_a\trep_a\n"
        "\n"
        "rep_b\trep_b\n"
        "   \n"
    )
    assert parse_foldseek_cluster_tsv(str(tsv)) == 2


# -------------------------- cluster_diversity_from_dir ------------------


def test_cluster_diversity_from_dir_invokes_foldseek_and_returns_ratio(
    tmp_path, monkeypatch
):
    # Build a directory with three fake "designable" PDBs.
    samples_dir = tmp_path / "designable"
    samples_dir.mkdir()
    for i in range(3):
        (samples_dir / f"sample_{i}.pdb").write_text("HEADER fake\nEND\n")

    cluster_tsv_text = "rep_a\trep_a\nrep_a\tsample_1\nrep_b\trep_b\n"

    def _fake_run(cmd, **kwargs):
        # Foldseek call signature:
        #   <bin> easy-cluster <samples_dir> <prefix>/res <prefix> --alignment-type 1 ...
        # The implementation passes a tmp scratch dir; we simulate the cluster
        # tsv that easy-cluster would write at <prefix>/res_cluster.tsv.
        assert cmd[0].endswith("foldseek")
        assert cmd[1] == "easy-cluster"
        # cmd[2] is the samples dir; cmd[3] is <prefix>/res; cmd[4] is the tmp prefix dir.
        prefix_res = cmd[3]
        # Foldseek writes <prefix_res>_cluster.tsv
        out_tsv = prefix_res + "_cluster.tsv"
        with open(out_tsv, "w") as f:
            f.write(cluster_tsv_text)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = cluster_diversity_from_dir(
        str(samples_dir), foldseek_bin="/usr/local/bin/foldseek"
    )
    # 2 clusters / 3 designable -> 0.6666...
    assert result["n_clusters"] == 2
    assert result["n_designable"] == 3
    assert math.isclose(result["cluster_diversity"], 2.0 / 3.0, abs_tol=1e-9)


def test_cluster_diversity_from_dir_empty_dir_returns_none(tmp_path):
    samples_dir = tmp_path / "empty"
    samples_dir.mkdir()
    result = cluster_diversity_from_dir(
        str(samples_dir), foldseek_bin="/usr/local/bin/foldseek"
    )
    assert result["n_clusters"] == 0
    assert result["n_designable"] == 0
    assert result["cluster_diversity"] is None


# -------------------------- designability_rate --------------------------


def test_designability_rate_min_below_threshold_counts():
    # Each sample provides a list of 8 scRMSDs (one per ProteinMPNN seq).
    sample_rmsds = [
        [3.5, 1.5, 2.7, 4.0, 2.1, 5.0, 3.0, 2.9],   # min=1.5 -> designable
        [4.0, 3.5, 5.0, 6.0, 7.0, 8.0, 4.5, 3.8],   # min=3.5 -> not
        [1.9, 2.0, 2.1, 2.2, 1.8, 2.5, 2.3, 2.4],   # min=1.8 -> designable
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],   # min=2.0 -> NOT (strict <)
    ]
    result = designability_rate(sample_rmsds, threshold=2.0)
    assert result["n_total"] == 4
    assert result["n_designable"] == 2
    assert math.isclose(result["designability"], 0.5, abs_tol=1e-9)
    assert result["designable_indices"] == [0, 2]


def test_designability_rate_empty():
    result = designability_rate([], threshold=2.0)
    assert result["n_total"] == 0
    assert result["n_designable"] == 0
    assert result["designability"] is None
    assert result["designable_indices"] == []


def test_designability_rate_custom_threshold():
    sample_rmsds = [
        [2.5, 2.6],
        [2.9, 3.0],
        [3.0, 3.0],
    ]
    # threshold=3.0: first two designable, last not (2.5<3, 2.9<3, 3.0 not <3).
    result = designability_rate(sample_rmsds, threshold=3.0)
    assert result["n_designable"] == 2
    assert result["designable_indices"] == [0, 1]


# -------------------------- parse_tmalign_output ------------------------


_TMALIGN_SAMPLE_OUTPUT = """\

 *********************************************************************
 * TM-align (Version 20220412): protein structure alignment          *
 *********************************************************************

Name of Chain_1: a.pdb (to be superimposed onto Chain_2)
Name of Chain_2: b.pdb
Length of Chain_1: 100 residues
Length of Chain_2: 100 residues

Aligned length= 100, RMSD=   1.23, Seq_ID=n_identical/n_aligned= 0.500
TM-score= 0.72340 (if normalized by length of Chain_1, i.e., LN=100, d0=3.99)
TM-score= 0.71200 (if normalized by length of Chain_2, i.e., LN=100, d0=3.99)
(You should use TM-score normalized by length of the reference protein)
"""


def test_parse_tmalign_output_picks_chain1_normalized():
    score = parse_tmalign_output(_TMALIGN_SAMPLE_OUTPUT)
    assert math.isclose(score, 0.72340, abs_tol=1e-6)


def test_parse_tmalign_output_chain2_when_requested():
    score = parse_tmalign_output(_TMALIGN_SAMPLE_OUTPUT, normalize_by="chain2")
    assert math.isclose(score, 0.71200, abs_tol=1e-6)


def test_parse_tmalign_output_missing_line_raises():
    with pytest.raises(ValueError):
        parse_tmalign_output("nothing useful here\n")


def test_parse_tmalign_output_unknown_normalize_raises():
    with pytest.raises(ValueError):
        parse_tmalign_output(_TMALIGN_SAMPLE_OUTPUT, normalize_by="foo")


# -------------------------- compute_novelty -----------------------------


def test_compute_novelty_returns_none_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = compute_novelty(designable_pdbs=["a.pdb", "b.pdb"])
    assert result is None
    # Documented placeholder log line must show up in captured output.
    text = " ".join(rec.getMessage() for rec in caplog.records)
    # Loguru by default doesn't propagate to logging — eval_metrics installs
    # a propagation bridge so caplog can see it. If neither of these are seen,
    # the test fails loudly here.
    assert "Novelty" in text or "novelty" in text
    assert "not yet implemented" in text or "placeholder" in text
