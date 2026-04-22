# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Shared helpers for per-protein sampling + metric computation used by both
ProteinValEvalCallback and TrainSubsetRmsdCallback.

Keeping the per-protein scoring block here (rather than duplicating it in
each callback) ensures the two sets of curves — val and train-subset —
always use identical sampling code, so any gap between them is attributable
to the dataset partition, not to implementation drift.
"""

import os

import numpy as np
import torch
import wandb

from proteinfoundation.callbacks.protein_eval import (
    _ca_rmsd,
    _ca_rmsd_with_reflection,
    _chirality_sign,
)
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

# C-alpha is at index 1 in the atom37 representation.
CA_INDEX = 1


def extract_gt_ca(graph):
    """Pull ground-truth CA coords from a PyG graph, respecting the CA mask.

    Returns (gt_ca, n_res) or (None, 0) when no valid CA atoms remain.
    """
    ca_mask = graph.coord_mask[:, CA_INDEX]  # [n_res] bool
    gt_ca = graph.coords[:, CA_INDEX, :][ca_mask].cpu().numpy()  # [m, 3]
    n_res = gt_ca.shape[0]
    return (gt_ca, n_res) if n_res > 0 else (None, 0)


def write_gt_pdb(gt_ca, pid, tmp_dir):
    """Write a CA-only ground-truth PDB under tmp_dir; return the path."""
    n_res = gt_ca.shape[0]
    atom37 = np.zeros((n_res, 37, 3), dtype=np.float32)
    atom37[:, CA_INDEX, :] = gt_ca
    gt_path = os.path.join(tmp_dir, f"gt_{pid}.pdb")
    write_prot_to_pdb(atom37, gt_path, overwrite=True, no_indexing=True)
    return gt_path


def score_protein(pl_module, protein, nsamples, nsteps_base, tmp_dir, step):
    """Generate MF-1 / MF-10x samples for a single protein and compute metrics.

    Draws ``nsamples`` independent samples per mode, averages RMSD/chirality
    across draws, and writes the best-RMSD draw to a PDB so the wandb table
    viewer shows the representative structure.

    Args:
        pl_module: Lightning module with .generate() and .samples_to_atom37()
        protein: dict with keys {id, n_res, gt_ca, gt_pdb_path}
        nsamples: number of generations per mode; must be >= 1
        nsteps_base: base MeanFlow integration step count (mf1 uses this,
            mf10x uses 10 * this)
        tmp_dir: directory for the written sample PDBs
        step: training global_step, used to disambiguate PDB filenames

    Returns a dict with the averaged per-protein scores + pdb paths:
        {mf1_pdb, mf10x_pdb,
         rmsd_mf1, rmsd_mf10x, rmsd_refl_mf1, rmsd_refl_mf10x,
         chir_mf1, chir_mf10x, rmsd_mf1_best, rmsd_mf10x_best}
    """
    pid = protein["id"]
    n_res = protein["n_res"]
    gt_ca = protein["gt_ca"]

    mf1_draws = []
    mf10x_draws = []
    for _ in range(max(1, int(nsamples))):
        samples_mf1 = pl_module.generate(
            nsamples=1, n=n_res, nsteps=nsteps_base, mask=None
        )
        atom37_mf1 = pl_module.samples_to_atom37(samples_mf1).cpu().numpy()
        gen_ca_mf1 = atom37_mf1[0, :, CA_INDEX, :]

        samples_mf10x = pl_module.generate(
            nsamples=1, n=n_res, nsteps=nsteps_base * 10, mask=None
        )
        atom37_mf10x = pl_module.samples_to_atom37(samples_mf10x).cpu().numpy()
        gen_ca_mf10x = atom37_mf10x[0, :, CA_INDEX, :]

        mf1_draws.append(
            {
                "atom37": atom37_mf1[0],
                "rmsd": _ca_rmsd(gen_ca_mf1, gt_ca),
                "rmsd_refl": _ca_rmsd_with_reflection(gen_ca_mf1, gt_ca),
                "chir": _chirality_sign(gen_ca_mf1, gt_ca),
            }
        )
        mf10x_draws.append(
            {
                "atom37": atom37_mf10x[0],
                "rmsd": _ca_rmsd(gen_ca_mf10x, gt_ca),
                "rmsd_refl": _ca_rmsd_with_reflection(gen_ca_mf10x, gt_ca),
                "chir": _chirality_sign(gen_ca_mf10x, gt_ca),
            }
        )

    # Representative draw for the table viewer = best rmsd.
    best_mf1 = min(mf1_draws, key=lambda d: d["rmsd"])
    best_mf10x = min(mf10x_draws, key=lambda d: d["rmsd"])

    pdb_mf1 = os.path.join(tmp_dir, f"gen_{pid}_mf1_step{step}.pdb")
    write_prot_to_pdb(best_mf1["atom37"], pdb_mf1, overwrite=True, no_indexing=True)
    pdb_mf10x = os.path.join(tmp_dir, f"gen_{pid}_mf10x_step{step}.pdb")
    write_prot_to_pdb(best_mf10x["atom37"], pdb_mf10x, overwrite=True, no_indexing=True)

    return {
        "mf1_pdb": pdb_mf1,
        "mf10x_pdb": pdb_mf10x,
        "rmsd_mf1": float(np.mean([d["rmsd"] for d in mf1_draws])),
        "rmsd_mf10x": float(np.mean([d["rmsd"] for d in mf10x_draws])),
        "rmsd_refl_mf1": float(np.mean([d["rmsd_refl"] for d in mf1_draws])),
        "rmsd_refl_mf10x": float(np.mean([d["rmsd_refl"] for d in mf10x_draws])),
        "chir_mf1": float(np.mean([d["chir"] for d in mf1_draws])),
        "chir_mf10x": float(np.mean([d["chir"] for d in mf10x_draws])),
        "rmsd_mf1_best": float(best_mf1["rmsd"]),
        "rmsd_mf10x_best": float(best_mf10x["rmsd"]),
    }


SAMPLES_TABLE_COLUMNS = [
    "global_step",
    "lr",
    "protein_id",
    "rmsd_mf1",
    "rmsd_mf10x",
    "rmsd_reflected_mf1",
    "rmsd_reflected_mf10x",
    "chirality_mf1",
    "chirality_mf10x",
    "rmsd_mf1_best",
    "rmsd_mf10x_best",
    "ground_truth",
    "mf1",
    "mf10x",
]


def build_samples_table(proteins, results, step, current_lr):
    """Construct an annotated wandb.Table from a list of proteins + results.

    One row per protein. The columns are fixed in SAMPLES_TABLE_COLUMNS so both
    val and train-subset tables are schema-compatible (a reader can join them).
    """
    table = wandb.Table(columns=SAMPLES_TABLE_COLUMNS)
    for protein, res in zip(proteins, results):
        table.add_data(
            int(step),
            current_lr if current_lr is not None else float("nan"),
            protein["id"],
            float(res["rmsd_mf1"]),
            float(res["rmsd_mf10x"]),
            float(res["rmsd_refl_mf1"]),
            float(res["rmsd_refl_mf10x"]),
            float(res["chir_mf1"]),
            float(res["chir_mf10x"]),
            float(res["rmsd_mf1_best"]),
            float(res["rmsd_mf10x_best"]),
            wandb.Molecule(protein["gt_pdb_path"]),
            wandb.Molecule(res["mf1_pdb"]),
            wandb.Molecule(res["mf10x_pdb"]),
        )
    return table


def current_optimizer_lr(trainer):
    """Best-effort fetch of the current LR from the first optimizer.

    Returns None if the optimizer list is empty or attribute access fails.
    Callers should substitute float('nan') for missing values before writing
    into a wandb.Table (wandb rejects None).
    """
    try:
        optimizers = trainer.optimizers
        if optimizers:
            return float(optimizers[0].param_groups[0]["lr"])
    except Exception:
        pass
    return None


def aggregate_log_dict(prefix, proteins, results, step):
    """Build the mean-over-proteins scalar dict for a sampling round.

    ``prefix`` is e.g. "val" or "train_subset"; keys become
    ``<prefix>/rmsd_mf1`` etc. ``trainer/global_step`` is set on the caller's
    log_dict, not here — val and train_subset share the same global_step
    axis so it would be redundant.
    """
    return {
        f"{prefix}/rmsd_mf1": float(np.mean([r["rmsd_mf1"] for r in results])),
        f"{prefix}/rmsd_mf10x": float(np.mean([r["rmsd_mf10x"] for r in results])),
        f"{prefix}/rmsd_reflected_mf1": float(
            np.mean([r["rmsd_refl_mf1"] for r in results])
        ),
        f"{prefix}/rmsd_reflected_mf10x": float(
            np.mean([r["rmsd_refl_mf10x"] for r in results])
        ),
        f"{prefix}/chirality_mf1": float(np.mean([r["chir_mf1"] for r in results])),
        f"{prefix}/chirality_mf10x": float(
            np.mean([r["chir_mf10x"] for r in results])
        ),
        f"{prefix}/rmsd_mf1_best": float(
            np.mean([r["rmsd_mf1_best"] for r in results])
        ),
        f"{prefix}/rmsd_mf10x_best": float(
            np.mean([r["rmsd_mf10x_best"] for r in results])
        ),
    }
