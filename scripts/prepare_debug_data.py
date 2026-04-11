"""
Prepare a single-protein debug dataset for MeanFlow training.

Reads data/debug/raw/1ubq.cif, processes it into a PyG .pt file
(same pipeline as PDBLightningDataModule._load_and_process_pdb),
and writes the CSV index expected by the datamodule.

Usage:
    python scripts/prepare_debug_data.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pathlib

import pandas as pd
import torch

from graphein_utils.graphein_utils import protein_to_pyg
from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.datasets.transforms import ChainBreakPerResidueTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR


DATA_DIR = pathlib.Path(ROOT) / "data" / "debug"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FORMAT = "cif"


def process_protein(pdb_name: str) -> str:
    """Process a single CIF file into a .pt graph, mirroring
    PDBLightningDataModule._load_and_process_pdb."""
    path = RAW_DIR / f"{pdb_name}.{FORMAT}"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    fill_value_coords = 1e-5
    graph = protein_to_pyg(
        path=str(path),
        chain_selection="all",
        keep_insertions=True,
        store_het=False,
        store_bfactor=True,
        fill_value_coords=fill_value_coords,
    )

    fname = f"{pdb_name}.pt"
    graph.id = pdb_name
    coord_mask = graph.coords != fill_value_coords
    graph.coord_mask = coord_mask[..., 0]
    graph.residue_type = torch.tensor(
        [resname_to_idx[residue] for residue in graph.residues]
    ).long()
    graph.database = "pdb"
    graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
    graph.residue_pdb_idx = torch.tensor(
        [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
    )
    graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

    # Apply chain-break transform (used in training)
    chain_break_transform = ChainBreakPerResidueTransform()
    graph = chain_break_transform(graph)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graph, PROCESSED_DIR / fname)
    return fname


def main():
    pdb_name = "1ubq"
    print(f"Processing {pdb_name}...")
    fname = process_protein(pdb_name)

    # Write CSV index (expected by PDBLightningDataModule when dataselector=None)
    # The CSV filename must match data_dir.name: "debug.csv"
    csv_path = DATA_DIR / "debug.csv"
    df = pd.DataFrame({"pdb": [pdb_name], "id": [pdb_name]})
    df.to_csv(csv_path, index=False)

    # Verify
    graph = torch.load(PROCESSED_DIR / fname, weights_only=False)
    n_res = graph.coords.shape[0]
    n_atoms = graph.coords.shape[1]
    print(f"Saved {PROCESSED_DIR / fname}")
    print(f"  residues:      {n_res}")
    print(f"  atoms/residue:  {n_atoms}")
    print(f"  coords shape:   {tuple(graph.coords.shape)}")
    print(f"  coord_mask:     {tuple(graph.coord_mask.shape)}")
    print(f"  residue_type:   {tuple(graph.residue_type.shape)}")
    print(f"  chain_breaks:   {tuple(graph.chain_breaks_per_residue.shape)}")
    print(f"Wrote CSV index:  {csv_path}")


if __name__ == "__main__":
    main()
