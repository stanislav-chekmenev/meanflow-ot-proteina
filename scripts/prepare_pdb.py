"""
Prepare a PDB dataset for MeanFlow training.

Downloads CIF files, processes them into .pt graphs, writes the CSV index, and
emits the input FASTA (`seq_<file_identifier>.fasta`) ready for mmseqs2.

Usage:
    DATA_PATH=/path/to/data python scripts/prepare_pdb.py [--fraction 0.5]

The script is a thin wrapper around PDBLightningDataModule.prepare_data() plus
the FASTA write from datasplitter.split_data(). The mmseqs2 clustering step and
the train/val/test split are intentionally skipped here so prep can run on a
node without mmseqs; run mmseqs separately on a machine that has it, then the
trainer's setup('fit') will pick up the cluster files and build splits.
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a PDB subset for training."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of PDB to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = os.environ.get("DATA_PATH")
    if not data_path:
        logger.error(
            "DATA_PATH environment variable is not set. "
            "Export it to the directory where datasets should be stored, e.g.:\n"
            "  export DATA_PATH=/netscratch/schekmenev/data"
        )
        sys.exit(1)

    logger.info(f"DATA_PATH={data_path}")
    logger.info(f"fraction={args.fraction}")

    # Hydra compose — must use initialize_config_dir with an absolute path
    # because this script may be invoked from an arbitrary working directory
    # (e.g. a git worktree).
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = os.path.join(ROOT, "configs", "datasets_config", "pdb")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="pdb_train",
            overrides=[
                f"datamodule.dataselector.fraction={args.fraction}",
                "datamodule.num_workers=16",
            ],
        )

    dm = instantiate(cfg.datamodule)

    logger.info("Running prepare_data() ...")
    dm.prepare_data()

    if not dm.dataselector:
        logger.info("No dataselector configured; skipping FASTA write.")
        return

    file_identifier = dm._get_file_identifier(dm.dataselector)
    csv_path = dm.data_dir / f"{file_identifier}.csv"

    import pandas as pd
    from proteinfoundation.utils.cluster_utils import (
        df_to_fasta,
        setup_clustering_file_paths,
    )

    df_data = pd.read_csv(csv_path)
    n_rows = len(df_data)

    sim = dm.datasplitter.split_sequence_similarity
    input_fasta_filepath, cluster_fasta_filepath, cluster_tsv_filepath = (
        setup_clustering_file_paths(dm.data_dir, file_identifier, sim)
    )

    if not input_fasta_filepath.exists():
        logger.info(f"Writing input FASTA to {input_fasta_filepath}")
        df_to_fasta(df=df_data, output_file=input_fasta_filepath)
    else:
        logger.info(f"Input FASTA already exists, skipping: {input_fasta_filepath}")

    logger.info("Dataset ready (prep only — run mmseqs2 next, then training will split).")
    print(f"\nDataset summary")
    print(f"  rows       : {n_rows}")
    print(f"  CSV        : {csv_path}")
    print(f"  input FASTA: {input_fasta_filepath}")
    print(f"\nNext step (on a node with mmseqs2 on PATH, from {dm.data_dir}):")
    print(
        f"  mmseqs easy-cluster {input_fasta_filepath.name} pdb_cluster tmp "
        f"--min-seq-id {sim} -c 0.8 --cov-mode 1"
    )
    print(f"  mv pdb_cluster_rep_seq.fasta {cluster_fasta_filepath.name}")
    print(f"  mv pdb_cluster_cluster.tsv   {cluster_tsv_filepath.name}")


if __name__ == "__main__":
    main()
