"""
Prepare a ~1000-protein PDB dataset for MeanFlow training.

Downloads CIF files, processes them into .pt graphs, writes the CSV index,
runs mmseqs2 clustering, and creates train/val/test splits.

Usage:
    DATA_PATH=/path/to/data python scripts/prepare_pdb_1k.py [--fraction 0.008]

The script is a thin wrapper around PDBLightningDataModule.prepare_data() and
setup('fit').  All filtering, downloading, and processing logic lives there.
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
        default=0.008,
        help="Fraction of PDB to use (default: 0.008, ~1000 chains).",
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

    logger.info("Running setup('fit') ...")
    dm.setup("fit")

    # Determine the CSV path for the user
    csv_name = (
        dm._get_file_identifier(dm.dataselector) + ".csv"
        if dm.dataselector
        else f"{dm.data_dir.name}.csv"
    )
    csv_path = dm.data_dir / csv_name

    train_size = len(dm.train_ds)
    val_size = len(dm.val_ds)
    # test_ds is only created for stage='test'; estimate from dfs_splits if available
    if dm.test_ds is not None:
        test_size = len(dm.test_ds)
    else:
        test_split = dm.dfs_splits.get("test") if dm.dfs_splits is not None else None
        test_size = len(test_split) if test_split is not None else None

    logger.info("Dataset ready.")
    print(f"\nDataset summary")
    print(f"  train : {train_size}")
    print(f"  val   : {val_size}")
    print(f"  test  : {test_size if test_size is not None else '(not loaded)'}")
    print(f"  CSV   : {csv_path}")


if __name__ == "__main__":
    main()
