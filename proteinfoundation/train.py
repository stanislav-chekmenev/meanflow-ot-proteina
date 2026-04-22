# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import argparse
import json
import pickle
import traceback
from pathlib import Path

import hydra
import lightning as L
import loralib as lora
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ema_utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.metric_utils import (
    transform_global_percentage_to_mask_dropout,
)
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
    StartupInfoCallback,
)
from proteinfoundation.callbacks.protein_eval import ProteinEvalCallback
from proteinfoundation.callbacks.protein_train_eval import TrainSubsetRmsdCallback
from proteinfoundation.callbacks.protein_val_eval import ProteinValEvalCallback


class GlobalStepWandbLogger(WandbLogger):
    """WandbLogger that pins the ``trainer/global_step`` x-axis to the real
    optimizer global_step.

    Lightning's eval loop passes the per-dataloader val-batch counter as the
    ``step`` argument to ``log_metrics`` (see ``evaluation_loop.py:459``),
    which the stock WandbLogger writes verbatim into ``trainer/global_step``
    — corrupting the x-axis for every ``val/*_step`` metric. This override
    rewrites ``step`` from the attached trainer so the x-axis is always the
    true optimizer step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainer_ref = None

    def attach_trainer(self, trainer):
        self._trainer_ref = trainer

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if self._trainer_ref is not None:
            step = int(self._trainer_ref.global_step)
        return super().log_metrics(metrics, step=step)


class AttachTrainerToLoggerCallback(Callback):
    """Binds the trainer to ``GlobalStepWandbLogger`` as soon as setup runs,
    so subsequent ``log_metrics`` calls can pull the real global_step."""

    def setup(self, trainer, pl_module, stage):
        for lg in trainer.loggers:
            if isinstance(lg, GlobalStepWandbLogger):
                lg.attach_trainer(trainer)


# Things that should only be done by a single process
@rank_zero_only
def log_info(msg):
    logger.info(msg)


@rank_zero_only
def store_configs(path_configs):
    for cfg, path in path_configs:
        with open(path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)


@rank_zero_only
def log_configs(path_configs, wandb_logger, run_name):
    if wandb_logger is None:
        return
    artifact = wandb.Artifact(f"config_files_{run_name}", type="config")
    for _, path in path_configs:
        artifact.add_file(path)
    wandb_logger.experiment.log_artifact(artifact)


@rank_zero_only
def create_dir(checkpoint_path_store, parents=True, exist_ok=True):
    Path(checkpoint_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def _build_ckpt_callbacks(checkpoint_path_store, cfg_log):
    """Build the list of checkpoint callbacks.

    Always returns the "last" ckpt callback (overwrites itself every
    ``last_ckpt_every_n_steps`` steps, used for requeuing).

    If ``cfg_log.top_k_by_val_rmsd_mf1`` > 0, additionally returns a top-k
    callback that monitors ``val/rmsd_mf1`` (lower-is-better) and keeps the
    best K checkpoints. Lightning evaluates the monitor at validation-end,
    which is the natural cadence; do NOT set ``every_n_train_steps`` on the
    top-k callback — it would couple to train steps and fire before the
    monitor is populated.

    Args:
        checkpoint_path_store: directory where checkpoints are written.
        cfg_log: OmegaConf-like log block with keys
            ``last_ckpt_every_n_steps``, ``checkpoint_every_n_steps``,
            ``top_k_by_val_rmsd_mf1`` (default 3).

    Returns:
        list of EmaModelCheckpoint callbacks (length 1 or 2).
    """
    args_ckpt_last = {
        "dirpath": checkpoint_path_store,
        "save_weights_only": False,
        "filename": "ignore",
        "every_n_train_steps": cfg_log.last_ckpt_every_n_steps,
        "save_last": True,
    }
    callbacks = [EmaModelCheckpoint(**args_ckpt_last)]

    top_k = int(cfg_log.get("top_k_by_val_rmsd_mf1", 3))
    if top_k > 0:
        args_ckpt_topk = {
            "dirpath": checkpoint_path_store,
            "save_last": False,
            "save_weights_only": False,
            "filename": "chk_{epoch:08d}_{step:012d}_{val/rmsd_mf1:.4f}",
            "monitor": "val/rmsd_mf1",
            "mode": "min",
            "save_top_k": top_k,
            "save_on_train_epoch_end": False,
        }
        callbacks.append(EmaModelCheckpoint(**args_ckpt_topk))
        log_info(
            f"Checkpointing: keeping top-{top_k} by val/rmsd_mf1 (mode=min) "
            f"+ last ckpt every {cfg_log.last_ckpt_every_n_steps} steps."
        )
    else:
        log_info(
            f"Checkpointing: top-k disabled (top_k_by_val_rmsd_mf1=0); "
            f"keeping only last ckpt every {cfg_log.last_ckpt_every_n_steps} steps."
        )

    return callbacks


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="training_ca",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Avoids checkpoints and wandb logging, mostly for debugging.",
    )
    parser.add_argument(
        "--single", action="store_true", help="Sets single node and single GPU, ignoring config file."
    )
    parser.add_argument(
        "--show_prog_bar",
        action="store_true",
        help="Shows progress bar as training progresses.",
    )
    parser.add_argument(
        "--exp_overrides",
        nargs="*",
        default=[],
        help="Hydra overrides for the experiment config, e.g. opt.accumulate_grad_batches=4",
    )
    parser.add_argument(
        "--data_overrides",
        nargs="*",
        default=[],
        help="Hydra overrides for the dataset config, e.g. datamodule.batch_size=32",
    )
    args = parser.parse_args()

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout
    log_info(f"Avoid wandb and checkpointing: {args.nolog}")
    callbacks = [SeedCallback()]  # Different devices will be assigend different seeds

    # Load experiment config
    config_path = "../configs/experiment_config"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name, overrides=args.exp_overrides)
        if args.single:
            # Rewrite number of GPUs and nodes for local runs or if single flag is used
            cfg_exp.hardware.ngpus_per_node_ = 1
            cfg_exp.hardware.nnodes_ = 1
            cfg_exp.run_name_ = cfg_exp.run_name_ + "_single"
        log_info(f"Exp config {cfg_exp}")

    # Set training precision
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        # precision = "16"
        precision = "bf16-mixed"

    # Set training fold labels dropout rate based on global percentage
    if cfg_exp.training.get("fold_label_sample_ratio") is not None:
        log_info("Setting fold label dropout rate based on fold_label_sample_ratio")
        (
            cfg_exp.training.mask_T_prob,
            cfg_exp.training.mask_A_prob,
            cfg_exp.training.mask_C_prob,
        ) = transform_global_percentage_to_mask_dropout(
            cfg_exp.training.fold_label_sample_ratio
        )
        log_info(
            "Set mask_T_prob: %.3f, mask_A_prob: %.3f, mask_C_prob: %.3f"
            % (
                cfg_exp.training.mask_T_prob,
                cfg_exp.training.mask_A_prob,
                cfg_exp.training.mask_C_prob,
            )
        )

    # Set run name and root directory for the run, used to store things
    run_name = cfg_exp.run_name_
    log_info(f"Job name: {run_name}")
    root_run = os.path.join(
        ".", "store", run_name
    )  # Everything stored in ./store/<run_id>
    log_info(f"Root run: {root_run}")

    # Set checkpoint directory
    checkpoint_path_store = os.path.join(
        root_run, "checkpoints"
    )  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {checkpoint_path_store}")

    # Check if last checkpoint exists (this is useful if interrupted, it starts from last checkpoint)
    last_ckpt_name = fetch_last_ckpt(checkpoint_path_store)
    last_ckpt_path = (
        os.path.join(checkpoint_path_store, last_ckpt_name)
        if last_ckpt_name is not None
        else None
    )
    log_info(f"Last checkpoint: {last_ckpt_path}")

    # Extract number of cpus from config file
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(
        f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}"
    )

    # If no checkpoint set seed for correct initialization
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    # Load data config
    dataset_config_subdir = cfg_exp.get("dataset_config_subdir", None)
    if dataset_config_subdir is not None:
        # if args.dataset_config_subdir:
        config_path = f"../configs/datasets_config/{dataset_config_subdir}"
    else:
        config_path = "../configs/datasets_config/"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"], overrides=args.data_overrides)
        cfg_data.datamodule.num_workers = num_cpus  # Overwrite number of cpus
        if cfg_data.get("exclude_id_pkl_path") is not None:
            with open(cfg_data.exclude_id_pkl_path, "rb") as fin:
                exclude_ids = pickle.load(fin)
            if cfg_data.datamodule.dataselector.exclude_ids is not None:
                cfg_data.datamodule.dataselector.exclude_ids += exclude_ids
            else:
                cfg_data.datamodule.dataselector.exclude_ids = exclude_ids
        log_info(f"Data config {cfg_data}")

    # create datamodule containing default train and val dataloader
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)

    # Set logger — DDP-safe: disable wandb on non-rank-0 to avoid deadlock under srun
    wandb_logger = None
    csv_logger = None
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    if local_rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    if cfg_exp.log.log_wandb and not args.nolog:
        wandb_entity = cfg_exp.log.get("wandb_entity", None)
        wandb_run_name = cfg_exp.log.get("wandb_run_name", None)
        # Generate a unique run ID so each restart creates a fresh WandB run
        # instead of resuming the previous one.
        wandb_run_id = wandb.util.generate_id()
        wandb_logger = GlobalStepWandbLogger(
            project=cfg_exp.log.wandb_project,
            id=wandb_run_id,
            entity=wandb_entity,
            name=wandb_run_name or run_name,
            config=OmegaConf.to_container(cfg_exp, resolve=True),
        )
        callbacks.append(AttachTrainerToLoggerCallback())
        callbacks.append(LogEpochTimeCallback())
        callbacks.append(LogSetpTimeCallback())
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        # Guardrail: log dataloader length and optimizer-steps-per-epoch at
        # training start so accumulate_grad_batches surprises (PL 2.5 flush on
        # final batch) are visible in every job's log.
        callbacks.append(StartupInfoCallback())
    else:
        # CSV fallback so metrics are saved even without wandb
        csv_logger = CSVLogger(save_dir=root_run, name="csv_logs")

    # Protein eval callback — generate and visualize proteins periodically
    eval_cfg = cfg_exp.get("eval", None)
    if eval_cfg is not None and eval_cfg.get("enabled", False) and wandb_logger is not None:
        gt_path = eval_cfg.get("ground_truth_pdb", None)
        eval_nsamples = int(eval_cfg.get("nsamples", 1))
        callbacks.append(
            ProteinEvalCallback(
                eval_every_n_steps=eval_cfg.every_n_steps,
                n_residues=eval_cfg.n_residues,
                run_name=run_name,
                ground_truth_pdb_path=gt_path,
                nsamples=eval_nsamples,
            )
        )

    if eval_cfg is not None and eval_cfg.get("val_enabled", False) and wandb_logger is not None:
        n_val = int(eval_cfg.get("n_val_proteins", 16))
        val_nsamples = int(eval_cfg.get("nsamples", 1))
        callbacks.append(
            ProteinValEvalCallback(
                run_name=run_name,
                n_val_proteins=n_val,
                nsamples=val_nsamples,
            )
        )

    if (
        eval_cfg is not None
        and eval_cfg.get("train_subset_enabled", False)
        and wandb_logger is not None
    ):
        n_train = int(eval_cfg.get("n_train_proteins", 16))
        train_subset_nsamples = int(eval_cfg.get("nsamples", 1))
        train_subset_seed = int(eval_cfg.get("train_subset_seed", 42))
        callbacks.append(
            TrainSubsetRmsdCallback(
                run_name=run_name,
                n_train_proteins=n_train,
                nsamples=train_subset_nsamples,
                seed=train_subset_seed,
            )
        )

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))

    # Set checkpointing
    if cfg_exp.log.checkpoint and not args.nolog:
        create_dir(checkpoint_path_store, parents=True, exist_ok=True)
        ckpt_callbacks = _build_ckpt_callbacks(checkpoint_path_store, cfg_exp.log)
        callbacks.extend(ckpt_callbacks)

        # Save and log config files
        path_configs = [
            (
                cfg_data,
                os.path.join(checkpoint_path_store, f"data_config_{run_name}.json"),
            ),
            (
                cfg_exp,
                os.path.join(checkpoint_path_store, f"exp_config_{run_name}.json"),
            ),
        ]
        store_configs(path_configs)
        log_configs(path_configs, wandb_logger, run_name)

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    # Define model
    model = Proteina(cfg_exp, store_dir=root_run)

    # If LoRA is tunred on, replace Linear with LoRA layers
    if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model, cfg_exp.lora.r, cfg_exp.lora.lora_alpha, cfg_exp.lora.lora_dropout
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # Train
    plugins = []
    show_prog_bar = args.show_prog_bar
    active_logger = wandb_logger if wandb_logger is not None else csv_logger
    trainer = L.Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=active_logger,
        log_every_n_steps=cfg_exp.opt.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=None,  # Leave like this
        val_check_interval=cfg_exp.opt.val_check_interval,
        strategy=cfg_exp.opt.dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=1,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )
    try:
        trainer.fit(
            model, datamodule, ckpt_path=last_ckpt_path
        )  # If None then it starts from scratch
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Training failed:\n{tb}")
        if wandb.run is not None:
            try:
                wandb.run.alert(
                    title="Training crashed",
                    text=str(e),
                    level=wandb.AlertLevel.ERROR,
                )
                wandb.finish(exit_code=1)
            except Exception:
                pass
        sys.exit(1)

    # Ensure wandb run finalizes properly on clean exit
    if wandb.run is not None:
        wandb.finish()
