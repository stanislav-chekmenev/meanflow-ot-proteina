#!/bin/bash
#SBATCH --job-name=hyp_adp_wt_scaling
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/hyp_adp_wt_scaling_%j.out
#SBATCH --error=logs/hyp_adp_wt_scaling_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
# REPO_ROOT is the primary repo (shared venv tarball, data, logs).
# CODE_ROOT is this hypothesis's git worktree — configs and code come from
# the hypothesis branch so the run picks up branch-specific changes.
REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
CODE_ROOT="$REPO_ROOT/.claude/worktrees/agent-a7dc3fe7"
DATA_ROOT="/netscratch/schekmenev"

# Unpack .venv to scratch (once per node, skips if already present)
# Create tarball on login node:
#   uv venv --relocatable --python 3.11 && uv sync
#   tar czf "$REPO_ROOT/venv.tar.gz" -C "$REPO_ROOT" .venv
PROJECT_NAME="$(basename "$REPO_ROOT")"
ENV_TARBALL="$REPO_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/venvs/$PROJECT_NAME/.venv"

echo "Setting up virtual environment on node-local scratch..."
STAMP_FILE="$DATA_ROOT/venvs/$PROJECT_NAME/.tarball_mtime"
TARBALL_MTIME="$(stat -c %Y "$ENV_TARBALL")"
LOCAL_STAMP="$(cat "$STAMP_FILE" 2>/dev/null || echo "")"

if [ "$TARBALL_MTIME" != "$LOCAL_STAMP" ]; then
    echo "$(hostname): venv outdated or missing (tarball mtime=$TARBALL_MTIME, local=$LOCAL_STAMP) — re-extracting..."
    rm -rf "$DATA_ROOT/venvs/$PROJECT_NAME"
    mkdir -p "$DATA_ROOT/venvs/$PROJECT_NAME"
    if ! tar xzf "$ENV_TARBALL" -C "$DATA_ROOT/venvs/$PROJECT_NAME"; then
        echo "ERROR: Failed to unpack $ENV_TARBALL" >&2
        exit 1
    fi
    echo "$TARBALL_MTIME" > "$STAMP_FILE"
    echo "$(hostname): env unpacked to $ENV_LOCAL"
else
    echo "$(hostname): env up-to-date at $ENV_LOCAL"
fi
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"

echo "Virtual environment set up at $ENV_LOCAL"

# Ensure project is importable — use CODE_ROOT so imports resolve to the
# hypothesis-branch code, not the primary repo's main branch.
export PYTHONPATH="${PYTHONPATH:-}:$CODE_ROOT"
export PYTHONUNBUFFERED=1

# DATA_PATH env var — configs resolve ${oc.env:DATA_PATH}/debug/.
# Data lives in REPO_ROOT (shared across worktrees).
export DATA_PATH="$REPO_ROOT/data"

mkdir -p "$REPO_ROOT/logs"

# --- Prepare debug data (if not already done) ---
if [ ! -f "$DATA_PATH/debug/processed/1ubq.pt" ]; then
    echo "Processing debug protein..."
    python "$CODE_ROOT/scripts/prepare_debug_data.py"
fi

# --- NCCL (single-GPU, but keep for consistency) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# --- Hypothesis knobs --------------------------------------------------------
# Hypothesis: apply adp_wt = (loss_mf.detach()+eps)^norm_p (currently MF-only
# at model_trainer_base.py:252-259) to the chirality hinge too, keeping
# MF and chirality gradients on the same scale under norm_p > 0.
# Canonical test value is MEANFLOW_NORM_P=1; sweep {0, 2} off this same
# branch as norm_p invariance curve controls.
EXP_NAME="hyp-adp-wt-scaling"

# GUARDRAILS (spec §Guardrails, frozen @07e5b14):
ACCUMULATE_GRAD_BATCHES=1   # #1: removes silent no-op (see memory analysis)
EVAL_NSAMPLES=16            # #2: continuous chirality signal instead of 1-bit
DATAMODULE_REPEAT=2         # #4: unchanged (pinned)
DATAMODULE_BATCH_SIZE=2     # #4: unchanged (pinned)
MAX_EPOCHS=20000            # #4: unchanged (pinned)

# --- Chirality hinge loss (ON, canonical config) ---
CHIRALITY_ENABLED=True
CHIRALITY_WEIGHT=1.0
CHIRALITY_MARGIN_ALPHA=0.1
CHIRALITY_STRIDE=1

# --- MeanFlow adp_wt exponent (the hypothesis's core knob) ---
# Canonical test: norm_p=1. Sweep {0 (control), 2} as separate invocations
# to trace the norm_p invariance curve.
MEANFLOW_NORM_P=1

# --- Self-conditioning OFF (per pitch: isolate adp_wt/chirality coupling) ---
SELF_COND=False
SELF_COND_PROB=0.0

# --- Other baseline knobs (unchanged from train_debug.sbatch) ---
EVAL_EVERY_N_STEPS=250
OT_COUPLING_ENABLED=False
NCPUS_PER_TASK_TRAIN=16
NCPUS_PER_TASK_PREPRO=2
NOISE_SAMPLES=0
LOSS_ACCUMULATION_STEPS=1
WARMUP_STEPS=1000
LR_DECAY_TOTAL_STEPS=23000
INIT_LR=1e-6
END_LR=5e-6
LR=1.e-4

# --- Startup banner ----------------------------------------------------------
echo "==========================================="
echo "[hyp_adp_wt_scaling] Hypothesis knobs:"
echo "  MEANFLOW_NORM_P=$MEANFLOW_NORM_P (0=control, 1=canonical test, 2=stronger)"
echo "  CHIRALITY_ENABLED=$CHIRALITY_ENABLED weight=$CHIRALITY_WEIGHT"
echo "  CHIRALITY_MARGIN_ALPHA=$CHIRALITY_MARGIN_ALPHA stride=$CHIRALITY_STRIDE"
echo "  SELF_COND=$SELF_COND self_cond_prob=$SELF_COND_PROB"
echo "[guardrails] accum=$ACCUMULATE_GRAD_BATCHES  eval.nsamples=$EVAL_NSAMPLES"
echo "             repeat=$DATAMODULE_REPEAT  bs=$DATAMODULE_BATCH_SIZE  max_epochs=$MAX_EPOCHS"
echo "The TrainLoaderStatsCallback will print len(train_dataloader) and the"
echo "estimated per-epoch optimizer-step count at fit-start (guardrail #3)."
echo "==========================================="

# --- Run training ---
echo "Starting hyp_adp_wt_scaling training..."
srun python "$CODE_ROOT/proteinfoundation/train.py" \
    --config_name training_ca_debug \
    --single \
    --show_prog_bar \
    --exp_overrides \
        run_name_=$EXP_NAME \
        opt.lr=$LR \
        opt.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        opt.max_epochs=$MAX_EPOCHS \
        opt.warmup_steps=$WARMUP_STEPS \
        opt.lr_decay_total_steps=$LR_DECAY_TOTAL_STEPS \
        opt.init_lr=$INIT_LR \
        opt.end_lr=$END_LR \
        eval.every_n_steps=$EVAL_EVERY_N_STEPS \
        eval.nsamples=$EVAL_NSAMPLES \
        training.ot_coupling.enabled=$OT_COUPLING_ENABLED \
        training.meanflow.norm_p=$MEANFLOW_NORM_P \
        hardware.ncpus_per_task_train_=$NCPUS_PER_TASK_TRAIN \
        training.ot_coupling.noise_samples=$NOISE_SAMPLES \
        training.loss_accumulation_steps=$LOSS_ACCUMULATION_STEPS \
        hardware.ncpus_per_task_prepro_=$NCPUS_PER_TASK_PREPRO \
        training.self_cond=$SELF_COND \
        training.self_cond_prob=$SELF_COND_PROB \
        training.chirality_loss.enabled=$CHIRALITY_ENABLED \
        training.chirality_loss.weight=$CHIRALITY_WEIGHT \
        training.chirality_loss.margin_alpha=$CHIRALITY_MARGIN_ALPHA \
        training.chirality_loss.stride=$CHIRALITY_STRIDE \
    --data_overrides \
        datamodule.repeat=$DATAMODULE_REPEAT \
        datamodule.batch_size=$DATAMODULE_BATCH_SIZE

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
