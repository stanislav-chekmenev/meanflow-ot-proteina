#!/bin/bash
#SBATCH --job-name=mf_debug
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/train_debug_%j.out
#SBATCH --error=logs/train_debug_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
DATA_ROOT="/netscratch/schekmenev"

# Unpack .venv to scratch (once per node, skips if already present)
# Create tarball on login node:
#   uv venv --relocatable --python 3.11 && uv sync
#   tar czf "$PROJECT_ROOT/venv.tar.gz" -C "$PROJECT_ROOT" .venv
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
ENV_TARBALL="$PROJECT_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/venvs/$PROJECT_NAME/.venv"    

echo "Setting up virtual environment on node-local scratch..."
# Version check: re-extract only when tarball has changed (compare mtime stamp)
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

# Ensure project is importable
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# DATA_PATH env var — configs resolve ${oc.env:DATA_PATH}/debug/
export DATA_PATH="$PROJECT_ROOT/data"

mkdir -p "$PROJECT_ROOT/logs"

# --- Prepare debug data (if not already done) ---
if [ ! -f "$DATA_PATH/debug/processed/1ubq.pt" ]; then
    echo "Processing debug protein..."
    python "$PROJECT_ROOT/scripts/prepare_debug_data.py"
fi

# --- NCCL (single-GPU, but keep for consistency) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# --- Configurable Hydra overrides ---
# Experiment config (training_ca_debug.yaml)
EXP_NAME="meanflow-ot-bs2"
ACCUMULATE_GRAD_BATCHES=16
EVAL_EVERY_N_STEPS=100
OT_COUPLING_ENABLED=True
MEANFLOW_NORM_P=1.0
NCPUS_PER_TASK_TRAIN=2
NCPUS_PER_TASK_PREPRO=2

# Dataset config (pdb_debug.yaml)
DATAMODULE_REPEAT=2
DATAMODULE_BATCH_SIZE=2

# --- Run training ---
echo "Starting MeanFlow debug training..."
srun python "$PROJECT_ROOT/proteinfoundation/train.py" \
    --config_name training_ca_debug \
    --single \
    --show_prog_bar \
    --exp_overrides \
        run_name_=$EXP_NAME \
        opt.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        eval.every_n_steps=$EVAL_EVERY_N_STEPS \
        training.ot_coupling.enabled=$OT_COUPLING_ENABLED \
        training.meanflow.norm_p=$MEANFLOW_NORM_P \
        hardware.ncpus_per_task_train_=$NCPUS_PER_TASK_TRAIN \
        hardware.ncpus_per_task_prepro_=$NCPUS_PER_TASK_PREPRO \
    --data_overrides \
        datamodule.repeat=$DATAMODULE_REPEAT \
        datamodule.batch_size=$DATAMODULE_BATCH_SIZE

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
