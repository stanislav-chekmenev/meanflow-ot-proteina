#!/bin/bash
#SBATCH --job-name=mf_debug
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-01:00:00
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
if [ ! -d "$ENV_LOCAL/bin" ]; then
    mkdir -p "$DATA_ROOT/venvs/$PROJECT_NAME"
    if ! tar xzf "$ENV_TARBALL" -C "$DATA_ROOT/venvs/$PROJECT_NAME"; then
        echo "ERROR: Failed to unpack $ENV_TARBALL" >&2
        exit 1
    fi
    echo "$(hostname): env unpacked to $ENV_LOCAL"
else
    echo "$(hostname): env already at $ENV_LOCAL"
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

# --- Run training ---
# MeanFlow debug: 1 protein (1UBQ, 76 residues), 1 GPU, no wandb/checkpointing.
#
# Reference: CIFAR10 MeanFlow v1 uses torchrun with:
#   --ratio 0.75  --P_mean_t -0.6  --P_std_t 1.6
#   --P_mean_r -4.0  --P_std_r 1.6  --norm_p 0.75
# Current proteina config uses shared P_mean/P_std for t and r.
# The values can be tuned in training_ca_debug.yaml.
echo "Starting MeanFlow debug training..."
srun python "$PROJECT_ROOT/proteinfoundation/train.py" \
    --config_name training_ca_debug \
    --nolog \
    --single \
    --show_prog_bar

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
