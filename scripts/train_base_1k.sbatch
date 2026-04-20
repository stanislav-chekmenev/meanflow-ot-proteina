#!/bin/bash
#SBATCH --job-name=mf_base_1k
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_base_1k_%j.out
#SBATCH --error=logs/train_base_1k_%j.err

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

# DATA_PATH env var — configs resolve ${oc.env:DATA_PATH}/pdb_train/
export DATA_PATH="$PROJECT_ROOT/data"

mkdir -p "$PROJECT_ROOT/logs"

# Dataset fraction — needed by preflight check below and by the training run
DATAMODULE_FRACTION=0.008

# --- Preflight: check that pdb_train data has been prepared ---
EXPECTED_CSV_GLOB="$DATA_PATH/pdb_train/df_pdb_f${DATAMODULE_FRACTION}_*.csv"
if ! ls $EXPECTED_CSV_GLOB 1> /dev/null 2>&1; then
    echo "ERROR: No prepped data CSV found matching $EXPECTED_CSV_GLOB" >&2
    echo "Run: PYTHONPATH=\$PROJECT_ROOT python \$PROJECT_ROOT/scripts/prepare_pdb_1k.py --fraction $DATAMODULE_FRACTION" >&2
    exit 1
fi
echo "Prepped data found: $(ls $EXPECTED_CSV_GLOB)"

# --- NCCL (single-GPU, but keep for consistency) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# --- Configurable Hydra overrides ---
# Experiment config (training_ca_debug.yaml)
EXP_NAME="meanflow-base-1k-p0"
ACCUMULATE_GRAD_BATCHES=3
VAL_CHECK_INTERVAL=1000
OT_COUPLING_ENABLED=False
MEANFLOW_NORM_P=0
NCPUS_PER_TASK_TRAIN=16
NCPUS_PER_TASK_PREPRO=2
NOISE_SAMPLES=0
LOSS_ACCUMULATION_STEPS=1
MAX_EPOCHS=1000
WARMUP_STEPS=2000
LR_DECAY_TOTAL_STEPS=42000
INIT_LR=0.0
END_LR=1e-5
LR=1e-4

# Self-conditioning
SELF_COND=False
SELF_COND_PROB=0.5

# Chirality hinge loss
CHIRALITY_ENABLED=False
CHIRALITY_WEIGHT=1.0
CHIRALITY_MARGIN_ALPHA=0.1
CHIRALITY_STRIDE=1

# Dataset config
DATAMODULE_BATCH_SIZE=4
DATAMODULE_NUM_WORKERS=16

# --- Run training ---
echo "Starting MeanFlow base-1k training..."
cd "$PROJECT_ROOT"
srun python "$PROJECT_ROOT/proteinfoundation/train.py" \
    --config_name training_ca_debug \
    --single \
    --show_prog_bar \
    --exp_overrides \
        run_name_=$EXP_NAME \
        dataset=pdb_train \
        opt.lr=$LR \
        opt.init_lr=$INIT_LR \
        opt.end_lr=$END_LR \
        opt.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        opt.max_epochs=$MAX_EPOCHS \
        opt.warmup_steps=$WARMUP_STEPS \
        opt.lr_decay_total_steps=$LR_DECAY_TOTAL_STEPS \
        opt.val_check_interval=$VAL_CHECK_INTERVAL \
        eval.enabled=False \
        eval.val_enabled=True \
        eval.n_val_proteins=16 \
        training.ot_coupling.enabled=$OT_COUPLING_ENABLED \
        training.ot_coupling.noise_samples=$NOISE_SAMPLES \
        training.meanflow.norm_p=$MEANFLOW_NORM_P \
        training.loss_accumulation_steps=$LOSS_ACCUMULATION_STEPS \
        training.self_cond=$SELF_COND \
        training.self_cond_prob=$SELF_COND_PROB \
        training.chirality_loss.enabled=$CHIRALITY_ENABLED \
        training.chirality_loss.weight=$CHIRALITY_WEIGHT \
        training.chirality_loss.margin_alpha=$CHIRALITY_MARGIN_ALPHA \
        training.chirality_loss.stride=$CHIRALITY_STRIDE \
        hardware.ncpus_per_task_train_=$NCPUS_PER_TASK_TRAIN \
        hardware.ncpus_per_task_prepro_=$NCPUS_PER_TASK_PREPRO \
        log.checkpoint=True \
        log.checkpoint_every_n_steps=5000 \
        log.last_ckpt_every_n_steps=1000 \
    --data_overrides \
        datamodule.batch_size=$DATAMODULE_BATCH_SIZE \
        datamodule.dataselector.fraction=$DATAMODULE_FRACTION \
        datamodule.num_workers=$DATAMODULE_NUM_WORKERS

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
