#!/bin/bash
#SBATCH --job-name=mf_ot
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/train_ot_%j.out
#SBATCH --error=logs/train_ot_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
# REPO_ROOT is the primary repo (shared venv tarball, logs).
# CODE_ROOT is this worktree — configs and code come from here so schema
# changes in the worktree take effect.
REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
CODE_ROOT="$REPO_ROOT"
DATA_ROOT="/netscratch/schekmenev"

# --- Data staging config ---
# Dataset dir name doubles as the scratch stage name + manifest basename.
DATA_NAME=monomers_minl50_maxl256_ds_frac_0.5
DATA_SRC=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}/
DATA_SRC_MD5=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}.md5
DATA_STAGE_DIR=/netscratch/schekmenev/${DATA_NAME}
DATA_STAGE_MD5=/netscratch/schekmenev/${DATA_NAME}.md5
FID=df_pdb_f0.5_minl50_maxl256_mtprotein_etdiffractionEM_mino1_maxo1_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse
SEQID=0.5

compute_manifest() {
    find "$1" -type f ! -name '*.md5' -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | md5sum | awk '{print $1}'
}

# Unpack .venv to scratch (once per node, skips if already present)
# Create tarball on login node:
#   uv venv --relocatable --python 3.11 && uv sync
#   tar czf "$REPO_ROOT/venv.tar.gz" -C "$REPO_ROOT" .venv
PROJECT_NAME="$(basename "$REPO_ROOT")"
ENV_TARBALL="$REPO_ROOT/venv.tar.gz"
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

# --- Stage dataset to node-local scratch ---
mkdir -p "$(dirname "$DATA_STAGE_DIR")"

SRC_HASH=$(compute_manifest "$DATA_SRC")
SRC_HASH_FILE=$(cat "$DATA_SRC_MD5" 2>/dev/null || echo "")
if [ "$SRC_HASH_FILE" != "$SRC_HASH" ]; then
    if ! echo "$SRC_HASH" > "$DATA_SRC_MD5" 2>/dev/null; then
        echo "WARN: could not update source manifest $DATA_SRC_MD5"
    fi
fi

STAGE_HASH_FILE=$(cat "$DATA_STAGE_MD5" 2>/dev/null || echo "")
if [ ! -d "$DATA_STAGE_DIR" ] || [ "$STAGE_HASH_FILE" != "$SRC_HASH" ]; then
    echo "$(hostname): dataset stage outdated or missing (src=$SRC_HASH, stage=$STAGE_HASH_FILE) — syncing..."
    # rsync isn't installed on every compute node (e.g. gpu01, gpu02). Use it
    # when present for delta updates; otherwise wipe + cp -a to get the same
    # "stage exactly matches source" semantics as `rsync -a --delete`.
    if command -v rsync >/dev/null 2>&1; then
        mkdir -p "$DATA_STAGE_DIR"
        if ! rsync -a --delete "$DATA_SRC"/ "$DATA_STAGE_DIR"/; then
            echo "ERROR: rsync failed" >&2
            exit 1
        fi
    else
        echo "$(hostname): rsync not available — falling back to cp -a"
        rm -rf "$DATA_STAGE_DIR"
        mkdir -p "$DATA_STAGE_DIR"
        if ! cp -a "$DATA_SRC"/. "$DATA_STAGE_DIR"/; then
            echo "ERROR: cp -a failed" >&2
            exit 1
        fi
    fi
    echo "$SRC_HASH" > "$DATA_STAGE_MD5"
    echo "$(hostname): staged $(find "$DATA_STAGE_DIR" -type f | wc -l) files to $DATA_STAGE_DIR"
else
    echo "$(hostname): dataset already staged and up-to-date (md5=$SRC_HASH)"
fi

CLUST_FA="$DATA_STAGE_DIR/cluster_seqid_${SEQID}_${FID}.fasta"
CLUST_TSV="$DATA_STAGE_DIR/cluster_seqid_${SEQID}_${FID}.tsv"
SEQ_FA="$DATA_STAGE_DIR/seq_${FID}.fasta"

if [ ! -f "$CLUST_FA" ] || [ ! -f "$CLUST_TSV" ]; then
    echo "ERROR: cluster files missing on stage — pre-generate them on a node with mmseqs and place them in $DATA_SRC" >&2
    echo "  expected: $CLUST_FA" >&2
    echo "  expected: $CLUST_TSV" >&2
    exit 1
fi
echo "Cluster files present — skipping mmseqs."

# Ensure project is importable — use CODE_ROOT so imports resolve to this
# worktree rather than the main repo.
export PYTHONPATH="${PYTHONPATH:-}:$CODE_ROOT"
export PYTHONUNBUFFERED=1

# DATA_PATH env var — configs resolve ${oc.env:DATA_PATH}/ directly
export DATA_PATH="$DATA_STAGE_DIR"

mkdir -p "$REPO_ROOT/logs"

# Dataset fraction — needed by preflight check below and by the training run
DATAMODULE_FRACTION=0.5

# --- Preflight: check that the dataset has been prepared ---
EXPECTED_CSV_GLOB="$DATA_STAGE_DIR/df_pdb_f${DATAMODULE_FRACTION}_*.csv"
if ! ls $EXPECTED_CSV_GLOB 1> /dev/null 2>&1; then
    echo "ERROR: No prepped data CSV found matching $EXPECTED_CSV_GLOB" >&2
    echo "Run: PYTHONPATH=\$CODE_ROOT python \$CODE_ROOT/scripts/prepare_pdb.py --fraction $DATAMODULE_FRACTION" >&2
    exit 1
fi
echo "Prepped data found: $(ls $EXPECTED_CSV_GLOB)"

# --- NCCL (multi-GPU DDP) ---
# h100 partition: no node has working InfiniBand — always disable IB/P2P.
# gpu partition: gpu01 and gpu02 have no working InfiniBand (confirmed by
# first-collective broadcast hangs on gpu01, job 39414). gpu03 works.
# NCCL_IB_DISABLE=1 prevents NCCL from even probing libibverbs.
export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=0
NODES_WITHOUT_IB="gpu01 gpu02"
if [ "$SLURM_JOB_PARTITION" = "h100" ]; then
    echo "$(hostname): h100 partition has no InfiniBand — disabling NCCL IB"
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
elif echo " $NODES_WITHOUT_IB " | grep -q " $(hostname) "; then
    echo "$(hostname): no InfiniBand on this node — disabling NCCL IB"
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
else
    export NCCL_IB_DISABLE=0
    export NCCL_P2P_DISABLE=0
fi

# --- h100 partition: redirect ckpts to /netscratch + stage GearNet weights ---
# h100 has its own /netscratch separate from the gpu partition's. Checkpoints
# live there to avoid filling the project directory; GearNet weights (76 MB,
# read every FID firing) are staged once for faster loading.
CKPT_ROOT_DIR=""
GEARNET_CKPT_PATH=""
if [ "$SLURM_JOB_PARTITION" = "h100" ]; then
    CKPT_ROOT_DIR="/netscratch/schekmenev/ckpt/mf_proteina"
    mkdir -p "$CKPT_ROOT_DIR"

    GEARNET_SRC="$CODE_ROOT/data/metric_factory/model_weights/gearnet_ca.pth"
    GEARNET_STAGE="$CKPT_ROOT_DIR/gearnet_ca.pth"
    if [ ! -f "$GEARNET_SRC" ]; then
        echo "ERROR: GearNet weights missing at $GEARNET_SRC" >&2
        exit 1
    fi
    SRC_SIZE=$(stat -c %s "$GEARNET_SRC")
    STAGE_SIZE=$(stat -c %s "$GEARNET_STAGE" 2>/dev/null || echo "")
    if [ "$SRC_SIZE" != "$STAGE_SIZE" ]; then
        echo "$(hostname): staging GearNet weights -> $GEARNET_STAGE"
        cp -f "$GEARNET_SRC" "$GEARNET_STAGE"
    else
        echo "$(hostname): GearNet weights already staged at $GEARNET_STAGE"
    fi
    GEARNET_CKPT_PATH="$GEARNET_STAGE"
fi

# Controls how long PyTorch waits for other processes to join the group
# Useful if one node fails early and the other is just hanging.
export TORCH_NCCL_BLOCKING_WAIT=1  # Forces synchronous error reporting
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # Enable async error handling

# Controls how long NCCL waits for the initial handshake between nodes
# Default is 15-30 minutes. Set to 60 seconds for debugging.
export NCCL_COMM_ID_TIMEOUT=60

# --- Configurable Hydra overrides ---
# Experiment config (training_ca.yaml)
EXP_NAME="mf-ot-eucl-loss-accum4"
ACCUMULATE_GRAD_BATCHES=4
# VAL_CHECK_INTERVAL is in OPTIMIZER STEPS. Lightning's val_check_interval is
# measured in dataloader batches, so we multiply by ACCUMULATE_GRAD_BATCHES
# when passing it as the opt.val_check_interval override below.
VAL_CHECK_INTERVAL=5000
OT_COUPLING_ENABLED=True
MEANFLOW_NORM_P=1
OT_POOL_SIZE=256
LOSS_ACCUMULATION_STEPS=4
MAX_EPOCHS=2500
WARMUP_STEPS=500
LR_DECAY_TOTAL_STEPS=100000
INIT_LR=1e-6
END_LR=1e-5
LR=1e-4

# Checkpointing
# True: resume from store/<run>/checkpoints/last.ckpt if it exists.
# False: ignore any existing ckpt and start fresh (old files on disk are NOT removed;
# rm them manually before the run if you want a clean store/<run>/checkpoints/ dir).
RESUME_FROM_LAST_CKPT=False

# Eval (SamplesLoggingCallback — see eval block in training_ca.yaml)
EVAL_ENABLED=True
EVAL_N_SAMPLES=8
EVAL_LENGTHS="[64,128,192,256]"

# Self-conditioning
SELF_COND=True
SELF_COND_PROB=0.5

# Dataset config
DATAMODULE_BATCH_SIZE=4
DATAMODULE_NUM_WORKERS=16

#Hardware config (also set via sbatch args)
NCPUS_PER_TASK_TRAIN=16
NCPUS_PER_TASK_PREPRO=2
NGPUS_PER_NODE=2
NNODES=1

# --- Run training ---
# cd into CODE_ROOT so Python's implicit sys.path[0] (cwd) matches the
# worktree, ensuring configs/ and proteinfoundation/ are loaded from here.
echo "Starting MeanFlow training..."
cd "$CODE_ROOT"
# --kill-on-bad-exit=1: if any rank exits non-zero, kill the whole step.
# Without this srun leaves surviving ranks hanging (typically stuck in an
# NCCL collective) until the job wall-time, observed as a 9h hang in job
# 40785 where rank 0 crashed on a ModelCheckpoint MisconfigurationException
# and rank 1 kept the allocation alive.
VAL_CHECK_INTERVAL_BATCHES=$(( VAL_CHECK_INTERVAL * ACCUMULATE_GRAD_BATCHES ))

# Build h100-only Hydra overrides (empty array on other partitions).
H100_OVERRIDES=()
if [ -n "$CKPT_ROOT_DIR" ]; then
    H100_OVERRIDES+=("log.ckpt_root_dir=$CKPT_ROOT_DIR")
fi
if [ -n "$GEARNET_CKPT_PATH" ]; then
    H100_OVERRIDES+=("fid.gearnet_ckpt_path=$GEARNET_CKPT_PATH")
fi

srun --kill-on-bad-exit=1 python "$CODE_ROOT/proteinfoundation/train.py" \
    --config_name training_ca \
    --show_prog_bar \
    --exp_overrides \
        run_name_=$EXP_NAME \
        hardware.ngpus_per_node_=$NGPUS_PER_NODE \
        hardware.nnodes_=$NNODES \
        dataset=pdb_train \
        opt.lr=$LR \
        opt.init_lr=$INIT_LR \
        opt.end_lr=$END_LR \
        opt.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        opt.max_epochs=$MAX_EPOCHS \
        opt.warmup_steps=$WARMUP_STEPS \
        opt.lr_decay_total_steps=$LR_DECAY_TOTAL_STEPS \
        opt.val_check_interval=$VAL_CHECK_INTERVAL_BATCHES \
        log.resume_from_last_ckpt=$RESUME_FROM_LAST_CKPT \
        eval.enabled=$EVAL_ENABLED \
        training.ot_coupling.enabled=$OT_COUPLING_ENABLED \
        training.ot_coupling.ot_pool_size=$OT_POOL_SIZE \
        training.meanflow.norm_p=$MEANFLOW_NORM_P \
        training.loss_accumulation_steps=$LOSS_ACCUMULATION_STEPS \
        training.self_cond=$SELF_COND \
        training.self_cond_prob=$SELF_COND_PROB \
        hardware.ncpus_per_task_train_=$NCPUS_PER_TASK_TRAIN \
        hardware.ncpus_per_task_prepro_=$NCPUS_PER_TASK_PREPRO \
        "${H100_OVERRIDES[@]}" \
    --data_overrides \
        datamodule.batch_size=$DATAMODULE_BATCH_SIZE \
        datamodule.dataselector.fraction=$DATAMODULE_FRACTION \
        datamodule.num_workers=$DATAMODULE_NUM_WORKERS

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
