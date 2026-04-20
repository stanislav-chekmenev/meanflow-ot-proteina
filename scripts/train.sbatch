#!/bin/bash
#SBATCH --job-name=mf_base
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_base_%j.out
#SBATCH --error=logs/train_base_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
DATA_ROOT="/netscratch/schekmenev"

# --- Data staging config ---
DATA_SRC=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina_ds
DATA_SRC_MD5=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina_ds.md5
DATA_STAGE_ROOT=/netscratch/schekmenev/ot_mf_proteina_ds
DATA_STAGE_DIR=$DATA_STAGE_ROOT/pdb_train
DATA_STAGE_MD5=/netscratch/schekmenev/ot_mf_proteina_ds.md5
FID=df_pdb_f0.008_minl50_maxl256_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse
SEQID=0.5

compute_manifest() {
    find "$1" -type f ! -name '*.md5' -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | md5sum | awk '{print $1}'
}

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

# --- Stage dataset to node-local scratch ---
mkdir -p "$(dirname "$DATA_STAGE_ROOT")"
mkdir -p "$DATA_STAGE_ROOT"

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
    mkdir -p "$DATA_STAGE_DIR"
    if ! rsync -a --delete "$DATA_SRC"/ "$DATA_STAGE_DIR"/; then
        echo "ERROR: rsync failed" >&2
        exit 1
    fi
    echo "$SRC_HASH" > "$DATA_STAGE_MD5"
    echo "$(hostname): staged $(find "$DATA_STAGE_DIR" -type f | wc -l) files to $DATA_STAGE_DIR"
else
    echo "$(hostname): dataset already staged and up-to-date (md5=$SRC_HASH)"
fi

CLUST_FA="$DATA_STAGE_DIR/cluster_seqid_${SEQID}_${FID}.fasta"
CLUST_TSV="$DATA_STAGE_DIR/cluster_seqid_${SEQID}_${FID}.tsv"
SEQ_FA="$DATA_STAGE_DIR/seq_${FID}.fasta"

if [ -f "$CLUST_FA" ] && [ -f "$CLUST_TSV" ]; then
    echo "Cluster files present — skipping mmseqs."
else
    echo "Cluster files missing — running mmseqs easy-cluster..."
    if ! command -v mmseqs >/dev/null 2>&1; then
        echo "ERROR: mmseqs not found on PATH" >&2
        exit 1
    fi
    if [ ! -f "$SEQ_FA" ]; then
        echo "ERROR: input fasta missing: $SEQ_FA" >&2
        exit 1
    fi
    MMSEQS_WORK=$(mktemp -d -t mmseqs.XXXXXX)
    trap 'rm -rf "$MMSEQS_WORK"' EXIT
    pushd "$MMSEQS_WORK" >/dev/null
    if ! mmseqs easy-cluster "$SEQ_FA" pdb_cluster tmp --min-seq-id $SEQID -c 0.8 --cov-mode 1; then
        echo "ERROR: mmseqs easy-cluster failed" >&2
        popd >/dev/null
        exit 1
    fi
    if ! mv pdb_cluster_rep_seq.fasta "$CLUST_FA"; then
        echo "ERROR: failed to move cluster rep fasta to $CLUST_FA" >&2
        popd >/dev/null
        exit 1
    fi
    if ! mv pdb_cluster_cluster.tsv "$CLUST_TSV"; then
        echo "ERROR: failed to move cluster tsv to $CLUST_TSV" >&2
        popd >/dev/null
        exit 1
    fi
    popd >/dev/null
    echo "mmseqs clustering complete."

    # Copy back to source-of-truth so future jobs skip mmseqs
    COPYBACK_OK=1
    if ! cp -f "$CLUST_FA" "$CLUST_TSV" "$DATA_SRC/"; then
        echo "WARN: could not copy cluster files back to $DATA_SRC"
        COPYBACK_OK=0
    fi

    # Refresh manifests to reflect new cluster files.
    # If copy-back succeeded, source and stage are now bit-equivalent; use the
    # source hash for both. If copy-back failed, the stage has cluster files that
    # the source lacks — still write the SOURCE hash to the stage manifest so the
    # next job's stage-vs-source comparison matches and rsync is skipped (which
    # would otherwise --delete the staged cluster files). The cluster check below
    # will then re-run mmseqs next time, which is wasteful but non-destructive.
    if [ "$COPYBACK_OK" = "1" ]; then
        SRC_HASH=$(compute_manifest "$DATA_SRC")
        if ! echo "$SRC_HASH" > "$DATA_SRC_MD5" 2>/dev/null; then
            echo "WARN: could not update source manifest $DATA_SRC_MD5"
        fi
    fi
    echo "$SRC_HASH" > "$DATA_STAGE_MD5"
fi

# Ensure project is importable
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# DATA_PATH env var — configs resolve ${oc.env:DATA_PATH}/pdb_train/
export DATA_PATH="$DATA_STAGE_ROOT"

mkdir -p "$PROJECT_ROOT/logs"

# Dataset fraction — needed by preflight check below and by the training run
DATAMODULE_FRACTION=0.008

# --- Preflight: check that pdb_train data has been prepared ---
EXPECTED_CSV_GLOB="$DATA_STAGE_DIR/df_pdb_f${DATAMODULE_FRACTION}_*.csv"
if ! ls $EXPECTED_CSV_GLOB 1> /dev/null 2>&1; then
    echo "ERROR: No prepped data CSV found matching $EXPECTED_CSV_GLOB" >&2
    echo "Run: PYTHONPATH=\$PROJECT_ROOT python \$PROJECT_ROOT/scripts/prepare_pdb.py --fraction $DATAMODULE_FRACTION" >&2
    exit 1
fi
echo "Prepped data found: $(ls $EXPECTED_CSV_GLOB)"

# --- NCCL (single-GPU, but keep for consistency) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# --- Configurable Hydra overrides ---
# Experiment config (training_ca.yaml)
EXP_NAME="meanflow-base-p0"
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
INIT_LR=1e-6
END_LR=1e-5
LR=1e-4

# Eval
EVAL_ENABLED=True
EVAL_N_VAL_PROTEINS=16

# Self-conditioning
SELF_COND=True
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
echo "Starting MeanFlow base training..."
cd "$PROJECT_ROOT"
srun python "$PROJECT_ROOT/proteinfoundation/train.py" \
    --config_name training_ca \
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
        eval.val_enabled=$EVAL_ENABLED \
        eval.n_val_proteins=$EVAL_N_VAL_PROTEINS \
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
    --data_overrides \
        datamodule.batch_size=$DATAMODULE_BATCH_SIZE \
        datamodule.dataselector.fraction=$DATAMODULE_FRACTION \
        datamodule.num_workers=$DATAMODULE_NUM_WORKERS

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
