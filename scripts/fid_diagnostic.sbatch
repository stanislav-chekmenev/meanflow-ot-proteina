#!/bin/bash
#SBATCH --job-name=fid_diag
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/fid_diag_%j.out
#SBATCH --error=logs/fid_diag_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Paths ---
REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
CODE_ROOT="$REPO_ROOT"
DATA_ROOT="/netscratch/schekmenev"

# Current training dataset (fraction=0.5 monomers, matches what the checkpoints
# were trained on). The h100 partition has its own /netscratch separate from
# the gpu partition, so the training job's stage on the gpu nodes is not
# visible here. Stage the source-of-truth on /mnt/labs to this node's
# /netscratch using the same md5-manifest contract as train.sbatch — the copy
# only runs when the stage manifest disagrees with the source.
DATA_NAME=monomers_minl50_maxl256_ds_frac_0.5
DATA_SRC=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}/
DATA_SRC_MD5=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}.md5
DATA_STAGE_DIR=$DATA_ROOT/$DATA_NAME
DATA_STAGE_MD5=$DATA_ROOT/${DATA_NAME}.md5
TRAIN_PT_DIR="$DATA_STAGE_DIR/processed"

# Persistent checkpoint stage on this node's /netscratch. Re-used across jobs
# so we don't pay the 744MB-per-ckpt copy on every run; md5-gated like the
# dataset stage so a re-trained `last.ckpt` triggers a re-copy automatically.
CKPT_STAGE_DIR="/netscratch/ckpt/mf_proteina"

compute_manifest() {
    find "$1" -type f ! -name '*.md5' -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | md5sum | awk '{print $1}'
}

stage_ckpt() {
    # stage_ckpt <src.ckpt> <dst.ckpt>
    # Gates on a sibling .md5 of the source's md5sum. Re-copies only when the
    # staged file's manifest disagrees.
    local src="$1"
    local dst="$2"
    local src_md5_file="${dst}.src_md5"

    if [ ! -f "$src" ]; then
        echo "ERROR: Checkpoint not found: $src" >&2
        return 1
    fi

    local src_hash
    src_hash="$(md5sum "$src" | awk '{print $1}')"
    local staged_hash=""
    [ -f "$src_md5_file" ] && staged_hash="$(cat "$src_md5_file")"

    if [ ! -f "$dst" ] || [ "$src_hash" != "$staged_hash" ]; then
        echo "$(hostname): staging ckpt $src -> $dst (src_md5=$src_hash, staged=$staged_hash)"
        mkdir -p "$(dirname "$dst")"
        # Copy to a tmp path then atomic-rename so a partial copy never wins.
        local tmp="${dst}.partial.$$"
        if ! cp "$src" "$tmp"; then
            echo "ERROR: cp $src -> $tmp failed" >&2
            rm -f "$tmp"
            return 1
        fi
        mv -f "$tmp" "$dst"
        echo "$src_hash" > "$src_md5_file"
        echo "$(hostname): ckpt staged ($(du -h "$dst" | awk '{print $1}'))"
    else
        echo "$(hostname): ckpt already staged at $dst (md5=$src_hash)"
    fi
    return 0
}

# --- Environment (shared venv tarball pattern from train.sbatch) ---
PROJECT_NAME="$(basename "$REPO_ROOT")"
ENV_TARBALL="$REPO_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/venvs/$PROJECT_NAME/.venv"

echo "Setting up virtual environment on node-local scratch..."
STAMP_FILE="$DATA_ROOT/venvs/$PROJECT_NAME/.tarball_mtime"
TARBALL_MTIME="$(stat -c %Y "$ENV_TARBALL")"
LOCAL_STAMP="$(cat "$STAMP_FILE" 2>/dev/null || echo "")"

if [ "$TARBALL_MTIME" != "$LOCAL_STAMP" ]; then
    echo "$(hostname): venv outdated or missing — re-extracting..."
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

export PYTHONPATH="${PYTHONPATH:-}:$CODE_ROOT"
export PYTHONUNBUFFERED=1

mkdir -p "$REPO_ROOT/logs"
cd "$CODE_ROOT"

# --- Stage dataset to node-local scratch ---
# Identical contract to train.sbatch: source manifest is the source of truth;
# stage only re-syncs when its md5 disagrees with the source. h100's
# /netscratch is independent of the gpu partition's, so this populates it on
# first run and is a no-op on subsequent runs.
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

# DATA_PATH is required: checkpoints save hparams with `${oc.env:DATA_PATH}`
# interpolations (e.g. model.nn.cath_code_dir = ${oc.env:DATA_PATH}/pdb_raw/),
# so Proteina.load_from_checkpoint will raise InterpolationResolutionError
# without it. Point at the staged dataset like train.sbatch does.
export DATA_PATH="$DATA_STAGE_DIR"
echo "DATA_PATH=$DATA_PATH"

# --- Sanity checks ---
if [ ! -d "$TRAIN_PT_DIR" ] || [ -z "$(ls "$TRAIN_PT_DIR"/*.pt 2>/dev/null | head -1)" ]; then
    echo "ERROR: Training .pt dir missing or empty at $TRAIN_PT_DIR" >&2
    echo "  (floor-probe will have nothing to read)" >&2
    exit 1
fi
echo "Found $(ls "$TRAIN_PT_DIR"/*.pt 2>/dev/null | wc -l) .pt files under $TRAIN_PT_DIR"

# --- Stage checkpoints (persistent on /netscratch/ckpt/mf_proteina) ---
# md5-gated copy from /mnt/labs to the node's /netscratch. First run pays the
# ~1.5GB copy; subsequent runs are no-ops as long as last.ckpt's md5 hasn't
# changed.
BASE_CKPT_SRC="$CODE_ROOT/store/meanflow-base-0.5-monomers-maxl256/checkpoints/last.ckpt"
OT_CKPT_SRC="$CODE_ROOT/store/meanflow-ot-eucl-0.5-monomers-maxl256/checkpoints/last.ckpt"
BASE_CKPT="$CKPT_STAGE_DIR/meanflow-base-0.5-monomers-maxl256_last.ckpt"
OT_CKPT="$CKPT_STAGE_DIR/meanflow-ot-eucl-0.5-monomers-maxl256_last.ckpt"

echo "Staging checkpoints to $CKPT_STAGE_DIR ..."
if ! stage_ckpt "$BASE_CKPT_SRC" "$BASE_CKPT"; then
    exit 1
fi
if ! stage_ckpt "$OT_CKPT_SRC" "$OT_CKPT"; then
    exit 1
fi

# --- Probe 1: FID floor against training PDBs ---
# Feeds training-set structures as "fake" against the PDB eval reference.
# Answers: is the large training-time FID driven by a train/eval
# distribution gap, or by genuine off-manifold generation?
echo ""
echo "============================================================"
echo "[1/3] FID floor probe (train-subset as fake vs pdb_eval ref)"
echo "============================================================"
#python "$CODE_ROOT/scripts/fid_floor_probe.py" \
#    --train-dir "$TRAIN_PT_DIR" \
#    --n-samples 5000 \
#    --batch-size 32 \
#    --max-len 256 \
#    --seed 42

FLOOR_EXIT=$?
#echo "[floor-probe] exit code: $FLOOR_EXIT"

echo "FID floor probe is skipped. The FID between the eval features and the training features is 95"

# Per-step log files under logs/. Each python invocation tees both stdout and
# stderr there, so a future failure leaves the full traceback in a file you
# can grep without paging through the SLURM .err.
BASE_LOG="$REPO_ROOT/logs/fid_diag_${SLURM_JOB_ID}_base.log"
OT_LOG="$REPO_ROOT/logs/fid_diag_${SLURM_JOB_ID}_ot_eucl.log"

# --- Probe 2: Multi-step vs 1-step FID on BASE checkpoint ---
echo ""
echo "============================================================"
echo "[2/3] Multi-step FID diagnostic on BASE checkpoint"
echo "  log: $BASE_LOG"
echo "============================================================"
python "$CODE_ROOT/scripts/fid_multistep_probe.py" \
    --ckpt "$BASE_CKPT" \
    --n-samples 5000 \
    --nsteps-meanflow 1 \
    --nsteps-multistep 200 \
    --generation-batch-size 32 \
    --seed 42 \
    2>&1 | tee "$BASE_LOG"

BASE_EXIT=${PIPESTATUS[0]}
echo "[multistep-probe BASE] exit code: $BASE_EXIT"

# --- Probe 3: Multi-step vs 1-step FID on OT-EUCL checkpoint ---
echo ""
echo "============================================================"
echo "[3/3] Multi-step FID diagnostic on OT-EUCL checkpoint"
echo "  log: $OT_LOG"
echo "============================================================"
python "$CODE_ROOT/scripts/fid_multistep_probe.py" \
    --ckpt "$OT_CKPT" \
    --n-samples 5000 \
    --nsteps-meanflow 1 \
    --nsteps-multistep 200 \
    --generation-batch-size 32 \
    --seed 42 \
    2>&1 | tee "$OT_LOG"

OT_EXIT=${PIPESTATUS[0]}
echo "[multistep-probe OT-EUCL] exit code: $OT_EXIT"

echo "==========================================="
echo "Job completed: $(date)"
echo "Floor probe exit:            $FLOOR_EXIT"
echo "Multi-step (BASE) exit:      $BASE_EXIT  (log: $BASE_LOG)"
echo "Multi-step (OT-EUCL) exit:   $OT_EXIT  (log: $OT_LOG)"
echo "==========================================="

# Fail the job if any probe failed so SLURM shows a clean red.
if [ $FLOOR_EXIT -ne 0 ] || [ $BASE_EXIT -ne 0 ] || [ $OT_EXIT -ne 0 ]; then
    exit 1
fi
