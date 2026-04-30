#!/bin/bash
#SBATCH --job-name=dump_fid_ref
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=logs/dump_fid_ref_%j.out
#SBATCH --error=logs/dump_fid_ref_%j.err

set -euo pipefail

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Paths ---
REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
DATA_ROOT="/netscratch/schekmenev"

# Source dataset (preprocessed monomers, fraction=0.1) and node-local stage.
# Same md5-manifest contract as train.sbatch / fid_diagnostic.sbatch: stage
# only re-syncs when the source manifest disagrees.
DATA_NAME=monomers_minl50_maxl256_ds_frac_0.1
DATA_SRC=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}/
DATA_SRC_MD5=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}.md5
DATA_STAGE_DIR=$DATA_ROOT/$DATA_NAME
DATA_STAGE_MD5=$DATA_ROOT/${DATA_NAME}.md5
PT_DIR="$DATA_STAGE_DIR/processed"

# Output: drop the new reference next to the existing pdb_eval_ca_features.pth
# so fid.real_features_path can point at it with a short relative path.
OUTPUT_PATH="$REPO_ROOT/data/metric_factory/features/pdb_eval_ca_features_monomers_frac0.1.pth"

compute_manifest() {
    find "$1" -type f ! -name '*.md5' -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | md5sum | awk '{print $1}'
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

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"
export PYTHONUNBUFFERED=1

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

# --- Stage dataset to node-local /netscratch ---
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

# --- Sanity check ---
if [ ! -d "$PT_DIR" ] || [ -z "$(ls "$PT_DIR"/*.pt 2>/dev/null | head -1)" ]; then
    echo "ERROR: .pt dir missing or empty at $PT_DIR" >&2
    exit 1
fi
N_PT=$(ls "$PT_DIR"/*.pt 2>/dev/null | wc -l)
echo "Found $N_PT .pt files under $PT_DIR"

# --- Run the dump ---
DUMP_LOG="$REPO_ROOT/logs/dump_fid_ref_${SLURM_JOB_ID}.log"
echo ""
echo "============================================================"
echo "Dumping FID reference features"
echo "  data: $PT_DIR"
echo "  output: $OUTPUT_PATH"
echo "  log: $DUMP_LOG"
echo "============================================================"

python "$REPO_ROOT/scripts/dump_pdb_eval_ca_features.py" \
    --data-dir "$PT_DIR" \
    --output "$OUTPUT_PATH" \
    --batch-size 16 \
    --max-len 256 \
    2>&1 | tee "$DUMP_LOG"

DUMP_EXIT=${PIPESTATUS[0]}
echo "[dump] exit code: $DUMP_EXIT"

echo "==========================================="
echo "Job completed: $(date)"
echo "Dump exit:   $DUMP_EXIT  (log: $DUMP_LOG)"
echo "Output file: $OUTPUT_PATH"
[ -f "$OUTPUT_PATH" ] && echo "Output size: $(du -h "$OUTPUT_PATH" | awk '{print $1}')"
echo "==========================================="

exit $DUMP_EXIT
