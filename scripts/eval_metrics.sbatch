#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=spot
#SBATCH --qos=spot
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_metrics_%j.out
#SBATCH --error=logs/eval_metrics_%j.err

set -euo pipefail

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-?}"
echo "==========================================="

# --- Paths ---
REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
DATA_ROOT="/netscratch/schekmenev"

# Checkpoint to evaluate. OVERRIDE this to point at the run/ckpt you want.
# Training writes ckpts to node-local /netscratch on the h100 partition, which
# is invisible to the spot/gpu partitions where eval typically runs. Stage the
# desired ckpt to shared FS first (manual cp from the h100 node), then point
# this path at the staged copy.
CKPT_PATH="/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/ckpts/mf-base-frac0.5/checkpoints/last-EMA.ckpt"

# Output directory (created if missing). Default: under node-local /netscratch
# so generated PDBs + ProteinMPNN/ESMFold scratch don't hit the slow shared FS.
# Override with EVAL_OUTPUT_DIR=... before sbatch if needed.
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$DATA_ROOT/mf_proteina_inference/eval_metrics_${SLURM_JOB_ID}}"

# Sampling config. Override with environment variables (EVAL_LENGTHS, etc.)
# at sbatch time if you want a different grid.
EVAL_LENGTHS="${EVAL_LENGTHS:-50,100,150,200,250}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_MAX_NSAMPLES_PER_BATCH="${EVAL_MAX_NSAMPLES_PER_BATCH:-8}"
# MeanFlow multi-step sampling. Override with EVAL_NSTEPS=1 for single-step.
EVAL_NSTEPS="${EVAL_NSTEPS:-1}"
EVAL_SEED="${EVAL_SEED:-5}"
EVAL_DESIGNABILITY_THRESHOLD="${EVAL_DESIGNABILITY_THRESHOLD:-2.0}"

# If set non-empty, evaluate the EMA-averaged weights snapshot instead of the
# online ones. The EMA callback writes a sibling file `<name>-EMA.ckpt` next
# to every checkpoint; we just rewrite CKPT_PATH to point at that sibling.
# Loading an `-EMA` ckpt directly is the supported inference path
# (proteinfoundation/utils/ema_utils/ema_callback.py:148).
EVAL_USE_EMA="${EVAL_USE_EMA:-1}"

# Skip flags (set non-empty to skip). Default: run everything.
EVAL_SKIP_DESIGNABILITY="${EVAL_SKIP_DESIGNABILITY:-}"
EVAL_SKIP_FID="${EVAL_SKIP_FID:-}"
EVAL_SKIP_TM_DIVERSITY="${EVAL_SKIP_TM_DIVERSITY:-}"
EVAL_SKIP_CLUSTER_DIVERSITY="${EVAL_SKIP_CLUSTER_DIVERSITY:-}"
EVAL_QUICK_TEST="${EVAL_QUICK_TEST:-}"

# Bundled binaries — pre-staged on every compute partition's local /netscratch
# at /netscratch/schekmenev/protein_utils/{foldseek,tmalign}/...
PROTEIN_UTILS_DIR="$DATA_ROOT/protein_utils"
FOLDSEEK_BIN="$PROTEIN_UTILS_DIR/foldseek/bin/foldseek"
TMALIGN_BIN="$PROTEIN_UTILS_DIR/tmalign/TMalign"

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

# wandb's service subprocess polls for a port file with a 30s default timeout.
# On slow shared FS the poll can exceed 30s and crash wandb.init. Bump to 5
# minutes — same as train.sbatch (note double underscore).
export WANDB__SERVICE_WAIT=300

# DATA_PATH is required for `Proteina.load_from_checkpoint` to resolve
# `${oc.env:DATA_PATH}` references baked into the saved hparams. The model
# config sets `cath_code_dir: ${DATA_PATH}/pdb_raw/`, and ckpt load reads
# `pdb_raw/cath_label_mapping.pt` from that path. So we need the dataset
# staged on this node's /netscratch — same pattern as train.sbatch.
DATA_NAME=monomers_minl50_maxl256_ds_frac_0.1
DATA_SRC=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}/
DATA_SRC_MD5=/mnt/labs/data/bronstein/schekmenev/ot_mf_proteina/${DATA_NAME}.md5
DATA_STAGE_DIR=/netscratch/schekmenev/${DATA_NAME}
DATA_STAGE_MD5=/netscratch/schekmenev/${DATA_NAME}.md5

compute_manifest() {
    find "$1" -type f ! -name '*.md5' -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | md5sum | awk '{print $1}'
}

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

export DATA_PATH="$DATA_STAGE_DIR"

# --- Stage ProteinMPNN CA weights to /netscratch ---
# scRMSD spawns one ProteinMPNN python process per sample (`os.system` in
# proteinfoundation/metrics/designability.py:run_proteinmpnn). Each process
# re-reads the 6.3 MB weights file. Reading hundreds of times from shared FS
# is wasteful; stage once to /netscratch and pass the path via PMPNN_WEIGHTS_DIR
# (consumed by run_proteinmpnn → --path_to_model_weights).
PMPNN_SRC_DIR="$REPO_ROOT/ProteinMPNN/ca_model_weights"
PMPNN_STAGE_DIR="/netscratch/schekmenev/protein_utils/proteinmpnn/ca_model_weights"
if [ ! -s "$PMPNN_SRC_DIR/v_48_020.pt" ]; then
    echo "ERROR: ProteinMPNN CA weights missing at source: $PMPNN_SRC_DIR/v_48_020.pt" >&2
    echo "  Download from upstream:" >&2
    echo "    curl -fsSL -o $PMPNN_SRC_DIR/v_48_020.pt https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/ca_model_weights/v_48_020.pt" >&2
    exit 1
fi
mkdir -p "$PMPNN_STAGE_DIR"
for w in v_48_002.pt v_48_010.pt v_48_020.pt; do
    [ -f "$PMPNN_SRC_DIR/$w" ] || continue
    SRC_SIZE=$(stat -c %s "$PMPNN_SRC_DIR/$w")
    STAGE_SIZE=$(stat -c %s "$PMPNN_STAGE_DIR/$w" 2>/dev/null || echo "")
    if [ "$SRC_SIZE" != "$STAGE_SIZE" ]; then
        echo "$(hostname): staging ProteinMPNN $w -> $PMPNN_STAGE_DIR/$w"
        cp -f "$PMPNN_SRC_DIR/$w" "$PMPNN_STAGE_DIR/$w"
    fi
done
echo "$(hostname): ProteinMPNN weights staged at $PMPNN_STAGE_DIR"
export PMPNN_WEIGHTS_DIR="$PMPNN_STAGE_DIR"

# --- Stage ESMFold (facebook/esmfold_v1) HF cache to /netscratch ---
# proteinfoundation/metrics/designability.py:run_and_store_esm reloads the
# 8.4 GB ESMFold model PER scRMSD CALL via `EsmForProteinFolding.from_pretrained
# (..., cache_dir=os.environ["CACHE_DIR"])` (only when SLURM_JOB_ID is set).
# Reading 8.4 GB N_samples times from shared NFS is unworkable; stage the
# whole HF cache to /netscratch and point CACHE_DIR there. Compute nodes
# generally have no internet, so the source must be pre-populated on shared
# FS first (typically via a one-time download on the login node).
ESMFOLD_HF_REPO_DIR="models--facebook--esmfold_v1"
ESMFOLD_SRC_HUB="$HOME/.cache/huggingface/hub/$ESMFOLD_HF_REPO_DIR"
ESMFOLD_STAGE_ROOT="/netscratch/schekmenev/protein_utils/hf_cache"
ESMFOLD_STAGE_HUB="$ESMFOLD_STAGE_ROOT/hub/$ESMFOLD_HF_REPO_DIR"
if [ ! -d "$ESMFOLD_SRC_HUB" ] || [ ! -f "$ESMFOLD_SRC_HUB/refs/main" ]; then
    echo "ERROR: ESMFold HF cache missing at source: $ESMFOLD_SRC_HUB" >&2
    echo "  Pre-populate it on the login node (which has internet):" >&2
    echo '    python -c "from transformers import EsmForProteinFolding, AutoTokenizer;' >&2
    echo '    AutoTokenizer.from_pretrained(\"facebook/esmfold_v1\");' >&2
    echo '    EsmForProteinFolding.from_pretrained(\"facebook/esmfold_v1\")"' >&2
    exit 1
fi
mkdir -p "$ESMFOLD_STAGE_ROOT/hub"
SRC_BYTES=$(du -sb "$ESMFOLD_SRC_HUB" 2>/dev/null | awk '{print $1}')
STAGE_BYTES=$(du -sb "$ESMFOLD_STAGE_HUB" 2>/dev/null | awk '{print $1}' || echo "")
if [ "$SRC_BYTES" != "$STAGE_BYTES" ]; then
    echo "$(hostname): ESMFold cache stage outdated or missing (src=$SRC_BYTES bytes, stage=${STAGE_BYTES:-none}) — syncing..."
    if command -v rsync >/dev/null 2>&1; then
        if ! rsync -a --delete "$ESMFOLD_SRC_HUB"/ "$ESMFOLD_STAGE_HUB"/; then
            echo "ERROR: rsync ESMFold cache failed" >&2
            exit 1
        fi
    else
        echo "$(hostname): rsync not available — falling back to cp -a"
        rm -rf "$ESMFOLD_STAGE_HUB"
        mkdir -p "$ESMFOLD_STAGE_HUB"
        if ! cp -a "$ESMFOLD_SRC_HUB"/. "$ESMFOLD_STAGE_HUB"/; then
            echo "ERROR: cp -a ESMFold cache failed" >&2
            exit 1
        fi
    fi
    echo "$(hostname): ESMFold cache staged ($SRC_BYTES bytes) to $ESMFOLD_STAGE_HUB"
else
    echo "$(hostname): ESMFold cache already staged and up-to-date at $ESMFOLD_STAGE_HUB"
fi
# CACHE_DIR is what designability.py passes as `from_pretrained(cache_dir=...)`.
# Quirk: the `cache_dir` kwarg expects a directory that contains
# `models--<org>--<repo>/...` *directly* — no `/hub/` subdir. (`HF_HOME` is
# the one that appends `/hub/`.) So when staging the on-disk hub layout we
# point CACHE_DIR at the `/hub` subdir we created.
export CACHE_DIR="$ESMFOLD_STAGE_ROOT/hub"
# HF_HOME instead expects to be a parent that contains `/hub/`, so it points
# at the root. Belt-and-suspenders: any transformers code that bypasses the
# cache_dir kwarg still lands in the same staged copy.
export HF_HOME="$ESMFOLD_STAGE_ROOT"
# Refuse to phone home if the local cache is incomplete — fail fast instead
# of hanging on a doomed download attempt from a compute node without internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$EVAL_OUTPUT_DIR"
cd "$REPO_ROOT"

# --- EMA selection ---
# When EVAL_USE_EMA is non-empty, swap CKPT_PATH for its `-EMA.ckpt` sibling.
# Idempotent: if CKPT_PATH already ends in `-EMA.ckpt` we leave it alone.
# Set EVAL_USE_EMA= (empty) to evaluate the online weights instead.
if [ -n "$EVAL_USE_EMA" ]; then
    case "$CKPT_PATH" in
        *-EMA.ckpt)
            echo "EVAL_USE_EMA set; CKPT_PATH already points at an EMA ckpt — using as-is."
            ;;
        *.ckpt)
            EMA_CKPT="${CKPT_PATH%.ckpt}-EMA.ckpt"
            if [ ! -f "$EMA_CKPT" ]; then
                echo "ERROR: EVAL_USE_EMA set but EMA sibling not found:" >&2
                echo "  expected: $EMA_CKPT" >&2
                echo "  online:   $CKPT_PATH" >&2
                exit 1
            fi
            echo "EVAL_USE_EMA set; switching from online -> EMA weights:"
            echo "  online: $CKPT_PATH"
            echo "  ema:    $EMA_CKPT"
            CKPT_PATH="$EMA_CKPT"
            ;;
        *)
            echo "ERROR: EVAL_USE_EMA set but CKPT_PATH does not end in .ckpt: $CKPT_PATH" >&2
            exit 1
            ;;
    esac
else
    echo "EVAL_USE_EMA empty — evaluating online weights at $CKPT_PATH"
fi

# --- Sanity: ckpt + binaries exist ---
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found at $CKPT_PATH" >&2
    echo "  Edit CKPT_PATH at the top of this script." >&2
    exit 1
fi
if [ ! -x "$FOLDSEEK_BIN" ]; then
    echo "ERROR: foldseek binary missing or not executable: $FOLDSEEK_BIN" >&2
    echo "  Expected pre-staged at $PROTEIN_UTILS_DIR/foldseek/bin/foldseek" >&2
    echo "  on this compute partition's local /netscratch." >&2
    exit 1
fi
if [ ! -x "$TMALIGN_BIN" ]; then
    echo "ERROR: TMalign binary missing or not executable: $TMALIGN_BIN" >&2
    echo "  Expected pre-staged at $PROTEIN_UTILS_DIR/tmalign/TMalign" >&2
    echo "  on this compute partition's local /netscratch." >&2
    exit 1
fi

# --- h100 partition: redirect FID dependencies to staged copies ---
# Mirrors train.sbatch (lines ~173-218): GearNet weights and the FID real
# features get staged once on the h100 /netscratch and re-used. The eval
# script reads features from a precomputed .pth — no source dataset staging
# is needed.
GEARNET_CKPT_PATH="$REPO_ROOT/data/metric_factory/model_weights/gearnet_ca.pth"
FID_REAL_FEATURES_PATH="$REPO_ROOT/data/metric_factory/features/pdb_eval_ca_features_monomers_frac0.1_rot1.pth"
if [ "$SLURM_JOB_PARTITION" = "h100" ]; then
    CKPT_ROOT_DIR="/netscratch/schekmenev/ckpt/mf_proteina"
    mkdir -p "$CKPT_ROOT_DIR"

    GEARNET_SRC="$REPO_ROOT/data/metric_factory/model_weights/gearnet_ca.pth"
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

    FID_FEATURES_SRC="$REPO_ROOT/data/metric_factory/features/pdb_eval_ca_features_monomers_frac0.1_rot1.pth"
    FID_FEATURES_STAGE="$CKPT_ROOT_DIR/$(basename "$FID_FEATURES_SRC")"
    if [ ! -f "$FID_FEATURES_SRC" ]; then
        echo "ERROR: FID real-features missing at $FID_FEATURES_SRC" >&2
        echo "  Build it via: sbatch scripts/dump_pdb_eval_ca_features.sbatch" >&2
        exit 1
    fi
    SRC_SIZE=$(stat -c %s "$FID_FEATURES_SRC")
    STAGE_SIZE=$(stat -c %s "$FID_FEATURES_STAGE" 2>/dev/null || echo "")
    if [ "$SRC_SIZE" != "$STAGE_SIZE" ]; then
        echo "$(hostname): staging FID real-features -> $FID_FEATURES_STAGE"
        cp -f "$FID_FEATURES_SRC" "$FID_FEATURES_STAGE"
    else
        echo "$(hostname): FID real-features already staged at $FID_FEATURES_STAGE"
    fi
    FID_REAL_FEATURES_PATH="$FID_FEATURES_STAGE"
fi


# --- Build CLI args ---
ARGS=(
    "--ckpt-path" "$CKPT_PATH"
    "--num-samples-per-length" "$EVAL_NUM_SAMPLES"
    "--lengths" "$EVAL_LENGTHS"
    "--output-dir" "$EVAL_OUTPUT_DIR"
    "--gearnet-ckpt" "$GEARNET_CKPT_PATH"
    "--fid-real-features" "$FID_REAL_FEATURES_PATH"
    "--foldseek-bin" "$FOLDSEEK_BIN"
    "--tmalign-bin" "$TMALIGN_BIN"
    "--nsteps" "$EVAL_NSTEPS"
    "--seed" "$EVAL_SEED"
    "--designability-threshold" "$EVAL_DESIGNABILITY_THRESHOLD"
    "--max-nsamples-per-batch" "$EVAL_MAX_NSAMPLES_PER_BATCH"
)
[ -n "$EVAL_QUICK_TEST" ]            && ARGS+=("--quick-test")
[ -n "$EVAL_SKIP_DESIGNABILITY" ]    && ARGS+=("--skip-designability")
[ -n "$EVAL_SKIP_FID" ]              && ARGS+=("--skip-fid")
[ -n "$EVAL_SKIP_TM_DIVERSITY" ]     && ARGS+=("--skip-tm-diversity")
[ -n "$EVAL_SKIP_CLUSTER_DIVERSITY" ] && ARGS+=("--skip-cluster-diversity")

EVAL_LOG="$REPO_ROOT/logs/eval_metrics_${SLURM_JOB_ID}.log"

echo ""
echo "============================================================"
echo "Running eval_metrics.py"
echo "  ckpt:           $CKPT_PATH"
echo "  output_dir:     $EVAL_OUTPUT_DIR"
echo "  lengths:        $EVAL_LENGTHS"
echo "  samples/length: $EVAL_NUM_SAMPLES"
echo "  meanflow nsteps:$EVAL_NSTEPS"
echo "  seed:           $EVAL_SEED"
echo "  gearnet:        $GEARNET_CKPT_PATH"
echo "  fid_features:   $FID_REAL_FEATURES_PATH"
echo "  foldseek_bin:   $FOLDSEEK_BIN"
echo "  tmalign_bin:    $TMALIGN_BIN"
echo "  log:            $EVAL_LOG"
echo "  refold_rmsd:    auto (logged with designability; per-length + total in results.json)"
echo "============================================================"

python "$REPO_ROOT/scripts/eval_metrics.py" "${ARGS[@]}" 2>&1 | tee "$EVAL_LOG"
EVAL_EXIT=${PIPESTATUS[0]}
echo "[eval_metrics] exit code: $EVAL_EXIT"

echo "==========================================="
echo "Job completed: $(date)"
echo "Eval exit:    $EVAL_EXIT  (log: $EVAL_LOG)"
echo "Output dir:   $EVAL_OUTPUT_DIR"
[ -f "$EVAL_OUTPUT_DIR/results.json" ] && echo "Results JSON: $EVAL_OUTPUT_DIR/results.json"
echo "==========================================="

exit $EVAL_EXIT
