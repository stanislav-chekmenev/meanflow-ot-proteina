#!/bin/bash
# Sweep eval_metrics.sbatch over EVAL_NSTEPS values.
#
# Submits one sbatch per nsteps in {1, 3, 10, 100, 200}. Each run is an
# independent job: own job id, own output dir tagged with nsteps so results
# are easy to locate after the queue drains.
#
# Usage:
#     ./scripts/eval_metrics_sweep.sh [-c CKPT_PATH] [-t SWEEP_TAG]
#
#     -c, --ckpt-path PATH   Override the checkpoint baked into eval_metrics.sbatch.
#     -t, --tag TAG          Custom sweep tag (default: timestamp).
#     -h, --help             Show this help.
#
# All other eval knobs use sbatch defaults (EMA, lengths 50..250, 100
# samples/length, seed=5). Override any EVAL_* by exporting before running:
#
#     EVAL_LENGTHS=100,200 ./scripts/eval_metrics_sweep.sh -c /path/to.ckpt

set -euo pipefail

REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
SBATCH_SCRIPT="$REPO_ROOT/scripts/eval_metrics.sbatch"
DATA_ROOT="/netscratch/schekmenev"

NSTEPS_LIST=(1 3 10 100 200)

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

# CLI flags. Both --ckpt-path and -c set CKPT_PATH so it falls into the
# passthrough path below alongside any env-var overrides.
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--ckpt-path)
            [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            CKPT_PATH="$2"
            shift 2
            ;;
        -t|--tag)
            [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            SWEEP_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage 2
            ;;
    esac
done

# Sweep tag shared across this batch of submissions so output dirs group
# together. Caller can override with --tag or SWEEP_TAG=foo.
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "ERROR: sbatch script not found at $SBATCH_SCRIPT" >&2
    exit 1
fi

# If a CKPT_PATH was provided (CLI or env), validate it exists up-front so
# we don't waste 5 queue submissions on a typo.
if [ -n "${CKPT_PATH:-}" ] && [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: --ckpt-path does not exist: $CKPT_PATH" >&2
    exit 1
fi

# Forward any EVAL_* / CKPT_PATH overrides from caller's environment to sbatch.
# `--export=ALL,VAR=val` keeps the login-shell env and adds/overrides the
# named vars per submission. EVAL_OUTPUT_DIR is set per-iteration to tag
# with nsteps so the 5 runs land in distinct, human-readable dirs.
PASSTHROUGH_VARS=(
    CKPT_PATH
    EVAL_USE_EMA
    EVAL_LENGTHS
    EVAL_NUM_SAMPLES
    EVAL_MAX_NSAMPLES_PER_BATCH
    EVAL_SEED
    EVAL_DESIGNABILITY_THRESHOLD
    EVAL_SKIP_DESIGNABILITY
    EVAL_SKIP_FID
    EVAL_SKIP_TM_DIVERSITY
    EVAL_SKIP_CLUSTER_DIVERSITY
    EVAL_QUICK_TEST
)

build_export_arg() {
    local nsteps="$1"
    local out_dir="$DATA_ROOT/mf_proteina_inference/eval_sweep_${SWEEP_TAG}/nsteps${nsteps}"
    local kvs="EVAL_NSTEPS=${nsteps},EVAL_OUTPUT_DIR=${out_dir}"
    for v in "${PASSTHROUGH_VARS[@]}"; do
        if [ -n "${!v:-}" ]; then
            kvs+=",${v}=${!v}"
        fi
    done
    echo "ALL,${kvs}"
}

echo "Sweep tag:      $SWEEP_TAG"
echo "nsteps values:  ${NSTEPS_LIST[*]}"
echo "ckpt override:  ${CKPT_PATH:-<sbatch default>}"
echo "Output root:    $DATA_ROOT/mf_proteina_inference/eval_sweep_${SWEEP_TAG}/"
echo "sbatch script:  $SBATCH_SCRIPT"
echo "============================================================"

for n in "${NSTEPS_LIST[@]}"; do
    export_arg=$(build_export_arg "$n")
    echo ""
    echo "Submitting EVAL_NSTEPS=$n ..."
    sbatch \
        --job-name="eval_nsteps${n}" \
        --export="$export_arg" \
        "$SBATCH_SCRIPT"
done

echo ""
echo "============================================================"
echo "Submitted ${#NSTEPS_LIST[@]} jobs. Tail with:"
echo "  squeue -u \"\$USER\" --name=eval_nsteps1,eval_nsteps3,eval_nsteps10,eval_nsteps100,eval_nsteps200"
echo "Results land under:"
echo "  $DATA_ROOT/mf_proteina_inference/eval_sweep_${SWEEP_TAG}/nsteps{1,3,10,100,200}/results.json"
