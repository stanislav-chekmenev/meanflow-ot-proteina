#!/bin/bash
#SBATCH --job-name=hyp_t_gate_chir
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/hyp_t_gate_chir_%j.out
#SBATCH --error=logs/hyp_t_gate_chir_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Hypothesis: t-gate-chir — gate chirality hinge to low-t only"
echo "==========================================="

# --- Environment ---
# REPO_ROOT is the primary repo (shared venv tarball, data, logs).
# CODE_ROOT is this hypothesis's git worktree — configs and code come from
# the hypothesis branch so the run picks up branch-specific changes.
REPO_ROOT="/mnt/labs/home/schekmenev/projects/meanflow-ot-proteina"
CODE_ROOT="$REPO_ROOT/.claude/worktrees/agent-ac694162"
DATA_ROOT="/netscratch/schekmenev"

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

# --- Configurable Hydra overrides ---
# Experiment config (training_ca_debug.yaml)
EXP_NAME="hyp-t-gate-chir"

# Guardrail #1: accumulate_grad_batches=1 matches reality on 1-micro-batch/epoch.
ACCUMULATE_GRAD_BATCHES=1
# Guardrail #2: eval at nsamples=16 so chirality is a mean in [-1, +1] rather
# than a 1-bit Bernoulli sample. Reducer in protein_eval.py averages across
# the batch (verified in docs/hypotheses/t-gate-chir.md).
EVAL_NSAMPLES=16

EVAL_EVERY_N_STEPS=250
OT_COUPLING_ENABLED=False
# Guardrail: isolate t-gate effect from norm_p interactions (slot 3 sweeps norm_p).
MEANFLOW_NORM_P=0
NCPUS_PER_TASK_TRAIN=16
NCPUS_PER_TASK_PREPRO=2
OT_POOL_SIZE=0
LOSS_ACCUMULATION_STEPS=1
# Guardrail #4: datamodule.repeat=2, batch_size=2, max_epochs=20000 unchanged.
MAX_EPOCHS=20000
WARMUP_STEPS=1000
LR_DECAY_TOTAL_STEPS=23000
INIT_LR=1e-6
END_LR=5e-6
LR=1.e-4

# Self-conditioning — disabled so this run isolates the chirality t-gate.
SELF_COND=False
SELF_COND_PROB=0.0

# Chirality hinge loss — ENABLED, t-gated to low-t only.
CHIRALITY_ENABLED=True
CHIRALITY_WEIGHT=1.0
CHIRALITY_MARGIN_ALPHA=0.1
CHIRALITY_STRIDE=1
# Hypothesis knob: gate off chirality for t >= 0.3 (hard indicator gate).
# Mechanism: x_1_pred = z - t*v is dominated by the z noise term at high t,
# making T_pred uninformative about handedness. Restricting the hinge to
# t < 0.3 keeps the signal where x_1_pred ≈ x_1 and avoids crushing the MF
# gradient with high-variance/low-signal chirality updates.
CHIRALITY_T_GATE_MAX=0.3
CHIRALITY_T_GATE_MODE=hard

# Dataset config (pdb_debug.yaml)
DATAMODULE_REPEAT=2
DATAMODULE_BATCH_SIZE=2

# --- Run training ---
echo "Starting MeanFlow training for hypothesis t-gate-chir..."
echo "Guardrails: accum=$ACCUMULATE_GRAD_BATCHES, eval_nsamples=$EVAL_NSAMPLES, norm_p=$MEANFLOW_NORM_P"
echo "t-gate: t_gate_max=$CHIRALITY_T_GATE_MAX, t_gate_mode=$CHIRALITY_T_GATE_MODE"

# cd into CODE_ROOT so Python's implicit sys.path[0] (cwd) matches the
# worktree. Otherwise 'import proteinfoundation' resolves to whichever
# directory sbatch was submitted from — typically the primary repo on
# main, which silently bypasses all hypothesis-branch code.
cd "$CODE_ROOT"
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
        training.ot_coupling.ot_pool_size=$OT_POOL_SIZE \
        training.loss_accumulation_steps=$LOSS_ACCUMULATION_STEPS \
        hardware.ncpus_per_task_prepro_=$NCPUS_PER_TASK_PREPRO \
        training.self_cond=$SELF_COND \
        training.self_cond_prob=$SELF_COND_PROB \
        training.chirality_loss.enabled=$CHIRALITY_ENABLED \
        training.chirality_loss.weight=$CHIRALITY_WEIGHT \
        training.chirality_loss.margin_alpha=$CHIRALITY_MARGIN_ALPHA \
        training.chirality_loss.stride=$CHIRALITY_STRIDE \
        training.chirality_loss.t_gate_max=$CHIRALITY_T_GATE_MAX \
        training.chirality_loss.t_gate_mode=$CHIRALITY_T_GATE_MODE \
    --data_overrides \
        datamodule.repeat=$DATAMODULE_REPEAT \
        datamodule.batch_size=$DATAMODULE_BATCH_SIZE

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
