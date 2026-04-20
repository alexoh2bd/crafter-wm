#!/bin/bash
#SBATCH --job-name=hwm_m32_w128
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# HWM v2 experiment: macro_dim=32, waypoint window=128
# Key design choices:
#   --macro_dim 32       CEM searches in R^32 instead of R^256 → tractable
#   --max_window 128     ActionEncoder sees navigation-then-interaction seqs
#   --context_len 12     pos-embed covers growing-T up to K_max+1=11 steps
#   Growing-T fix in chain_forward_loss (train/eval consistency)
#   Chain training K=4..10 waypoints, full-episode spans
#
# Saves to hwm_high_m32_w128 (does not overwrite other runs).
#
# Usage (from project root):
#   sbatch src/scripts/hwm_launch_m32_w128.sh
#   tail -f logs/hwm_m32_w128_<jobid>.out
#
# Environment overrides:
#   WANDB=0             disable wandb (on by default)
#   COST=probe|l1       planner cost function (default probe)
#   SKIP_TRAIN=1        skip training, go straight to eval (requires existing ckpt)

set -e

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SRC_DIR="${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec >"${PROJECT_ROOT}/logs/hwm_m32_w128_${SLURM_JOB_ID}.out" 2>&1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SRC_DIR=${SRC_DIR}"
echo "Run: macro_dim=32, max_window=128, growing-T fix"
cd "${SRC_DIR}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${HOME}/plan/.venv/bin/activate" ]]; then
    source "${HOME}/plan/.venv/bin/activate"
else
    echo "WARNING: No .venv found; using system python"
fi

export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

WANDB_FLAG="--wandb"
[[ "${WANDB:-}" == "0" ]] && WANDB_FLAG=""
COST="${COST:-probe}"

CHECKPOINT="${PROJECT_ROOT}/data/crafter/world_model/lewm_human_ft/best.pt"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
PROBE_PATH="${DATA_OUT}/probes.pkl"

HWM_LOGDIR="${PROJECT_ROOT}/data/crafter/world_model/hwm_high_m32_w128"
HWM_CKPT="${HWM_LOGDIR}/best.pt"
RESULTS_JSON="${PROJECT_ROOT}/results/results_m32_w128.json"

require_file() { [[ -f "$1" ]] || { echo "ERROR: Missing required file: $1"; exit 1; }; }

require_file "${CHECKPOINT}"
require_file "${GOAL_LIBRARY}"
require_file "${TRAJ_DATASET}"
require_file "${LATENTS_CACHE}"
require_file "${RIDGE_MODEL}"
[[ "${COST}" == "probe" ]] && require_file "${PROBE_PATH}"

mkdir -p "${HWM_LOGDIR}"
mkdir -p "${PROJECT_ROOT}/results"

# ── Step 1: Train ─────────────────────────────────────────────────────────────

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "=== SKIP_TRAIN=1: skipping training step ==="
    require_file "${HWM_CKPT}"
else
    echo "=== Step 1: Train HWM (macro_dim=32, max_window=128, growing-T) ==="
    echo "  logdir: ${HWM_LOGDIR}"

    python hwm/train_hwm_high.py \
        --checkpoint            "${CHECKPOINT}" \
        --traj_dataset          "${TRAJ_DATASET}" \
        --latents_cache         "${LATENTS_CACHE}" \
        --logdir                "${HWM_LOGDIR}" \
        --epochs                200 \
        --batch_size            512 \
        --lr                    3e-4 \
        --sigreg_lambda         0.2 \
        --triplets_per_episode  500 \
        --max_window            128 \
        --max_subseq_len        32 \
        --macro_dim             32 \
        --n_intermediates_min   4 \
        --n_intermediates_max   10 \
        --context_len           12 \
        --chain_full_episode \
        --val_frac              0.1 \
        --wandb_run_name        "hwm-m32-w128" \
        ${WANDB_FLAG}

    echo "Training done. Checkpoint: ${HWM_CKPT}"
fi

# ── Step 2: Eval ──────────────────────────────────────────────────────────────

rm -f "${RESULTS_JSON}"

for cond in hwm hwm_oracle; do
    echo "=== Step 2: ${cond} (cost=${COST}) ==="
    extra=(); [[ "${cond}" != hwm ]] && extra=(--append)
    python hwm/evaluate.py \
        --condition         "${cond}" \
        --checkpoint        "${CHECKPOINT}" \
        --hwm_checkpoint    "${HWM_CKPT}" \
        --goal_library      "${GOAL_LIBRARY}" \
        --ridge_model       "${RIDGE_MODEL}" \
        --latents_cache     "${LATENTS_CACHE}" \
        --traj_dataset      "${TRAJ_DATASET}" \
        --results_path      "${RESULTS_JSON}" \
        --n_episodes        65 \
        --seed_start        100 \
        --max_steps         1000 \
        --cost              "${COST}" \
        --probe_path        "${PROBE_PATH}" \
        ${WANDB_FLAG} "${extra[@]}"
done

echo ""
echo "Done: $(date)"
echo "Results: ${RESULTS_JSON}"
