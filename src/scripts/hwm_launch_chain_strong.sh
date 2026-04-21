#!/bin/bash
#SBATCH --job-name=hwm_chain_strong
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Chain HWM — “dense” run: more chains per episode, full-episode waypoint spans
# (--chain_full_episode, no --max_window cap), longer training.
# Saves to a *separate* folder from hwm_launch_chain.sh (hwm_high_chain) so both
# experiments can coexist without overwriting checkpoints.
#
# Usage (from project root):
#   sbatch src/scripts/hwm_launch_chain_strong.sh
#   tail -f logs/hwm_chain_strong_<jobid>.out
#
# Environment overrides: WANDB=0 to disable wandb (on by default), COST=probe|l1

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
    exec >"${PROJECT_ROOT}/logs/hwm_chain_strong_${SLURM_JOB_ID}.out" 2>&1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SRC_DIR=${SRC_DIR}"
echo "Run: dense chain training (separate save dir from hwm_high_chain)"
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

# pip install -q scikit-learn matplotlib moviepy imageio

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

# Separate from original chain run (hwm_high_chain + results_chain.json)
HWM_STRONG_LOGDIR="${PROJECT_ROOT}/data/crafter/world_model/hwm_high_chain_strong"
HWM_STRONG_CKPT="${HWM_STRONG_LOGDIR}/best.pt"
STRONG_RESULTS_JSON="${PROJECT_ROOT}/results/results_chain_strong.json"

require_file() { [[ -f "$1" ]] || { echo "ERROR: Missing required file: $1"; exit 1; }; }

require_file "${CHECKPOINT}"
require_file "${GOAL_LIBRARY}"
require_file "${TRAJ_DATASET}"
require_file "${LATENTS_CACHE}"
require_file "${RIDGE_MODEL}"
[[ "${COST}" == "probe" ]] && require_file "${PROBE_PATH}"

mkdir -p "${HWM_STRONG_LOGDIR}"
mkdir -p "${PROJECT_ROOT}/results"

echo "=== Step 1: Dense chain HWM (K=4-10, more sampling / epochs) ==="
echo "  logdir:   ${HWM_STRONG_LOGDIR}"
echo "  results:  ${STRONG_RESULTS_JSON}"

# python hwm/train_hwm_high.py \
#     --checkpoint            "${CHECKPOINT}" \
#     --traj_dataset          "${TRAJ_DATASET}" \
#     --latents_cache         "${LATENTS_CACHE}" \
#     --logdir                "${HWM_STRONG_LOGDIR}" \
#     --epochs                200 \
#     --batch_size            512 \
#     --lr                    3e-4 \
#     --sigreg_lambda         0.2 \
#     --triplets_per_episode  500 \
#     --max_subseq_len        32 \
#     --n_intermediates_min   4 \
#     --n_intermediates_max   16 \
#     --context_len           20 \
#     --chain_full_episode \
#     ${WANDB_FLAG}

require_file "${HWM_STRONG_CKPT}"
echo "Dense chain checkpoint: ${HWM_STRONG_CKPT}"

# rm -f "${STRONG_RESULTS_JSON}"

for cond in hwm hwm_oracle; do
    echo "=== Step 2: ${cond} (cost=${COST}) ==="
    extra=(); [[ "${cond}" != hwm ]] && extra=(--append)
    python hwm/evaluate.py \
        --condition         "${cond}" \
        --checkpoint        "${CHECKPOINT}" \
        --hwm_checkpoint    "${HWM_STRONG_CKPT}" \
        --goal_library      "${GOAL_LIBRARY}" \
        --ridge_model       "${RIDGE_MODEL}" \
        --latents_cache     "${LATENTS_CACHE}" \
        --traj_dataset      "${TRAJ_DATASET}" \
        --results_path      "${STRONG_RESULTS_JSON}" \
        --n_episodes        65 \
        --seed_start        100 \
        --max_steps         1000 \
        --cost              "${COST}" \
        --probe_path        "${PROBE_PATH}" \
        ${WANDB_FLAG} "${extra[@]}"
done

echo ""
echo "Done: $(date)"
echo "Results: ${STRONG_RESULTS_JSON}"
