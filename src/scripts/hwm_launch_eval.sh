#!/bin/bash
#SBATCH --job-name=hwm_eval_only
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Eval-only: hwm + hwm_oracle. No training or cache rebuild.

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
    exec >"${PROJECT_ROOT}/logs/hwm_eval_${SLURM_JOB_ID}.out" 2>&1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
cd "${SRC_DIR}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${HOME}/plan/.venv/bin/activate" ]]; then
    source "${HOME}/plan/.venv/bin/activate"
fi

export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
pip install -q scikit-learn matplotlib moviepy imageio

WANDB_FLAG=""; [[ -n "${WANDB:-}" ]] && WANDB_FLAG="--wandb"
COST="${COST:-probe}"

CHECKPOINT="${PROJECT_ROOT}/data/crafter/world_model/lewm_human_ft/best.pt"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
HWM_HIGH_CKPT="${PROJECT_ROOT}/data/crafter/world_model/hwm_high/best.pt"
RESULTS_JSON="${PROJECT_ROOT}/results/hwm_eval.json"
PROBE_PATH="${DATA_OUT}/probes.pkl"

require_file() { [[ -f "$1" ]] || { echo "ERROR: Missing $1"; exit 1; }; }

require_file "${CHECKPOINT}"
require_file "${HWM_HIGH_CKPT}"
require_file "${GOAL_LIBRARY}"
require_file "${TRAJ_DATASET}"
require_file "${LATENTS_CACHE}"
require_file "${RIDGE_MODEL}"
[[ "${COST}" == "probe" ]] && require_file "${PROBE_PATH}"

mkdir -p "${PROJECT_ROOT}/results"
rm -f "${RESULTS_JSON}"

for cond in hwm hwm_oracle; do
    echo "=== ${cond} (cost=${COST}) ==="
    extra=(); [[ "${cond}" != hwm ]] && extra=(--append)
    python hwm/evaluate.py \
        --condition "${cond}" \
        --checkpoint "${CHECKPOINT}" \
        --hwm_checkpoint "${HWM_HIGH_CKPT}" \
        --goal_library "${GOAL_LIBRARY}" \
        --ridge_model "${RIDGE_MODEL}" \
        --latents_cache "${LATENTS_CACHE}" \
        --traj_dataset "${TRAJ_DATASET}" \
        --results_path "${RESULTS_JSON}" \
        --n_episodes 65 --seed_start 100 --max_steps 1000 \
        --cost "${COST}" --probe_path "${PROBE_PATH}" \
        --n_workers 4 ${WANDB_FLAG} "${extra[@]}" \
        --wandb
done

echo "Done: ${RESULTS_JSON}"