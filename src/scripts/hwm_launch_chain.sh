#!/bin/bash
#SBATCH --job-name=hwm_chain
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Chain HWM pipeline: train with K=4-10 intermediate waypoints, then eval.
#
# Assumes the following artifacts already exist (built by hwm_launch.sh):
#   - lewm_balanced_ppo/best.pt      (LeWM checkpoint; train_lewm_balanced + config infra.logdir)
#   - wm_cache/trajectory_dataset.npz
#   - wm_cache/latents.npz
#   - wm_cache/goal_library.npz
#   - wm_cache/ridge_model.pkl
#   - wm_cache/probes.pkl            (only needed when COST=probe)
#
# Latents in wm_cache are encoder-specific; if those files were built from another
# LeWM, re-encode / rebuild cache with this checkpoint before training the chain HWM.
#
# For a separate run with heavier sampling + epochs (different save dir, see
# hwm_launch_chain_strong.sh → hwm_high_chain_strong / results_chain_strong.json).
#
# Usage (run from the project root, i.e. plan/):
#   sbatch src/scripts/hwm_launch_chain.sh
#   tail -f logs/hwm_chain_<jobid>.out
#
# Or run locally:
#   bash src/scripts/hwm_launch_chain.sh
#
# Environment overrides:
#   WANDB=0        disable Weights & Biases (enabled by default for train + eval)
#   COST=l1        use L1 cost instead of probe cost (default: probe)
#   LEWM_CHECKPOINT=/path/to/best.pt   override balanced LeWM path (default: lewm_balanced_ppo/best.pt)

set -e

# ── Resolve PROJECT_ROOT ──────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SRC_DIR="${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec >"${PROJECT_ROOT}/logs/hwm_chain_${SLURM_JOB_ID}.out" 2>&1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SRC_DIR=${SRC_DIR}"
cd "${SRC_DIR}"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${HOME}/plan/.venv/bin/activate" ]]; then
    source "${HOME}/plan/.venv/bin/activate"
else
    echo "WARNING: No .venv found; using system python"
fi

# ── Blackwell flags ───────────────────────────────────────────────────────────
export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

pip install -q scikit-learn matplotlib moviepy imageio

# ── Flags ─────────────────────────────────────────────────────────────────────
WANDB_FLAG="--wandb"
[[ "${WANDB:-}" == "0" ]] && WANDB_FLAG=""
COST="${COST:-probe}"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Balanced LeWM (see src/config_lewm_balanced.yaml infra.logdir → lewm_balanced_ppo)
CHECKPOINT="${LEWM_CHECKPOINT:-${PROJECT_ROOT}/data/crafter/world_model/lewm_balanced_ppo/best.pt}"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
PROBE_PATH="${DATA_OUT}/probes.pkl"

HWM_CHAIN_LOGDIR="${PROJECT_ROOT}/data/crafter/world_model/hwm_high_chain"
HWM_CHAIN_CKPT="${HWM_CHAIN_LOGDIR}/best.pt"
CHAIN_RESULTS_JSON="${PROJECT_ROOT}/results/results_chain.json"

echo "LeWM checkpoint (base world model): ${CHECKPOINT}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
require_file() { [[ -f "$1" ]] || { echo "ERROR: Missing required file: $1"; exit 1; }; }

require_file "${CHECKPOINT}"
require_file "${GOAL_LIBRARY}"
require_file "${TRAJ_DATASET}"
require_file "${LATENTS_CACHE}"
require_file "${RIDGE_MODEL}"
[[ "${COST}" == "probe" ]] && require_file "${PROBE_PATH}"

mkdir -p "${HWM_CHAIN_LOGDIR}"
mkdir -p "${PROJECT_ROOT}/results"

# ── Step 1: Train chain HWM ───────────────────────────────────────────────────
echo "=== Step 1: Training chain ActionEncoder + HighLevelPredictor (K=4-10) ==="
python hwm/train_hwm_high.py \
    --checkpoint            "${CHECKPOINT}" \
    --traj_dataset          "${TRAJ_DATASET}" \
    --latents_cache         "${LATENTS_CACHE}" \
    --logdir                "${HWM_CHAIN_LOGDIR}" \
    --epochs                100 \
    --batch_size            256 \
    --lr                    3e-4 \
    --sigreg_lambda         0.2 \
    --triplets_per_episode  200 \
    --max_window            128 \
    --max_subseq_len        32 \
    --n_intermediates_min   4 \
    --n_intermediates_max   10 \
    --context_len           12 \
    ${WANDB_FLAG}

require_file "${HWM_CHAIN_CKPT}"
echo "Chain HWM checkpoint: ${HWM_CHAIN_CKPT}"

# ── Step 2: Evaluate hwm + hwm_oracle with chain checkpoint ──────────────────
rm -f "${CHAIN_RESULTS_JSON}"

for cond in hwm hwm_oracle; do
    echo "=== Step 2: ${cond} (cost=${COST}) ==="
    extra=(); [[ "${cond}" != hwm ]] && extra=(--append)
    python hwm/evaluate.py \
        --condition         "${cond}" \
        --checkpoint        "${CHECKPOINT}" \
        --hwm_checkpoint    "${HWM_CHAIN_CKPT}" \
        --goal_library      "${GOAL_LIBRARY}" \
        --ridge_model       "${RIDGE_MODEL}" \
        --latents_cache     "${LATENTS_CACHE}" \
        --traj_dataset      "${TRAJ_DATASET}" \
        --results_path      "${CHAIN_RESULTS_JSON}" \
        --n_episodes        65 \
        --seed_start        100 \
        --max_steps         1000 \
        --cost              "${COST}" \
        --probe_path        "${PROBE_PATH}" \
        ${WANDB_FLAG} "${extra[@]}"
done

echo ""
echo "Done: $(date)"
echo "Results: ${CHAIN_RESULTS_JSON}"
