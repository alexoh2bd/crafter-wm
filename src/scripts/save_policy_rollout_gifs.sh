#!/usr/bin/env bash
# Record N policy rollouts as GIFs (default: HWM planner, 8 episodes).
# Paths align with train_lewm_v2_random_only_hwm.sh / RUN_TAG=random_wm.
#
# Usage (repo root):
#   bash src/scripts/save_policy_rollout_gifs.sh
#   N_GIFS=12 CONDITION=hwm sbatch --wrap="bash src/scripts/save_policy_rollout_gifs.sh"
#
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SRC="${PROJECT_ROOT}/src"

RUN_TAG="${RUN_TAG:-random_wm}"
N_GIFS="${N_GIFS:-8}"
CONDITION="${CONDITION:-hwm}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/rollout_gifs_${RUN_TAG}}"
SEED_START="${SEED_START:-100}"
MAX_STEPS="${MAX_STEPS:-1000}"

LEWM_CKPT="${LEWM_CKPT:-${PROJECT_ROOT}/data/crafter/world_model/lewm_v2_random_wm/step_25000_ratio_0.8.pt}"
HWM_CKPT="${HWM_CKPT:-${PROJECT_ROOT}/data/crafter/world_model/hwm_high_random_wm/best.pt}"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache_${RUN_TAG}"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
PROBE_PATH="${DATA_OUT}/probes.pkl"
RESULTS_JSON="${PROJECT_ROOT}/results/results_rollout_gifs_${RUN_TAG}.json"

if [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PY="${PROJECT_ROOT}/.venv/bin/python"
else
  PY="python3"
fi
export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT_DIR}"

echo "Recording ${N_GIFS} rollouts → ${OUT_DIR}"
echo "  condition=${CONDITION}  LeWM=${LEWM_CKPT}"
echo "  HWM=${HWM_CKPT}"

cd "${PROJECT_ROOT}"
exec "${PY}" "${SRC}/hwm/evaluate.py" \
  --condition "${CONDITION}" \
  --n_episodes "${N_GIFS}" \
  --seed_start "${SEED_START}" \
  --max_steps "${MAX_STEPS}" \
  --checkpoint "${LEWM_CKPT}" \
  --hwm_checkpoint "${HWM_CKPT}" \
  --goal_library "${GOAL_LIBRARY}" \
  --ridge_model "${RIDGE_MODEL}" \
  --latents_cache "${LATENTS_CACHE}" \
  --traj_dataset "${TRAJ_DATASET}" \
  --probe_path "${PROBE_PATH}" \
  --cost probe \
  --results_path "${RESULTS_JSON}" \
  --save_rollout_gifs "${OUT_DIR}" \
  --n_workers 1
