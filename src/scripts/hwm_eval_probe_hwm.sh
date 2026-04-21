#!/bin/bash
# Eval HWM only with probe cost at both planning levels (cem_high + cem_low).
# Intended for GPU nodes (CPU is too slow for 65 × HWM episodes).
#
# Usage from repo root:
#   sbatch --gres=gpu:1 --wrap="bash src/scripts/hwm_eval_probe_hwm.sh"
#   # or locally:
#   bash src/scripts/hwm_eval_probe_hwm.sh
#
# Override: RESULTS_JSON, N_EPISODES, HWM_CKPT, WANDB=1

set -e

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SRC_DIR="${PROJECT_ROOT}/src"
cd "${SRC_DIR}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/data/crafter/world_model/lewm_human_ft/best.pt}"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache"
HWM_CKPT="${HWM_CKPT:-${PROJECT_ROOT}/data/crafter/world_model/hwm_high/best.pt}"
RESULTS_JSON="${RESULTS_JSON:-${PROJECT_ROOT}/results/eval_probe_hwm_only.json}"
N_EPISODES="${N_EPISODES:-65}"

WANDB_ARGS=()
[[ "${WANDB:-0}" == "1" ]] && WANDB_ARGS=(--wandb)

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "checkpoint=${CHECKPOINT}"
echo "hwm_checkpoint=${HWM_CKPT}"
echo "results_path=${RESULTS_JSON}"
echo "n_episodes=${N_EPISODES} cost=probe"

python hwm/evaluate.py \
    --condition         hwm \
    --checkpoint        "${CHECKPOINT}" \
    --hwm_checkpoint    "${HWM_CKPT}" \
    --goal_library      "${DATA_OUT}/goal_library.npz" \
    --ridge_model       "${DATA_OUT}/ridge_model.pkl" \
    --latents_cache     "${DATA_OUT}/latents.npz" \
    --traj_dataset      "${DATA_OUT}/trajectory_dataset.npz" \
    --results_path      "${RESULTS_JSON}" \
    --n_episodes        "${N_EPISODES}" \
    --seed_start        100 \
    --max_steps         1000 \
    --cost              probe \
    --probe_path        "${DATA_OUT}/probes.pkl" \
    --n_workers         1 \
    "${WANDB_ARGS[@]}"

echo "Done. Inspect aggregates.hwm.per_achievement.collect_wood in:"
echo "  ${RESULTS_JSON}"
p