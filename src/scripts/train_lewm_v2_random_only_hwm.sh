#!/bin/bash
#SBATCH --job-name=lewm_rand_wm_hwm
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# =============================================================================
# train_lewm_v2_random_only_hwm.sh — LeWM trained on uniform-random rollouts only,
# then full HWM pipeline (goal lib → HWM high → probes → eval → figures).
#
# All artifacts use RUN_TAG (default: random_wm) so they do not collide with
# train_lewm_v2.sh (mixed 70/30 buffers under lewm_v2 / wm_cache).
#
#   LeWM:     data/crafter/world_model/lewm_v2_${RUN_TAG}/
#   HWM:      data/crafter/world_model/hwm_high_${RUN_TAG}/
#   caches:   data/crafter/wm_cache_${RUN_TAG}/
#   results:  results/results_${RUN_TAG}.json
#   figures:  results/figures_${RUN_TAG}/
#
# Defaults: SKIP_COLLECT=1 (reuse RANDOM_BUFFER), Stage 1 uses --random-only
# (no PPO buffer load).
#
# Resume Stage 1 from an existing LeWM .pt (e.g. mixed run in lewm_v2):
#   logs/lewm_v2_pipeline_11209986.out used logdir data/crafter/world_model/lewm_v2
# Default LEWM_RESUME points at that run’s latest.pt. Train from scratch: NO_LEWM_RESUME=1
#
# Pick up at Stage 2 (skip LeWM train): default SKIP_LEWM=1 uses LeWM weights from
#   data/crafter/world_model/lewm_v2_random_wm/step_25000_ratio_0.8.pt
# for Stages 3–6 (Stage 2 goal lib does not load a checkpoint). Train Stage 1: SKIP_LEWM=0
#
# Submit:
#   sbatch src/scripts/train_lewm_v2_random_only_hwm.sh
#
# Override RUN_TAG:
#   RUN_TAG=my_ablation sbatch src/scripts/train_lewm_v2_random_only_hwm.sh
# =============================================================================

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SRC_DIR="${PROJECT_ROOT}/src"

mkdir -p "${PROJECT_ROOT}/logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec >"${PROJECT_ROOT}/logs/lewm_random_wm_pipeline_${SLURM_JOB_ID}.out" 2>&1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SRC_DIR=${SRC_DIR}"
echo "Start: $(date -Is)"
echo

if [[ ! -f "${SRC_DIR}/hwm/train_lewm_v2.py" ]]; then
    echo "ERROR: Cannot find src/hwm/train_lewm_v2.py"
    exit 1
fi

cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${HOME}/plan/.venv/bin/activate" ]]; then
    source "${HOME}/plan/.venv/bin/activate"
else
    echo "WARNING: No .venv found; using system python"
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
else
    PY="python3"
fi

export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

resolve_repo_path() {
    local p="$1"
    [[ -z "${p}" ]] && { echo ""; return; }
    case "${p}" in
        /*) echo "${p}" ;;
        *)  echo "${PROJECT_ROOT}/${p}" ;;
    esac
}

# npz is zip-backed; partial writes show up as BadZipFile at load time.
goal_traj_npz_ok() {
    local g="$1" t="$2"
    [[ -f "${g}" && -f "${t}" ]] || return 1
    GOAL_LIBRARY="${g}" TRAJ_DATASET="${t}" "${PY}" -c "
import numpy as np, os
for k in ('GOAL_LIBRARY', 'TRAJ_DATASET'):
    np.load(os.environ[k])
" 2>/dev/null
}

# ── Isolated run tag (all dirs namespaced) ───────────────────────────────────
RUN_TAG="${RUN_TAG:-random_wm}"

SKIP_COLLECT="${SKIP_COLLECT:-1}"
SKIP_LEWM="${SKIP_LEWM:-1}"
SKIP_GOAL="${SKIP_GOAL:-0}"
SKIP_HWM="${SKIP_HWM:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

COLLECT_STEPS="${COLLECT_STEPS:-500000}"
COLLECT_OUTPUT="${COLLECT_OUTPUT:-data/crafter/random_rollouts/random_500k.pkl}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

LOGDIR="${LOGDIR:-data/crafter/world_model/lewm_v2_${RUN_TAG}}"
RANDOM_BUFFER="${RANDOM_BUFFER:-data/crafter/random_rollouts/random_500k.pkl}"
PPO_BUFFER="${PPO_BUFFER:-data/crafter/ppo_rollouts/crafter_teacher_data.pkl}"
N_STEPS="${N_STEPS:-100000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PRECISION="${PRECISION:-bf16}"
PREDICTOR_HIDDEN_DIM="${PREDICTOR_HIDDEN_DIM:-384}"
EXTRA_LEWM_ARGS="${EXTRA_LEWM_ARGS:-}"

NO_LEWM_RESUME="${NO_LEWM_RESUME:-0}"
if [[ "${NO_LEWM_RESUME}" == "1" ]]; then
    LEWM_RESUME=""
else
    LEWM_RESUME="${LEWM_RESUME:-data/crafter/world_model/lewm_v2/latest.pt}"
fi

HWM_LOGDIR="${HWM_LOGDIR:-data/crafter/world_model/hwm_high_${RUN_TAG}}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-65}"
COST="${COST:-probe}"
WANDB="${WANDB:-1}"
if [[ "${WANDB}" == "1" ]]; then WANDB_FLAG="--wandb"; else WANDB_FLAG=""; fi

LOGDIR_ABS="$(resolve_repo_path "${LOGDIR}")"
if [[ -n "${LEWM_CKPT:-}" ]]; then
    LEWM_CKPT="$(resolve_repo_path "${LEWM_CKPT}")"
elif [[ "${SKIP_LEWM}" == "1" ]]; then
    LEWM_CKPT="$(resolve_repo_path "data/crafter/world_model/lewm_v2_random_wm/step_25000_ratio_0.8.pt")"
else
    LEWM_CKPT="${LOGDIR_ABS}/best.pt"
fi
PPO_ABS="$(resolve_repo_path "${PPO_BUFFER}")"
HWM_LOGDIR_ABS="$(resolve_repo_path "${HWM_LOGDIR}")"
HWM_CKPT="${HWM_LOGDIR_ABS}/best.pt"

NPZ_DIR="${PROJECT_ROOT}/data/crafter/human"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache_${RUN_TAG}"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
PROBE_PATH="${DATA_OUT}/probes.pkl"
RESULTS_JSON="${PROJECT_ROOT}/results/results_${RUN_TAG}.json"
FIGURES_OUT="${PROJECT_ROOT}/results/figures_${RUN_TAG}"

mkdir -p "${LOGDIR_ABS}" "${HWM_LOGDIR_ABS}" "${FIGURES_OUT}" "${DATA_OUT}"

echo "RUN_TAG=${RUN_TAG}"
echo "  LeWM logdir:     ${LOGDIR_ABS}"
echo "  LeWM ckpt:       ${LEWM_CKPT}  (SKIP_LEWM=${SKIP_LEWM})"
echo "  HWM logdir:      ${HWM_LOGDIR_ABS}"
echo "  wm_cache:        ${DATA_OUT}"
echo "  results JSON:    ${RESULTS_JSON}"
echo "  figures dir:     ${FIGURES_OUT}"
echo

"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# =============================================================================
# Stage 0 — optional collection
# =============================================================================
COLLECT_OUTPUT_ABS="$(resolve_repo_path "${COLLECT_OUTPUT}")"

if [[ "${SKIP_COLLECT}" == "1" ]]; then
    echo "=== Stage 0: SKIPPED (SKIP_COLLECT=1) — using RANDOM_BUFFER ==="
elif [[ -f "${COLLECT_OUTPUT_ABS}" ]]; then
    echo "=== Stage 0: Random rollouts already exist — skipping ==="
    echo "  ${COLLECT_OUTPUT_ABS}"
else
    echo "=== Stage 0: Collecting ${COLLECT_STEPS} uniform-random steps ==="
    "${PY}" "${PROJECT_ROOT}/scripts/collect_random_rollouts.py" \
        --steps    "${COLLECT_STEPS}" \
        --output   "${COLLECT_OUTPUT_ABS}" \
        --seed     "${SEED}" \
        --workers  "${COLLECT_WORKERS}"
    echo "Stage 0 done."
fi

if [[ "${SKIP_COLLECT}" != "1" ]]; then
    RANDOM_BUFFER="${COLLECT_OUTPUT}"
fi

RANDOM_ABS="$(resolve_repo_path "${RANDOM_BUFFER}")"
if [[ ! -f "${RANDOM_ABS}" ]]; then
    echo "ERROR: RANDOM_BUFFER not found: ${RANDOM_ABS}"
    exit 1
fi
echo "  random rollouts: ${RANDOM_ABS}"
echo "  LeWM Stage 1:    random-only (no PPO buffer)"
echo

# =============================================================================
# Stage 1 — LeWM v2 on random rollouts only
# =============================================================================
if [[ "${SKIP_LEWM}" == "1" ]]; then
    echo "=== Stage 1: SKIPPED (SKIP_LEWM=1) ==="
    echo "  Using LeWM checkpoint: ${LEWM_CKPT}"
    if [[ ! -f "${LEWM_CKPT}" ]]; then
        echo "ERROR: LEWM_CKPT not found: ${LEWM_CKPT}"
        exit 1
    fi
else
    echo "=== Stage 1: Training LeWM v2 (random rollouts only) ==="
    echo "  logdir:     ${LOGDIR_ABS}"
    echo "  random buf: ${RANDOM_ABS}"
    echo "  steps:      ${N_STEPS}  seed: ${SEED}  batch: ${BATCH_SIZE}"
    RESUME_ARGS=()
    if [[ -n "${LEWM_RESUME:-}" ]]; then
        LEWM_RESUME_ABS="$(resolve_repo_path "${LEWM_RESUME}")"
        echo "  resume:     ${LEWM_RESUME_ABS}"
        RESUME_ARGS=(--resume "${LEWM_RESUME_ABS}")
    fi
    "${PY}" "${SRC_DIR}/hwm/train_lewm_v2.py" \
        --logdir                "${LOGDIR_ABS}" \
        --random-buffer         "${RANDOM_ABS}" \
        --random-only \
        --n-steps               "${N_STEPS}" \
        --batch-size            "${BATCH_SIZE}" \
        --seed                  "${SEED}" \
        --precision             "${PRECISION}" \
        --predictor-hidden-dim  "${PREDICTOR_HIDDEN_DIM}" \
        "${RESUME_ARGS[@]}" \
        ${WANDB_FLAG} \
        ${EXTRA_LEWM_ARGS}
    echo "Stage 1 done. Checkpoint: ${LEWM_CKPT}"
fi
echo

# =============================================================================
# Stage 2 — Goal library
# =============================================================================
if [[ "${SKIP_GOAL}" == "1" ]]; then
    echo "=== Stage 2: SKIPPED (SKIP_GOAL=1) ==="
elif goal_traj_npz_ok "${GOAL_LIBRARY}" "${TRAJ_DATASET}"; then
    echo "=== Stage 2: Goal library exists — skipping ==="
else
    if [[ -f "${GOAL_LIBRARY}" || -f "${TRAJ_DATASET}" ]]; then
        echo "=== Stage 2: Cached goal/trajectory .npz missing or corrupt (BadZipFile) — rebuilding ==="
        rm -f "${GOAL_LIBRARY}" "${TRAJ_DATASET}" "${LATENTS_CACHE}" "${PROBE_PATH}" "${RIDGE_MODEL}"
    fi
    echo "=== Stage 2: Building goal library + trajectory dataset ==="
    echo "  (no LeWM checkpoint; downstream uses ${LEWM_CKPT})"
    cd "${SRC_DIR}"
    "${PY}" hwm/build_goal_library.py \
        --npz_dir         "${NPZ_DIR}" \
        --out_dir         "${DATA_OUT}" \
        --eval_ep_indices 95 96 97 98 99
    cd "${PROJECT_ROOT}"
fi
echo

# =============================================================================
# Stage 3 — HWM high (chain-strong)
# =============================================================================
if [[ "${SKIP_HWM}" == "1" ]]; then
    echo "=== Stage 3: SKIPPED (SKIP_HWM=1) ==="
    if [[ ! -f "${HWM_CKPT}" ]]; then
        echo "ERROR: HWM checkpoint not found: ${HWM_CKPT}"
        exit 1
    fi
else
    echo "=== Stage 3: Training ActionEncoder + HighLevelPredictor (chain-strong) ==="
    echo "  LeWM ckpt:  ${LEWM_CKPT}"
    echo "  HWM logdir: ${HWM_LOGDIR_ABS}"
    cd "${SRC_DIR}"
    "${PY}" hwm/train_hwm_high.py \
        --checkpoint            "${LEWM_CKPT}" \
        --traj_dataset          "${TRAJ_DATASET}" \
        --latents_cache         "${LATENTS_CACHE}" \
        --logdir                "${HWM_LOGDIR_ABS}" \
        --epochs                200 \
        --batch_size            512 \
        --lr                    3e-4 \
        --sigreg_lambda         0.2 \
        --triplets_per_episode  500 \
        --max_subseq_len        32 \
        --n_intermediates_min   4 \
        --n_intermediates_max   16 \
        --context_len           20 \
        --chain_full_episode \
        --force_reencode \
        ${WANDB_FLAG}
    cd "${PROJECT_ROOT}"
    echo "Stage 3 done. Checkpoint: ${HWM_CKPT}"
fi
echo

# =============================================================================
# Stage 4 — Probes
# =============================================================================
if [[ "${SKIP_PROBES}" == "1" ]]; then
    echo "=== Stage 4: SKIPPED (SKIP_PROBES=1) ==="
elif [[ -f "${PROBE_PATH}" ]]; then
    echo "=== Stage 4: Probes already exist — skipping ==="
else
    echo "=== Stage 4: Fitting per-achievement linear probes ==="
    cd "${SRC_DIR}"
    "${PY}" hwm/evaluate.py \
        --fit_probes \
        --latents_cache  "${LATENTS_CACHE}" \
        --npz_dir        "${NPZ_DIR}" \
        --probe_path     "${PROBE_PATH}"
    cd "${PROJECT_ROOT}"
fi
echo

# =============================================================================
# Stage 5 — Evaluate
# =============================================================================
if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo "=== Stage 5: SKIPPED (SKIP_EVAL=1) ==="
else
    echo "=== Stage 5: Evaluating (${N_EVAL_EPISODES} episodes, cost=${COST}) ==="
    cd "${SRC_DIR}"
    first_cond=1
    for cond in hwm hwm_oracle; do
        echo "--- condition: ${cond} ---"
        append_flag=""
        [[ "${first_cond}" == "0" ]] && append_flag="--append"
        first_cond=0
        "${PY}" hwm/evaluate.py \
            --condition      "${cond}" \
            --checkpoint     "${LEWM_CKPT}" \
            --hwm_checkpoint "${HWM_CKPT}" \
            --goal_library   "${GOAL_LIBRARY}" \
            --ridge_model    "${RIDGE_MODEL}" \
            --latents_cache  "${LATENTS_CACHE}" \
            --traj_dataset   "${TRAJ_DATASET}" \
            --results_path   "${RESULTS_JSON}" \
            --n_episodes     "${N_EVAL_EPISODES}" \
            --seed_start     100 \
            --max_steps      1000 \
            --cost           "${COST}" \
            --probe_path     "${PROBE_PATH}" \
            --n_workers      4 \
            ${WANDB_FLAG} ${append_flag}
    done
    cd "${PROJECT_ROOT}"
fi
echo

# =============================================================================
# Stage 6 — Figures
# =============================================================================
if [[ "${SKIP_FIGURES}" == "1" ]]; then
    echo "=== Stage 6: SKIPPED (SKIP_FIGURES=1) ==="
else
    echo "=== Stage 6: Generating figures → ${FIGURES_OUT} ==="
    cd "${SRC_DIR}"
    "${PY}" hwm/plot_results.py \
        --results       "${RESULTS_JSON}" \
        --out           "${FIGURES_OUT}" \
        --checkpoint    "${LEWM_CKPT}" \
        --goal_library  "${GOAL_LIBRARY}" \
        --latents_cache "${LATENTS_CACHE}" \
        --traj_dataset  "${TRAJ_DATASET}"
    cd "${PROJECT_ROOT}"
fi
echo

echo "=== Pipeline complete: $(date -Is) ==="
echo "RUN_TAG:          ${RUN_TAG}"
echo "LeWM checkpoint:  ${LEWM_CKPT}"
echo "HWM checkpoint:   ${HWM_CKPT}"
echo "Results JSON:     ${RESULTS_JSON}"
echo "Figures:          ${FIGURES_OUT}/"
