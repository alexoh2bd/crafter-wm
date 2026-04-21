#!/bin/bash
#SBATCH --job-name=lewm_v2_pipeline
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# =============================================================================
# train_lewm_v2.sh — Full LeWM v2 + HWM pipeline
#
# Stage 0  (optional) Collect uniform-random rollouts — off by default (SKIP_COLLECT=1)
# Stage 1  Train LeWM v2 (mixed random + PPO buffers, predictor hidden dim 384)
# Stage 2  Build goal library + trajectory dataset from human playthroughs
# Stage 3  Train ActionEncoder + HighLevelPredictor (chain-strong recipe)
# Stage 4  Fit per-achievement linear probes
# Stage 5  Evaluate all conditions + oracle
# Stage 6  Generate figures
#
# Each stage can be skipped independently via environment variables:
#
#   SKIP_COLLECT=0   run uniform random collection to COLLECT_OUTPUT, then train on it (default)
#   SKIP_COLLECT=1   skip Stage 0 and train on RANDOM_BUFFER + PPO_BUFFER as-is
#   SKIP_LEWM=1      skip Stage 1 (e.g. checkpoint already exists)
#   SKIP_GOAL=1      skip Stage 2
#   SKIP_HWM=1       skip Stage 3
#   SKIP_PROBES=1    skip Stage 4 (auto-skipped when probes.pkl already exists)
#   SKIP_EVAL=1      skip Stage 5
#   SKIP_FIGURES=1   skip Stage 6
#
# All stages after Stage 1 require a valid LeWM checkpoint.  If SKIP_LEWM=1,
# set LEWM_CKPT to point at an existing checkpoint, e.g.:
#
#   SKIP_LEWM=1 LEWM_CKPT=data/crafter/world_model/lewm_v2/best.pt \
#       sbatch src/scripts/train_lewm_v2.sh
#
# Other overrides (all optional):
#   COLLECT_STEPS          random steps to collect  (default: 500000; only if SKIP_COLLECT=0)
#   COLLECT_OUTPUT         output pkl path          (default: data/crafter/random_rollouts/random_500k.pkl)
#   COLLECT_WORKERS        parallel CPU workers for collection (default: 1). Crafter has no GPU path.
#   LOGDIR                 LeWM v2 log dir    (default: data/crafter/world_model/lewm_v2)
#   RANDOM_BUFFER          (default: COLLECT_OUTPUT — set when SKIP_COLLECT=1 to override)
#   PPO_BUFFER             (default: data/crafter/ppo_rollouts/crafter_teacher_data.pkl)
#   N_STEPS                (default: 100000)
#   SEED                   (default: 0)
#   BATCH_SIZE             (default: 64)
#   PRECISION              (default: bf16)
#   PREDICTOR_HIDDEN_DIM   (default: 384)
#   HWM_LOGDIR             (default: data/crafter/world_model/hwm_high_v2)
#   N_EVAL_EPISODES        (default: 65)
#   COST                   probe|l1  (default: probe)
#   WANDB                  1|0       (default: 0) — enables --wandb for LeWM v2 + HWM stages
#   EXTRA_LEWM_ARGS        extra CLI flags for train_lewm_v2.py
#
# Submit from repo root:
#   sbatch src/scripts/train_lewm_v2.sh
# =============================================================================

set -euo pipefail

# ── Resolve PROJECT_ROOT (Slurm copies script to spool; use submit dir) ───────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SRC_DIR="${PROJECT_ROOT}/src"

mkdir -p "${PROJECT_ROOT}/logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec >"${PROJECT_ROOT}/logs/lewm_v2_pipeline_${SLURM_JOB_ID}.out" 2>&1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SRC_DIR=${SRC_DIR}"
echo "Start: $(date -Is)"
echo

if [[ ! -f "${SRC_DIR}/hwm/train_lewm_v2.py" ]]; then
    echo "ERROR: Cannot find src/hwm/train_lewm_v2.py"
    echo "  Run sbatch from repo root: cd /path/to/plan && sbatch src/scripts/train_lewm_v2.sh"
    exit 1
fi

cd "${PROJECT_ROOT}"

# ── Activate venv ─────────────────────────────────────────────────────────────
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

# ── Resolve repo-relative paths to absolute ───────────────────────────────────
resolve_repo_path() {
    local p="$1"
    [[ -z "${p}" ]] && { echo ""; return; }
    case "${p}" in
        /*) echo "${p}" ;;
        *)  echo "${PROJECT_ROOT}/${p}" ;;
    esac
}

# ── Stage skip flags ─────────────────────────────────────────────────────────
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_LEWM="${SKIP_LEWM:-0}"
SKIP_GOAL="${SKIP_GOAL:-0}"
SKIP_HWM="${SKIP_HWM:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

# ── User-facing config ────────────────────────────────────────────────────────
COLLECT_STEPS="${COLLECT_STEPS:-500000}"
COLLECT_OUTPUT="${COLLECT_OUTPUT:-data/crafter/random_rollouts/random_500k.pkl}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

LOGDIR="${LOGDIR:-data/crafter/world_model/lewm_v2}"
RANDOM_BUFFER="${RANDOM_BUFFER:-${COLLECT_OUTPUT:-data/crafter/random_rollouts/random_500k.pkl}}"
PPO_BUFFER="${PPO_BUFFER:-data/crafter/ppo_rollouts/crafter_teacher_data.pkl}"
N_STEPS="${N_STEPS:-100000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PRECISION="${PRECISION:-bf16}"
PREDICTOR_HIDDEN_DIM="${PREDICTOR_HIDDEN_DIM:-384}"
EXTRA_LEWM_ARGS="${EXTRA_LEWM_ARGS:-}"

HWM_LOGDIR="${HWM_LOGDIR:-data/crafter/world_model/hwm_high_v2}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-65}"
COST="${COST:-probe}"
WANDB="${WANDB:-1}"
if [[ "${WANDB}" == "1" ]]; then WANDB_FLAG="--wandb"; else WANDB_FLAG=""; fi

# Resolved absolute paths
LOGDIR_ABS="$(resolve_repo_path "${LOGDIR}")"
LEWM_CKPT="${LEWM_CKPT:-${LOGDIR_ABS}/best.pt}"
PPO_ABS="$(resolve_repo_path "${PPO_BUFFER}")"
HWM_LOGDIR_ABS="$(resolve_repo_path "${HWM_LOGDIR}")"
HWM_CKPT="${HWM_LOGDIR_ABS}/best.pt"

NPZ_DIR="${PROJECT_ROOT}/data/crafter/human"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
PROBE_PATH="${DATA_OUT}/probes.pkl"
RESULTS_JSON="${PROJECT_ROOT}/results/results_lewm_v2.json"

mkdir -p "${LOGDIR_ABS}" "${HWM_LOGDIR_ABS}" "${PROJECT_ROOT}/results" "${DATA_OUT}"

# ── Diagnostics ───────────────────────────────────────────────────────────────
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# =============================================================================
# Stage 0 — Collect uniform-random rollouts
# =============================================================================
COLLECT_OUTPUT_ABS="$(resolve_repo_path "${COLLECT_OUTPUT}")"

if [[ "${SKIP_COLLECT}" == "1" ]]; then
    echo "=== Stage 0: SKIPPED (SKIP_COLLECT=1) — training on RANDOM_BUFFER + PPO_BUFFER ==="
elif [[ -f "${COLLECT_OUTPUT_ABS}" ]]; then
    echo "=== Stage 0: Random rollouts already exist — skipping ==="
    echo "  ${COLLECT_OUTPUT_ABS}"
else
    echo "=== Stage 0: Collecting ${COLLECT_STEPS} uniform-random steps ==="
    echo "  output: ${COLLECT_OUTPUT_ABS}  workers: ${COLLECT_WORKERS}"
    "${PY}" "${PROJECT_ROOT}/scripts/collect_random_rollouts.py" \
        --steps    "${COLLECT_STEPS}" \
        --output   "${COLLECT_OUTPUT_ABS}" \
        --seed     "${SEED}" \
        --workers  "${COLLECT_WORKERS}"
    echo "Stage 0 done."
fi

# When collection runs (or existing COLLECT_OUTPUT is used), train on that file.
if [[ "${SKIP_COLLECT}" != "1" ]]; then
    RANDOM_BUFFER="${COLLECT_OUTPUT}"
fi

RANDOM_ABS="$(resolve_repo_path "${RANDOM_BUFFER}")"
if [[ ! -f "${RANDOM_ABS}" ]]; then
    echo "ERROR: RANDOM_BUFFER not found: ${RANDOM_ABS}"
    exit 1
fi
if [[ ! -f "${PPO_ABS}" ]]; then
    echo "ERROR: PPO_BUFFER not found: ${PPO_ABS}"
    exit 1
fi
echo "  LeWM buffers — random: ${RANDOM_ABS}"
echo "                 ppo:    ${PPO_ABS}"
echo

# =============================================================================
# Stage 1 — Train LeWM v2
# =============================================================================
if [[ "${SKIP_LEWM}" == "1" ]]; then
    echo "=== Stage 1: SKIPPED (SKIP_LEWM=1) ==="
    echo "  Using LeWM checkpoint: ${LEWM_CKPT}"
    if [[ ! -f "${LEWM_CKPT}" ]]; then
        echo "ERROR: LEWM_CKPT not found: ${LEWM_CKPT}"
        exit 1
    fi
else
    echo "=== Stage 1: Training LeWM v2 ==="
    echo "  logdir:     ${LOGDIR_ABS}"
    echo "  random buf: ${RANDOM_ABS}"
    echo "  ppo buf:    ${PPO_ABS}"
    echo "  steps:      ${N_STEPS}  seed: ${SEED}  batch: ${BATCH_SIZE}"
    "${PY}" "${SRC_DIR}/hwm/train_lewm_v2.py" \
        --logdir                "${LOGDIR_ABS}" \
        --random-buffer         "${RANDOM_ABS}" \
        --ppo-buffer            "${PPO_ABS}" \
        --n-steps               "${N_STEPS}" \
        --batch-size            "${BATCH_SIZE}" \
        --seed                  "${SEED}" \
        --precision             "${PRECISION}" \
        --predictor-hidden-dim  "${PREDICTOR_HIDDEN_DIM}" \
        ${WANDB_FLAG} \
        ${EXTRA_LEWM_ARGS}
    echo "Stage 1 done. Checkpoint: ${LEWM_CKPT}"
fi
echo

# =============================================================================
# Stage 2 — Build goal library + trajectory dataset
# =============================================================================
if [[ "${SKIP_GOAL}" == "1" ]]; then
    echo "=== Stage 2: SKIPPED (SKIP_GOAL=1) ==="
elif [[ -f "${GOAL_LIBRARY}" && -f "${TRAJ_DATASET}" ]]; then
    echo "=== Stage 2: Goal library exists — skipping ==="
else
    echo "=== Stage 2: Building goal library + trajectory dataset ==="
    cd "${SRC_DIR}"
    "${PY}" hwm/build_goal_library.py \
        --npz_dir         "${NPZ_DIR}" \
        --out_dir         "${DATA_OUT}" \
        --eval_ep_indices 95 96 97 98 99
    cd "${PROJECT_ROOT}"
fi
echo

# =============================================================================
# Stage 3 — Train HWM high-level modules (chain-strong recipe)
# =============================================================================
if [[ "${SKIP_HWM}" == "1" ]]; then
    echo "=== Stage 3: SKIPPED (SKIP_HWM=1) ==="
    if [[ ! -f "${HWM_CKPT}" ]]; then
        echo "ERROR: HWM checkpoint not found: ${HWM_CKPT}"
        exit 1
    fi
else
    echo "=== Stage 3: Training ActionEncoder + HighLevelPredictor (chain-strong) ==="
    echo "  checkpoint: ${LEWM_CKPT}"
    echo "  logdir:     ${HWM_LOGDIR_ABS}"
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
# Stage 4 — Fit per-achievement linear probes
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
# Stage 5 — Evaluate all conditions
# =============================================================================
if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo "=== Stage 5: SKIPPED (SKIP_EVAL=1) ==="
else
    echo "=== Stage 5: Evaluating all conditions (${N_EVAL_EPISODES} episodes, cost=${COST}) ==="
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
# Stage 6 — Generate figures
# =============================================================================
if [[ "${SKIP_FIGURES}" == "1" ]]; then
    echo "=== Stage 6: SKIPPED (SKIP_FIGURES=1) ==="
else
    echo "=== Stage 6: Generating figures ==="
    cd "${SRC_DIR}"
    "${PY}" hwm/plot_results.py \
        --results       "${RESULTS_JSON}" \
        --out           "${PROJECT_ROOT}/results" \
        --checkpoint    "${LEWM_CKPT}" \
        --goal_library  "${GOAL_LIBRARY}" \
        --latents_cache "${LATENTS_CACHE}" \
        --traj_dataset  "${TRAJ_DATASET}"
    cd "${PROJECT_ROOT}"
fi
echo

echo "=== Pipeline complete: $(date -Is) ==="
echo "LeWM checkpoint:  ${LEWM_CKPT}"
echo "HWM checkpoint:   ${HWM_CKPT}"
echo "Results JSON:     ${RESULTS_JSON}"
echo "Figures:          ${PROJECT_ROOT}/results/fig*.png"
