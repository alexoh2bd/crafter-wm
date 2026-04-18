#!/bin/bash
#SBATCH --job-name=lewm_teacher
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err



# =============================================================================
# teacher_launch.sh — Full LeWM pipeline with PPO teacher rollout collection
#
# Pipeline:
#   Stage 1 — Train a lightweight PPO teacher policy on Crafter
#   Stage 2 — Collect high-quality rollouts using the trained teacher
#   Stage 3 — Train LeWM world model on teacher rollouts (multi-seed)
#   Stage 4 — Build goal library from best checkpoint
#
# Usage:
#   sbatch teacher_launch.sh                  # default: config_teacher.yaml
#   sbatch teacher_launch.sh --config my.yaml # custom config
#   sbatch teacher_launch.sh --mode collect   # skip PPO, just collect rollouts
# =============================================================================

set -euo pipefail

# Slurm copies this script to a spool dir (e.g. .../job12345/slurm_script).
# Config and .venv live in the repo — resolve project root explicitly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
if [[ ! -f "${WORK_DIR}/teacherPPO.py" ]] && [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/src/teacherPPO.py" ]]; then
        WORK_DIR="${SLURM_SUBMIT_DIR}/src"
    elif [[ -f "${SLURM_SUBMIT_DIR}/teacherPPO.py" ]]; then
        WORK_DIR="${SLURM_SUBMIT_DIR}"
    fi
fi
cd "${WORK_DIR}"

# Project root: parent of src/ when code lives under src/ (all data/logs paths in YAML are relative to this).
if [[ "${WORK_DIR}" == */src ]]; then
    PROJECT_ROOT="$(cd "${WORK_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${WORK_DIR}"
fi

# Map repo-relative paths (data/..., logs/...) to absolute paths under PROJECT_ROOT.
resolve_repo_path() {
    local p="$1"
    [[ -z "${p}" ]] && { echo ""; return; }
    case "${p}" in
        /*) echo "${p}" ;;
        *)  echo "${PROJECT_ROOT}/${p}" ;;
    esac
}

# Prefer .venv next to src/ or repo root (Slurm cwd is often submit dir, not src/).
if [[ -x "${WORK_DIR}/.venv/bin/python" ]]; then
    PY="${WORK_DIR}/.venv/bin/python"
elif [[ -x "${WORK_DIR}/../.venv/bin/python" ]]; then
    PY="${WORK_DIR}/../.venv/bin/python"
else
    PY="python3"
fi
DEFAULT_CONFIG="${WORK_DIR}/config_teacher.yaml"
DEEP_CONFIG="${WORK_DIR}/config_teacher_deep.yaml"
CONFIG="${DEEP_CONFIG}"
MODE="all"   # train | collect | all

# Parse optional CLI args passed via sbatch --export or positional
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --mode)   MODE="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    echo "  WORK_DIR=${WORK_DIR}  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
    echo "  Submit from the repo (e.g. sbatch src/teacher_launch.sh) or set --config /abs/path/config_teacher.yaml"
    exit 1
fi

echo "================================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "WORK_DIR     : ${WORK_DIR}"
echo "Config       : ${CONFIG}"
echo "Mode         : ${MODE}"
echo "================================================================"

# ── Step 0: Install dependencies ─────────────────────────────────────────────
# cu121 tops out at sm_90 (Hopper). The RTX PRO 6000 Blackwell is sm_120 and
# requires PyTorch built against CUDA 12.8 (cu128) or later.
pip install -q --upgrade "torch>=2.6" torchvision \
    --index-url https://download.pytorch.org/whl/cu128
pip install -q crafter einops timm wandb pyyaml

# Blackwell (sm_120) stability flags — disable Triton GEMM & cuDNN frontend
# which have known issues on early Blackwell drivers.
export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ── Load config into shell variables ─────────────────────────────────────────
# Use CONFIG_PATH env (not sys.argv): heredoc + "$CONFIG" is fragile under sbatch.
export CONFIG_PATH="${CONFIG}"
eval "$("${PY}" <<'PY'
import os, shlex, yaml
path = os.environ["CONFIG_PATH"]
with open(path) as f:
    cfg = yaml.safe_load(f)
flat = {}
for section in cfg.values():
    if isinstance(section, dict):
        flat.update(section)
for k, v in flat.items():
    if v is None:
        continue
    key = k.upper()
    if isinstance(v, bool):
        val = "true" if v else "false"
    else:
        val = shlex.quote(str(v))
    print(f"{key}={val}")
PY
)"

: "${METHOD:?missing key rollout_collection.method in config}"
: "${TEACHER_SAVE_PATH:?missing ppo.teacher_save_path}"
: "${TEACHER_CHECKPOINT:?missing rollout_collection.teacher_checkpoint}"
: "${ROLLOUT_SAVE_PATH:?missing rollout_collection.rollout_save_path}"
: "${LOGDIR:?missing infra.logdir}"
: "${N_EPISODES:?missing rollout_collection.n_episodes}"
: "${PPO_TOTAL_STEPS:?missing ppo.ppo_total_steps}"
: "${N_ENVS:?missing ppo.n_envs}"
: "${EPOCHS:?missing training.epochs}"
: "${BATCH_SIZE:?missing training.batch_size}"
: "${LATENT_DIM:?missing model.latent_dim}"
: "${COLLECTION_EPSILON:?missing rollout_collection.collection_epsilon}"
: "${MAX_STEPS_PER_EPISODE:?missing rollout_collection.max_steps_per_episode}"

ROLLOUT_SAVE_PATH="$(resolve_repo_path "${ROLLOUT_SAVE_PATH}")"
TEACHER_SAVE_PATH="$(resolve_repo_path "${TEACHER_SAVE_PATH}")"
TEACHER_CHECKPOINT="$(resolve_repo_path "${TEACHER_CHECKPOINT}")"
LOGDIR="$(resolve_repo_path "${LOGDIR}")"

mkdir -p "${PROJECT_ROOT}/data"

echo ""
echo "Rollout collection method : ${METHOD}"
echo "PPO total steps           : ${PPO_TOTAL_STEPS}"
echo "Teacher checkpoint        : ${TEACHER_CHECKPOINT}"
echo "Rollout save path         : ${ROLLOUT_SAVE_PATH}"
echo "n_episodes (collect)      : ${N_EPISODES}"
echo "LeWM epochs               : ${EPOCHS}  batch_size=${BATCH_SIZE}  latent_dim=${LATENT_DIM}"
echo ""

mkdir -p "$(dirname "${TEACHER_CHECKPOINT}")"
mkdir -p "$(dirname "${ROLLOUT_SAVE_PATH}")"
mkdir -p "${LOGDIR}"

# ── Stage 1: Train PPO teacher ────────────────────────────────────────────────
# if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
#     echo "================================================================"
#     echo "Stage 1 — Training PPO teacher (${PPO_TOTAL_STEPS} steps, ${N_ENVS} envs)"
#     echo "================================================================"

#     "${PY}" teacherPPO.py \
#         --config "${CONFIG}" \
#         --mode   train \
#         --device cuda \
#         --teacher_save_path "${TEACHER_SAVE_PATH}" \
#         --total_steps       "${PPO_TOTAL_STEPS}"

#     echo "Stage 1 complete. Teacher saved to ${TEACHER_SAVE_PATH}"
# fi

# ── Stage 2: Collect teacher rollouts ─────────────────────────────────────────
# if [[ "${MODE}" == "collect" || "${MODE}" == "all" ]]; then
#     echo "================================================================"
#     echo "Stage 2 — Collecting ${N_EPISODES} teacher rollouts"
#     echo "          method=${METHOD}  epsilon=${COLLECTION_EPSILON}"
#     echo "================================================================"

    if [[ "${METHOD}" == "teacher_ppo" ]]; then
        "${PY}" teacherPPO.py \
            --config            "${CONFIG}" \
            --mode              collect \
            --device            cuda \
            --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
            --rollout_save_path  "${ROLLOUT_SAVE_PATH}" \
            --n_episodes         "${N_EPISODES}"    \
            --use_wandb
    elif [[ "${METHOD}" == "heuristic" || "${METHOD}" == "random" ]]; then
        # Fall back to ledata.py for non-teacher methods
        EPSILON="${COLLECTION_EPSILON:-0.2}"
        [[ "${METHOD}" == "random" ]] && EPSILON="1.0"
        "${PY}" ledata.py \
            --n_episodes             "${N_EPISODES}" \
            --max_steps_per_episode  "${MAX_STEPS_PER_EPISODE}" \
            --epsilon                "${EPSILON}" \
            --save_path              "${ROLLOUT_SAVE_PATH}"
    else
        echo "ERROR: Unknown rollout collection method '${METHOD}'"
        exit 1
    fi

    echo "Stage 2 complete. Rollouts saved to ${ROLLOUT_SAVE_PATH}"
fi

# ── Stage 3: Train LeWM world model (multi-seed) ──────────────────────────────
# echo "================================================================"
# echo "Stage 3 — Training LeWM on teacher rollouts (4 seeds × 1 GPU)"
# echo "================================================================"

# for GPU in 0; do
#     SEED=$((GPU + 10))
#     SEED_LOGDIR="${LOGDIR}_seed${SEED}"
#     mkdir -p "${SEED_LOGDIR}"

#     CUDA_VISIBLE_DEVICES=${GPU} "${PY}" letrain.py \
#         --config    "${CONFIG}" \
#         --data_path "${ROLLOUT_SAVE_PATH}" \
#         --logdir    "${SEED_LOGDIR}" \
#         --seed      "${SEED}" \
#         > "${SEED_LOGDIR}/stdout.log" 2>&1 &

#     echo "  Launched seed=${SEED} on GPU=${GPU} → ${SEED_LOGDIR}"
# done

# echo "All LeWM jobs launched."
# echo "Monitor with: tail -f ${LOGDIR}_seed*/stdout.log"

# # Wait for all background training jobs before continuing to Stage 4
# wait
# echo "All LeWM training jobs finished."

# # ── Stage 4: Build goal library from best checkpoint ─────────────────────────
# echo "================================================================"
# echo "Stage 4 — Building goal library"
# echo "================================================================"

# Pick the seed with the lowest val_loss checkpoint
# BEST_CKPT=""
# BEST_LOSS=999999
# for SEED in 10 11 12 13; do
#     CKPT="${LOGDIR}_seed${SEED}/best.pt"
#     if [[ -f "${CKPT}" ]]; then
#         VAL_LOSS=$("${PY}" - <<EOF
# import torch
# ckpt = torch.load("${CKPT}", map_location='cpu')
# print(f"{ckpt.get('val_loss', 999):.6f}")
# EOF
# )
#         echo "  seed=${SEED}  val_loss=${VAL_LOSS}  (${CKPT})"
#         if python3 -c "exit(0 if float('${VAL_LOSS}') < float('${BEST_LOSS}') else 1)"; then
#             BEST_LOSS="${VAL_LOSS}"
#             BEST_CKPT="${CKPT}"
#         fi
#     fi
# done

# if [[ -n "${BEST_CKPT}" ]]; then
#     echo "Best checkpoint: ${BEST_CKPT} (val_loss=${BEST_LOSS})"
#     "${PY}" legoal.py \
#         --checkpoint "${BEST_CKPT}" \
#         --data_path  "${ROLLOUT_SAVE_PATH}" \
#         --save_path  "${LOGDIR}/goal_library.pkl" \
#         --latent_dim "${LATENT_DIM}"
#     echo "Goal library saved to ${LOGDIR}/goal_library.pkl"
# else
#     echo "WARNING: No best.pt found. Skipping goal library construction."
# fi

# echo "================================================================"
# echo "Pipeline complete."
# echo "  Teacher  : ${TEACHER_SAVE_PATH}"
# echo "  Rollouts : ${ROLLOUT_SAVE_PATH}"
# echo "  LeWM     : ${LOGDIR}_seed*/best.pt"
# echo "  Goals    : ${LOGDIR}/goal_library.pkl"
# echo "================================================================"
