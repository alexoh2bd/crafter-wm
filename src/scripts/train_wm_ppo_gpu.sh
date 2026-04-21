#!/bin/bash
#SBATCH --job-name=lewm_ppo_gpu
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

# =============================================================================
# train_wm_ppo_gpu.sh — balanced LeWM → LeWM smoke → human finetune
#
# 1) Train LeWM with train_lewm_balanced.py on PPO / GPU rollout pickle(s).
# 2) LeWM smoke: random 50-step rollout, print actions + one-step latent pred MSE
#    (uses LOGDIR/best.pt).  Skip with SKIP_SMOKE=1.
# 3) Build crafter_human_train.pkl from data/crafter/human/*.npz (if any).
# 4) Finetune on that pkl with letrain.py + config_finetune_human.yaml, resuming
#    from the balanced best.pt (frozen encoder + rollout loss).
#
# Config: src/config_lewm_balanced.yaml — `pipeline.smoke_max_steps` / `smoke_seed`
# for the LeWM smoke; optional `teacher_checkpoint` is not used by this script (use
# ppo_smoke_rollout.py manually if you need a PPO policy check).
#
# Usage:
#   sbatch src/scripts/train_wm_ppo_gpu.sh
#   SKIP_SMOKE=1 sbatch src/scripts/train_wm_ppo_gpu.sh
#   SKIP_HUMAN_FT=1 sbatch src/scripts/train_wm_ppo_gpu.sh
#   EXTRA_DATA_PATH=data/crafter/ppo_rollouts/extra.pkl sbatch src/scripts/train_wm_ppo_gpu.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/letrain.py" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    SRC_DIR="${PROJECT_ROOT}/src"
else
    SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    PROJECT_ROOT="$(cd "${SRC_DIR}/.." && pwd)"
fi

if [[ ! -f "${SRC_DIR}/train_lewm_balanced.py" ]]; then
    echo "ERROR: Cannot find src/train_lewm_balanced.py under SRC_DIR=${SRC_DIR}"
    echo "  Run sbatch from repo root, e.g.: cd /path/to/plan && sbatch src/scripts/train_wm_ppo_gpu.sh"
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

resolve_repo_path() {
    local p="$1"
    [[ -z "${p}" ]] && { echo ""; return; }
    case "${p}" in
        /*) echo "${p}" ;;
        *)  echo "${PROJECT_ROOT}/${p}" ;;
    esac
}

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${SRC_DIR}/.venv/bin/python" ]]; then
    PY="${SRC_DIR}/.venv/bin/python"
else
    PY="python3"
fi

CONFIG="${SRC_DIR}/config_lewm_balanced.yaml"
DATA_PATH="$(resolve_repo_path "data/crafter/ppo_rollouts/crafter_teacher_data.pkl")"
EXTRA_DATA_PATH="${EXTRA_DATA_PATH:-}"
RESUME_PATH=""
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_HUMAN_FT="${SKIP_HUMAN_FT:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --data)         DATA_PATH="$(resolve_repo_path "$2")"; shift 2 ;;
        --extra-data)   EXTRA_DATA_PATH="$(resolve_repo_path "$2")"; shift 2 ;;
        --resume)       RESUME_PATH="$(resolve_repo_path "$2")"; shift 2 ;;
        --skip-smoke)   SKIP_SMOKE=1; shift ;;
        --skip-human-ft) SKIP_HUMAN_FT=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "${CONFIG}" in
    /*) ;;
    *) CONFIG="$(resolve_repo_path "${CONFIG}")" ;;
esac

[[ -f "${CONFIG}" ]] || { echo "ERROR: Config not found: ${CONFIG}"; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: Rollout data not found: ${DATA_PATH}"; exit 1; }

if [[ -n "${EXTRA_DATA_PATH}" ]] && [[ ! -f "${EXTRA_DATA_PATH}" ]]; then
    echo "ERROR: extra data not found: ${EXTRA_DATA_PATH}"
    exit 1
fi

echo "================================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "SRC_DIR      : ${SRC_DIR}"
echo "Config       : ${CONFIG}"
echo "Data         : ${DATA_PATH}"
[[ -n "${EXTRA_DATA_PATH}" ]] && echo "Extra data   : ${EXTRA_DATA_PATH}"
echo "================================================================"

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
    elif isinstance(v, list):
        val = shlex.quote(",".join(str(x) for x in v))
    else:
        val = shlex.quote(str(v))
    print(f"{key}={val}")
PY
)"

: "${LOGDIR:?missing infra.logdir in config}"

LOGDIR="$(resolve_repo_path "${LOGDIR}")"
mkdir -p "${LOGDIR}"

# ── Optional pipeline keys (from config_lewm_balanced.yaml `pipeline:` section)
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-50}"
SMOKE_SEED="${SMOKE_SEED:-0}"
HUMAN_NPZ_DIR="$(resolve_repo_path "${HUMAN_NPZ_DIR:-data/crafter/human}")"
HUMAN_TRAIN_PKL="$(resolve_repo_path "${HUMAN_TRAIN_PKL:-data/crafter/human/crafter_human_train.pkl}")"
HUMAN_FT_LOGDIR="$(resolve_repo_path "${HUMAN_FT_LOGDIR:-data/crafter/world_model/lewm_balanced_human_ft}")"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-config_finetune_human.yaml}"
case "${FINETUNE_CONFIG}" in
    /*) ;;
    *) FINETUNE_CONFIG="${SRC_DIR}/${FINETUNE_CONFIG}" ;;
esac

# ── 1) Balanced LeWM on PPO / GPU rollouts ───────────────────────────────────
echo ""
echo "================================================================"
echo "Stage 1 — Train LeWM (balanced) on PPO / GPU rollouts"
echo "================================================================"
echo "Log base     : ${LOGDIR}"

CMD=(
    "${PY}" "${SRC_DIR}/train_lewm_balanced.py"
    --config "${CONFIG}"
    --data_path "${DATA_PATH}"
    --logdir "${LOGDIR}"
)
[[ -n "${EXTRA_DATA_PATH}" ]] && CMD+=(--extra_data_path "${EXTRA_DATA_PATH}")
[[ -n "${RESUME_PATH}" ]] && CMD+=(--resume "${RESUME_PATH}")

"${CMD[@]}"

BALANCED_BEST="${LOGDIR}/best.pt"
[[ -f "${BALANCED_BEST}" ]] || { echo "ERROR: Missing balanced checkpoint: ${BALANCED_BEST}"; exit 1; }

# ── 2) LeWM smoke (trained checkpoint) ───────────────────────────────────────
if [[ "${SKIP_SMOKE}" != "1" ]]; then
    echo ""
    echo "================================================================"
    echo "Stage 2 — LeWM smoke (${SMOKE_MAX_STEPS} steps, random actions + pred MSE)"
    echo "================================================================"
    "${PY}" "${SRC_DIR}/lewm_smoke_rollout.py" \
        --checkpoint "${BALANCED_BEST}" \
        --max-steps "${SMOKE_MAX_STEPS}" \
        --seed "${SMOKE_SEED}" \
        --device cuda
else
    echo "SKIP_SMOKE=1 — skipping LeWM smoke rollout"
fi

# ── 3–4) Human pkl + finetune ────────────────────────────────────────────────
if [[ "${SKIP_HUMAN_FT}" != "1" ]]; then
    echo ""
    echo "================================================================"
    echo "Stage 3 — Build human train pickle from ${HUMAN_NPZ_DIR}"
    echo "================================================================"
    if compgen -G "${HUMAN_NPZ_DIR}"/*.npz > /dev/null; then
        mkdir -p "$(dirname "${HUMAN_TRAIN_PKL}")"
        "${PY}" "${SRC_DIR}/hwm/build_human_pkl.py" \
            --npz_dir "${HUMAN_NPZ_DIR}" \
            --out_path "${HUMAN_TRAIN_PKL}"

        [[ -f "${HUMAN_TRAIN_PKL}" ]] || { echo "ERROR: Human pkl not created: ${HUMAN_TRAIN_PKL}"; exit 1; }
        [[ -f "${FINETUNE_CONFIG}" ]] || { echo "ERROR: Finetune config not found: ${FINETUNE_CONFIG}"; exit 1; }

        echo ""
        echo "================================================================"
        echo "Stage 4 — Finetune LeWM on human rollouts (resume from balanced best.pt)"
        echo "================================================================"
        echo "  resume : ${BALANCED_BEST}"
        echo "  data   : ${HUMAN_TRAIN_PKL}"
        echo "  logdir : ${HUMAN_FT_LOGDIR}"
        mkdir -p "${HUMAN_FT_LOGDIR}"

        "${PY}" "${SRC_DIR}/letrain.py" \
            --config    "${FINETUNE_CONFIG}" \
            --data_path "${HUMAN_TRAIN_PKL}" \
            --logdir    "${HUMAN_FT_LOGDIR}" \
            --resume    "${BALANCED_BEST}" \
            --use_wandb

        echo "Human finetune checkpoints: ${HUMAN_FT_LOGDIR}/best.pt"
    else
        echo "WARNING: No *.npz under ${HUMAN_NPZ_DIR} — skipping human pkl build and finetune."
        echo "         Place human playthrough .npz files there or set SKIP_HUMAN_FT=1 to silence."
    fi
else
    echo "SKIP_HUMAN_FT=1 — skipping human pkl build and finetune"
fi

echo ""
echo "================================================================"
echo "Balanced checkpoints: ${LOGDIR}/best.pt"
echo "================================================================"
