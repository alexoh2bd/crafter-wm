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
# train_wm_teach.sh — Stage 3 only: train LeWM on pre-built teacher rollouts
#
# Lives under src/scripts/. Code is in src/; data and logs are relative to the
# repo root (PROJECT_ROOT). Default rollouts:
#   data/crafter/crafter_teacher_data.pkl
#
# Usage:
#   sbatch src/scripts/train_wm_teach.sh
#   sbatch src/scripts/train_wm_teach.sh --config src/config_teacher_deep.yaml
#   sbatch src/scripts/train_wm_teach.sh --data data/crafter/other.pkl
# =============================================================================

set -euo pipefail

# Slurm runs a *copy* of this script from a spool path (e.g. under /var/lib/slurm/).
# BASH_SOURCE then does not lie inside your repo — use SLURM_SUBMIT_DIR (= cwd
# from which `sbatch` was run) when it contains the expected tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/letrain.py" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    SRC_DIR="${PROJECT_ROOT}/src"
else
    SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    PROJECT_ROOT="$(cd "${SRC_DIR}/.." && pwd)"
fi

if [[ ! -f "${SRC_DIR}/letrain.py" ]]; then
    echo "ERROR: Cannot find src/letrain.py under SRC_DIR=${SRC_DIR}"
    echo "  Run sbatch from your repo root (directory that contains src/ ), e.g.:"
    echo "    cd /path/to/plan && sbatch src/scripts/train_wm_teach.sh"
    exit 1
fi

cd "${PROJECT_ROOT}"

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

CONFIG="${SRC_DIR}/config_teacher_deep.yaml"
# Default teacher rollouts (repo layout: data/crafter/…)
DATA_PATH="$(resolve_repo_path "data/crafter/crafter_teacher_data.pkl")"
RESUME_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --data)   DATA_PATH="$(resolve_repo_path "$2")"; shift 2 ;;
        --resume) RESUME_PATH="$(resolve_repo_path "$2")"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Relative --config is resolved against PROJECT_ROOT (same as data paths)
case "${CONFIG}" in
    /*) ;;
    *) CONFIG="$(resolve_repo_path "${CONFIG}")" ;;
esac

[[ -f "${CONFIG}" ]] || { echo "ERROR: Config not found: ${CONFIG}"; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: Rollout data not found: ${DATA_PATH}"; exit 1; }

echo "================================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "SRC_DIR      : ${SRC_DIR}"
echo "Config       : ${CONFIG}"
echo "Data         : ${DATA_PATH}"
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
    else:
        val = shlex.quote(str(v))
    print(f"{key}={val}")
PY
)"

: "${LOGDIR:?missing infra.logdir in config}"

LOGDIR="$(resolve_repo_path "${LOGDIR}")"
mkdir -p "${LOGDIR}"

echo "Log base     : ${LOGDIR}"
echo ""

# ── Stage 3: Train LeWM (Slurm allocates one GPU; PyTorch uses it as cuda:0) ──
echo "================================================================"
echo "Stage 3 — Training LeWM on teacher rollouts"
echo "================================================================"

"${PY}" "${SRC_DIR}/letrain.py" \
    --config    "${CONFIG}" \
    --data_path "${DATA_PATH}" \
    --logdir    "${LOGDIR}" \
    --use_wandb \
    ${RESUME_PATH:+--resume "${RESUME_PATH}"}

echo "Checkpoints: ${LOGDIR}/best.pt"
echo "================================================================"
