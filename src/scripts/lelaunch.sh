#!/bin/bash
#SBATCH --job-name=crafter_hwm
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit multiple seeds as an array job:
#   sbatch --array=10-13 lelaunch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
if [[ ! -f "${WORK_DIR}/letrain.py" ]] && [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/src/letrain.py" ]]; then
        WORK_DIR="${SLURM_SUBMIT_DIR}/src"
    fi
fi
cd "${WORK_DIR}"

if [[ "${WORK_DIR}" == */src ]]; then
    PROJECT_ROOT="$(cd "${WORK_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${WORK_DIR}"
fi

mkdir -p logs
mkdir -p "${PROJECT_ROOT}/data"

CONFIG="${WORK_DIR}/config.yaml"

resolve_repo_path() {
    local p="$1"
    [[ -z "${p}" ]] && { echo ""; return; }
    case "${p}" in
        /*) echo "${p}" ;;
        *)  echo "${PROJECT_ROOT}/${p}" ;;
    esac
}

# ── Step 1: Activate environment & install deps ───────────────────────────
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${HOME}/plan/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/plan/.venv/bin/activate"
else
    echo "WARNING: No .venv found at ${PROJECT_ROOT}/.venv; using system python"
fi

# cu121 tops out at sm_90; RTX PRO 6000 Blackwell (sm_120) requires cu128.
pip install -q --upgrade "torch>=2.6" torchvision \
    --index-url https://download.pytorch.org/whl/cu128
pip install -q crafter einops timm wandb pyyaml

# Blackwell (sm_120) stability flags
export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ── Step 2: Load config.yaml into shell variables ─────────────────────────
export CONFIG_PATH="${CONFIG}"
eval "$(python <<'EOF'
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
EOF
)"

: "${SAVE_PATH:?missing data.save_path in config.yaml}"
: "${LOGDIR:?missing infra.logdir in config.yaml}"

SAVE_PATH="$(resolve_repo_path "${SAVE_PATH}")"
LOGDIR="$(resolve_repo_path "${LOGDIR}")"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "Config loaded: n_episodes=${N_EPISODES}  context_len=${CONTEXT_LEN}"
echo "               latent_dim=${LATENT_DIM}   epochs=${EPOCHS}  batch_size=${BATCH_SIZE}"
echo "               save_path=${SAVE_PATH}  logdir base=${LOGDIR}"

# ── Step 3: Collect data (skipped if file already exists) ─────────────────
if [ ! -f "${SAVE_PATH}" ]; then
    echo "Collecting Crafter data → ${SAVE_PATH} ..."
    python ledata.py \
        --n_episodes            "${N_EPISODES}" \
        --max_steps_per_episode "${MAX_STEPS_PER_EPISODE}" \
        --epsilon               "${EPSILON}" \
        --save_path             "${SAVE_PATH}"
else
    echo "Data already exists at ${SAVE_PATH}, skipping collection."
fi

# ── Step 4: Train — one GPU per SLURM job; use --array for multi-seed ─────
SEED=${SLURM_ARRAY_TASK_ID:-${SEED}}
LOGDIR_RUN="${LOGDIR}_seed${SEED}"
mkdir -p "${LOGDIR_RUN}"

echo "Training seed=${SEED} → ${LOGDIR_RUN}"

python letrain.py \
    --config                "${CONFIG}" \
    --data_path             "${SAVE_PATH}" \
    --n_episodes            "${N_EPISODES}" \
    --max_steps_per_episode "${MAX_STEPS_PER_EPISODE}" \
    --context_len           "${CONTEXT_LEN}" \
    --latent_dim            "${LATENT_DIM}" \
    --sigreg_M              "${SIGREG_M}" \
    --sigreg_lambda         "${SIGREG_LAMBDA}" \
    --epochs                "${EPOCHS}" \
    --batch_size            "${BATCH_SIZE}" \
    --lr                    "${LR}" \
    --weight_decay          "${WEIGHT_DECAY}" \
    --warmup_steps          "${WARMUP_STEPS}" \
    --total_steps           "${TOTAL_STEPS}" \
    --log_every             "${LOG_EVERY}" \
    --logdir                "${LOGDIR_RUN}" \
    --seed                  "${SEED}"

# ── Step 5: Build goal library from best checkpoint ───────────────────────
echo "Building goal library..."
python legoal.py \
    --checkpoint "${LOGDIR_RUN}/best.pt" \
    --data_path  "${SAVE_PATH}" \
    --save_path  "${LOGDIR_RUN}/goal_library.pkl" \
    --latent_dim "${LATENT_DIM}"

echo "Done: $(date)"
