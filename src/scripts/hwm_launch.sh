#!/bin/bash
#SBATCH --job-name=hwm_eval
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
# NOTE: Slurm copies this script to a spool directory before running it, so
# ${BASH_SOURCE[0]} resolves to /var/lib/slurm/… rather than the repo.
# Relative #SBATCH --output paths would therefore point into the spool and
# fail with "Permission denied".  We use SLURM_SUBMIT_DIR (always the cwd of
# the sbatch call) to anchor everything, and write logs from inside the script.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# HWM evaluation pipeline:
#   Step 1 — Build goal library from human playthroughs (skipped if output exists)
#   Step 2 — Train ActionEncoder + HighLevelPredictor
#   Step 3 — Evaluate all three conditions + oracle ablation
#   Step 4 — Generate figures
#
# Usage (run from the project root, i.e. plan/):
#   sbatch src/scripts/hwm_launch.sh
#   tail -f logs/hwm_<jobid>.out
#
# Or run locally:
#   bash src/scripts/hwm_launch.sh

set -e

# ── Resolve PROJECT_ROOT robustly under both Slurm and local execution ────────
# Under Slurm, BASH_SOURCE[0] points to a temp copy in the spool; use
# SLURM_SUBMIT_DIR instead (the directory where sbatch was called).
# Under local execution, fall back to BASH_SOURCE[0] relative navigation.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SRC_DIR="${PROJECT_ROOT}/src"

# Create log dir and redirect all output there so nothing goes to the spool
mkdir -p "${PROJECT_ROOT}/logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec >"${PROJECT_ROOT}/logs/hwm_${SLURM_JOB_ID}.out" 2>&1
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

# ── Blackwell (sm_120) flags ──────────────────────────────────────────────────
export NCCL_P2P_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Install any missing deps (scikit-learn needed for plan_linear; wandb optional)
pip install -q scikit-learn matplotlib moviepy imageio

# Set WANDB=1 in the environment to enable wandb logging, e.g.:
#   WANDB=1 sbatch src/scripts/hwm_launch.sh
WANDB_FLAG="--wandb"


# ── Paths (relative to PROJECT_ROOT) ─────────────────────────────────────────
BASE_CKPT="${PROJECT_ROOT}/data/crafter/world_model/lewm_teacher_deep/best.pt"
HUMAN_FT_CKPT="${PROJECT_ROOT}/data/crafter/world_model/lewm_human_ft/best.pt"
HUMAN_PKL="${PROJECT_ROOT}/data/crafter/human/crafter_human_train.pkl"
# CHECKPOINT is set to the human-finetuned model after Step 0b completes.
# All downstream steps (goal library, HWM training, eval) use this variable.
CHECKPOINT="${HUMAN_FT_CKPT}"
NPZ_DIR="${PROJECT_ROOT}/data/crafter/human"
DATA_OUT="${PROJECT_ROOT}/data/crafter/wm_cache"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
HWM_HIGH_LOGDIR="${PROJECT_ROOT}/data/crafter/world_model/hwm_high"
HWM_HIGH_CKPT="${HWM_HIGH_LOGDIR}/best.pt"
RESULTS_JSON="${PROJECT_ROOT}/results/results.json"

mkdir -p "${PROJECT_ROOT}/data/crafter/world_model/hwm_high"
mkdir -p "${PROJECT_ROOT}/results"

# ── Purge stale artifacts from any previous run so everything is rebuilt
# against the current checkpoint and the correct eval split. ──────────────────
echo "Removing stale cache artifacts for clean rebuild..."
rm -f "${GOAL_LIBRARY}" "${TRAJ_DATASET}" "${LATENTS_CACHE}" "${RIDGE_MODEL}"
rm -f "${HWM_HIGH_LOGDIR}/best.pt" "${HWM_HIGH_LOGDIR}/latest.pt"
rm -f "${DATA_OUT}/probes.pkl"

# ── Step 0a: Build human training pkl ────────────────────────────────────────
if [[ ! -f "${HUMAN_PKL}" ]]; then
    echo "=== Step 0a: Building human training pkl ==="
    python hwm/build_human_pkl.py \
        --npz_dir  "${NPZ_DIR}" \
        --out_path "${HUMAN_PKL}"
else
    echo "=== Step 0a: Human pkl exists — skipping build ==="
fi

# ── Step 0b: Finetune LeWM on human data ─────────────────────────────────────
echo "=== Step 0b: Finetuning LeWM on human data (frozen encoder, rollout loss) ==="
mkdir -p "${PROJECT_ROOT}/data/crafter/world_model/lewm_human_ft"
bash "${SRC_DIR}/scripts/train_wm_teach.sh" \
    --config  "${SRC_DIR}/config_finetune_human.yaml" \
    --data    "${HUMAN_PKL}" \
    --resume  "${BASE_CKPT}"

if [[ ! -f "${HUMAN_FT_CKPT}" ]]; then
    echo "ERROR: Expected human-finetune checkpoint not found: ${HUMAN_FT_CKPT}"
    exit 1
fi
echo "Human-finetune checkpoint: ${HUMAN_FT_CKPT}"

# ── Step 1: Build goal library ────────────────────────────────────────────────
echo "=== Step 1: Building goal library ==="
python hwm/build_goal_library.py \
    --npz_dir         "${NPZ_DIR}" \
    --out_dir         "${DATA_OUT}" \
    --eval_ep_indices 95 96 97 98 99

# ── Step 2: Train HWM high-level modules ─────────────────────────────────────
echo "=== Step 2: Training ActionEncoder + HighLevelPredictor ==="
python hwm/train_hwm_high.py \
    --checkpoint     "${CHECKPOINT}" \
    --traj_dataset   "${TRAJ_DATASET}" \
    --latents_cache  "${LATENTS_CACHE}" \
    --logdir         "${HWM_HIGH_LOGDIR}" \
    --epochs         100 \
    --batch_size     256 \
    --lr             3e-4 \
    --sigreg_lambda  0.2 \
    --triplets_per_episode 200 \
    --max_window     128 \
    --max_subseq_len 32 \
    --force_reencode \
    ${WANDB_FLAG}

# ── Step 3a: Fit achievement probes (diagnostic + cost function) ──────────────
PROBE_PATH="${DATA_OUT}/probes.pkl"
if [[ -f "${PROBE_PATH}" ]]; then
    echo "Probe dict already exists — skipping Step 3a"
else
    echo "=== Step 3a: Fitting per-achievement linear probes ==="
    python hwm/evaluate.py \
        --fit_probes \
        --latents_cache  "${LATENTS_CACHE}" \
        --npz_dir        "${NPZ_DIR}" \
        --probe_path     "${PROBE_PATH}"
fi

# ── Step 3b: Evaluate all conditions with probe cost ─────────────────────────
echo "=== Step 3b: Evaluating all conditions (50 episodes each) ==="
python hwm/evaluate.py \
    --condition      all \
    --checkpoint     "${CHECKPOINT}" \
    --hwm_checkpoint "${HWM_HIGH_CKPT}" \
    --goal_library   "${GOAL_LIBRARY}" \
    --ridge_model    "${RIDGE_MODEL}" \
    --latents_cache  "${LATENTS_CACHE}" \
    --traj_dataset   "${TRAJ_DATASET}" \
    --results_path   "${RESULTS_JSON}" \
    --n_episodes     65 \
    --seed_start     100 \
    --max_steps      1000 \
    --cost           probe \
    --probe_path     "${PROBE_PATH}" \
    --n_workers      4 \
    ${WANDB_FLAG}

# ── Step 4: Generate figures ──────────────────────────────────────────────────
echo "=== Step 4: Generating figures ==="
python hwm/plot_results.py \
    --results       "${RESULTS_JSON}" \
    --out           "${PROJECT_ROOT}/results" \
    --checkpoint    "${CHECKPOINT}" \
    --goal_library  "${GOAL_LIBRARY}" \
    --latents_cache "${LATENTS_CACHE}" \
    --traj_dataset  "${TRAJ_DATASET}"

echo ""
echo "Done: $(date)"
echo "Results: ${RESULTS_JSON}"
echo "Figures: ${PROJECT_ROOT}/results/fig*.png"
