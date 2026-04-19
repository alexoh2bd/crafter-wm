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
pip install -q scikit-learn matplotlib

# Set WANDB=1 in the environment to enable wandb logging, e.g.:
#   WANDB=1 sbatch src/scripts/hwm_launch.sh
WANDB_FLAG="--wandb"


# ── Paths (relative to PROJECT_ROOT) ─────────────────────────────────────────
CHECKPOINT="${PROJECT_ROOT}/logs/lewm_teacher_deep/best.pt"
NPZ_DIR="${PROJECT_ROOT}/data/human_crafter"
DATA_OUT="${PROJECT_ROOT}/data"
GOAL_LIBRARY="${DATA_OUT}/goal_library.npz"
TRAJ_DATASET="${DATA_OUT}/trajectory_dataset.npz"
LATENTS_CACHE="${DATA_OUT}/latents.npz"
RIDGE_MODEL="${DATA_OUT}/ridge_model.pkl"
HWM_HIGH_LOGDIR="${PROJECT_ROOT}/logs/hwm_high"
HWM_HIGH_CKPT="${HWM_HIGH_LOGDIR}/best.pt"
RESULTS_JSON="${PROJECT_ROOT}/results/results.json"

mkdir -p "${PROJECT_ROOT}/logs/hwm_high"
mkdir -p "${PROJECT_ROOT}/results"

# ── Step 1: Build goal library ────────────────────────────────────────────────
if [[ -f "${GOAL_LIBRARY}" && -f "${TRAJ_DATASET}" ]]; then
    echo "Goal library and trajectory dataset already exist — skipping Step 1"
else
    echo "=== Step 1: Building goal library ==="
    python hwm/build_goal_library.py \
        --npz_dir "${NPZ_DIR}" \
        --out_dir "${DATA_OUT}"
fi

# ── Step 2: Train HWM high-level modules ─────────────────────────────────────
if [[ -f "${HWM_HIGH_CKPT}" ]]; then
    echo "HWM high-level checkpoint already exists — skipping Step 2"
else
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
        ${WANDB_FLAG}
fi

# ── Step 3: Evaluate all conditions ──────────────────────────────────────────
echo "=== Step 3: Evaluating all conditions (50 episodes each) ==="
python hwm/evaluate.py \
    --condition      all \
    --checkpoint     "${CHECKPOINT}" \
    --hwm_checkpoint "${HWM_HIGH_CKPT}" \
    --goal_library   "${GOAL_LIBRARY}" \
    --ridge_model    "${RIDGE_MODEL}" \
    --latents_cache  "${LATENTS_CACHE}" \
    --traj_dataset   "${TRAJ_DATASET}" \
    --results_path   "${RESULTS_JSON}" \
    --n_episodes     50 \
    --seed_start     100 \
    --max_steps      1000 \
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
