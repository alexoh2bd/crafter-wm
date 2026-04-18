#!/bin/bash
#SBATCH --job-name=crafter_hwm
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# Repo root: directory from which sbatch was run (Slurm sets this).
REPO="${SLURM_SUBMIT_DIR:-$HOME/plan}"
mkdir -p logs

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU(s):     $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo

# Activate virtual environment (prefer submit dir, then ~/plan)
if [[ -f "${REPO}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO}/.venv/bin/activate"
else
    # shellcheck source=/dev/null
    source "${HOME}/plan/.venv/bin/activate"
fi

# Blackwell (sm_100) stability: disable Triton GEMM and cuDNN frontend
# which have known bugs on early Blackwell drivers (580.x)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cudnn_frontend=false"
export NCCL_P2P_DISABLE=1

# Confirm Python version and GPU
python --version
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv

echo
echo "Diagnostics..."
python -c "
import os
print('Step 1: importing numpy'); import numpy; print('  ok', numpy.__version__)
print('Step 2: importing jax');   import jax; print('  ok', jax.__version__)
print('Step 3: jax.devices()');   print(' ', jax.devices())
print('Step 4: simple jax op');
import jax.numpy as jnp
x = jnp.ones((4,4))
print('  ok', x.sum())
print('Step 5: importing crafter'); import crafter; print('  ok')
print('All diagnostics passed')
"

echo
echo "Starting DreamerV3 training..."

python "${REPO}/dreamerv3/dreamerv3/main.py" \
  --logdir "${HOME}/logdir/crafter_hwm" \
  --configs crafter \
  --batch_size 64 \
  --logger.outputs '[jsonl,wandb]'

echo "Done: $(date)"
