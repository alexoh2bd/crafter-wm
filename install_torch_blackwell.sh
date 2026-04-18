#!/bin/bash
# One-time fix: replace cu121 PyTorch with cu128 build for Blackwell (sm_120).
# Run once interactively on a GPU node, or source before sbatch:
#   bash install_torch_blackwell.sh && sbatch src/teacher_launch.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO}/.venv/bin/activate"
else
    source "${HOME}/plan/.venv/bin/activate"
fi

echo "Current PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Installing PyTorch >=2.6 with CUDA 12.8 kernels (sm_120 support)..."

pip install --upgrade "torch>=2.6" torchvision \
    --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch
print('Installed:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Capability:', torch.cuda.get_device_capability(0))
"
echo "Done. Resubmit your jobs."
