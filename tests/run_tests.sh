#!/bin/bash
# =============================================================================
# run_tests.sh — Run the full LeWM test suite
#
# Usage (local):
#   cd ~/plan && bash tests/run_tests.sh
#
# Usage (Slurm — runs all tests on a compute node with GPU available):
#   sbatch tests/run_tests.sh
#
# Marks:
#   Tests that hit the real Crafter env (collect_crafter_data, train_ppo,
#   collect_rollouts_with_teacher) are inherently integration tests and
#   may take 1-3 minutes. All model/data/config tests run in <30 s on CPU.
# =============================================================================
#SBATCH --job-name=lewm_tests
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TESTS_DIR="${REPO}/tests"
SRC_DIR="${REPO}/src"

echo "==================================================================="
echo "Repo     : ${REPO}"
echo "Tests    : ${TESTS_DIR}"
echo "Date     : $(date)"
echo "==================================================================="

# ── Activate environment ──────────────────────────────────────────────────────
if [[ -f "${REPO}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO}/.venv/bin/activate"
elif [[ -f "${HOME}/plan/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/plan/.venv/bin/activate"
else
    echo "WARNING: No .venv found — using system Python"
fi

# ── Ensure pytest is available ────────────────────────────────────────────────
python -m pip install -q pytest

# ── Run pytest ────────────────────────────────────────────────────────────────
# --tb=short      concise tracebacks
# -v              one line per test (easier to scan in Slurm logs)
# --color=no      avoid ANSI codes in log files
# PYTHONPATH adds src/ so imports resolve without install
# Run from src/ so Python doesn't find plan/crafter.py (which shadows the
# installed crafter package when plan/ is the cwd or on sys.path).
cd "${SRC_DIR}"
python -m pytest "${TESTS_DIR}" \
    -v \
    --tb=short \
    --color=no \
    --durations=10 \
    --import-mode=importlib \
    -W ignore::FutureWarning \
    "$@"           # pass extra pytest args, e.g. -k test_lemodel

echo "==================================================================="
echo "Tests finished: $(date)"
echo "==================================================================="
