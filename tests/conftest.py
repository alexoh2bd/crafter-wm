"""
Shared pytest fixtures for LeWM pipeline tests.

All fixtures use tmp_path so they are isolated and auto-cleaned.
The src/ directory is added to sys.path so imports work without install.
"""

import sys
from pathlib import Path

import pytest
import torch

# Make src importable from any working directory.
# IMPORTANT: insert src/ BEFORE the repo root so plan/crafter.py (a rollout
# utility at the repo root) does NOT shadow the installed crafter package.
TESTS = Path(__file__).parent
SRC   = TESTS.parent / "src"
REPO  = SRC.parent
for _p in (str(SRC), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Ensure the repo root itself is NOT at position 0 (it shadows crafter pkg)
if str(REPO) in sys.path:
    sys.path.remove(str(REPO))
    sys.path.append(str(REPO))  # move to end so src/ takes priority

from helpers import ACTION_DIM, make_synthetic_pkl   # noqa: E402


@pytest.fixture
def synthetic_pkl(tmp_path) -> Path:
    p = tmp_path / "test_data.pkl"
    make_synthetic_pkl(p)
    return p


@pytest.fixture
def synthetic_pkl_teacher(tmp_path) -> Path:
    """Same format as teacher rollouts (collection_method=teacher_ppo)."""
    p = tmp_path / "teacher_data.pkl"
    make_synthetic_pkl(p, collection_method="teacher_ppo")
    return p


# ── Tiny model config ─────────────────────────────────────────────────────────

@pytest.fixture
def tiny_lewm_kwargs():
    """Minimal LeWM config that runs in <1 s on CPU."""
    return dict(
        img_size=64,
        patch_size=8,
        latent_dim=32,
        action_dim=ACTION_DIM,
        encoder_depth=1,
        encoder_heads=1,     # 32 / 1 = 32 head_dim — valid
        predictor_depth=1,
        predictor_heads=2,   # 32 / 2 = 16 head_dim — valid
        context_len=4,
        sigreg_M=16,
        sigreg_lambda=0.1,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_lewm_kwargs):
    from lemodel import LeWM
    return LeWM(**tiny_lewm_kwargs).cpu()


# ── Tiny training args namespace ─────────────────────────────────────────────

@pytest.fixture
def tiny_train_args(tmp_path, synthetic_pkl):
    import argparse
    return argparse.Namespace(
        data_path=str(synthetic_pkl),
        n_episodes=1,
        max_steps_per_episode=10,
        context_len=4,
        latent_dim=32,
        sigreg_M=16,
        sigreg_lambda=0.1,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        total_steps=10,
        log_every=999,       # suppress mid-epoch logging
        seed=0,
        logdir=str(tmp_path / "logs"),
    )


# ── Fake PPO checkpoint ──────────────────────────────────────────────────────

@pytest.fixture
def ppo_checkpoint(tmp_path):
    """Save a randomly initialised ActorCritic as a valid checkpoint."""
    from teacherPPO import ActorCritic
    policy = ActorCritic()
    ckpt_path = tmp_path / "ppo_teacher.pt"
    torch.save({"policy": policy.state_dict()}, ckpt_path)
    return ckpt_path
