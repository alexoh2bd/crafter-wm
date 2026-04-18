"""
Tests for ledata.py:
  - collect_crafter_data   (Stage 2 data collection)
  - CrafterDataset         (sequence windowing + tensor conversion)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ledata import ACTION_DIM, CrafterDataset, collect_crafter_data
from helpers import make_synthetic_pkl


# ── collect_crafter_data ──────────────────────────────────────────────────────

class TestCollectCrafterData:
    def test_saves_pickle(self, tmp_path):
        """1 episode, 10 steps max — file must appear and be loadable."""
        out = tmp_path / "data.pkl"
        collect_crafter_data(n_episodes=1, max_steps_per_episode=10, save_path=str(out))
        assert out.exists(), "Pickle not created"
        with open(out, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, dict)

    def test_pickle_has_required_keys(self, tmp_path):
        out = tmp_path / "data.pkl"
        data = collect_crafter_data(n_episodes=1, max_steps_per_episode=5, save_path=str(out))
        assert "trajectories" in data
        assert "goal_library" in data
        assert "action_dim" in data

    def test_action_dim_matches_constant(self, tmp_path):
        out = tmp_path / "data.pkl"
        data = collect_crafter_data(n_episodes=1, max_steps_per_episode=5, save_path=str(out))
        assert data["action_dim"] == ACTION_DIM

    def test_trajectory_structure(self, tmp_path):
        """Each trajectory has obs, actions, achievements of equal length."""
        out = tmp_path / "data.pkl"
        data = collect_crafter_data(n_episodes=2, max_steps_per_episode=5, save_path=str(out))
        for traj in data["trajectories"]:
            T = len(traj["obs"])
            assert len(traj["actions"]) == T
            assert len(traj["achievements"]) == T

    def test_obs_dtype_and_shape(self, tmp_path):
        out = tmp_path / "data.pkl"
        data = collect_crafter_data(n_episodes=1, max_steps_per_episode=5, save_path=str(out))
        traj = data["trajectories"][0]
        assert len(traj["obs"]) > 0
        frame = traj["obs"][0]
        assert frame.dtype == np.uint8
        assert frame.shape == (64, 64, 3)

    def test_action_values_in_range(self, tmp_path):
        out = tmp_path / "data.pkl"
        data = collect_crafter_data(n_episodes=2, max_steps_per_episode=10, save_path=str(out))
        for traj in data["trajectories"]:
            for a in traj["actions"]:
                assert 0 <= a < ACTION_DIM


# ── CrafterDataset ────────────────────────────────────────────────────────────

class TestCrafterDataset:
    CONTEXT = 4  # small so synthetic trajectories of len 40 yield sequences

    def test_loads_without_error(self, synthetic_pkl):
        ds = CrafterDataset(str(synthetic_pkl), context_len=self.CONTEXT)
        assert len(ds) > 0

    def test_item_obs_shape(self, synthetic_pkl):
        """obs tensor must be (T, C, H, W)."""
        ds = CrafterDataset(str(synthetic_pkl), context_len=self.CONTEXT)
        obs, _ = ds[0]
        assert obs.shape == (self.CONTEXT, 3, 64, 64), obs.shape

    def test_item_action_shape(self, synthetic_pkl):
        """action tensor must be (T, ACTION_DIM) one-hot."""
        ds = CrafterDataset(str(synthetic_pkl), context_len=self.CONTEXT)
        _, act = ds[0]
        assert act.shape == (self.CONTEXT, ACTION_DIM), act.shape

    def test_obs_value_range(self, synthetic_pkl):
        """Pixel values must be normalised to [0, 1]."""
        ds = CrafterDataset(str(synthetic_pkl), context_len=self.CONTEXT)
        obs, _ = ds[0]
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_action_is_one_hot(self, synthetic_pkl):
        """Each timestep's action row must sum to exactly 1."""
        ds = CrafterDataset(str(synthetic_pkl), context_len=self.CONTEXT)
        _, act = ds[0]
        row_sums = act.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(self.CONTEXT)), row_sums

    def test_obs_dtype_float32(self, synthetic_pkl):
        ds = CrafterDataset(str(synthetic_pkl), context_len=self.CONTEXT)
        obs, act = ds[0]
        assert obs.dtype == torch.float32
        assert act.dtype == torch.float32

    def test_short_trajectories_skipped(self, tmp_path):
        """Trajectories shorter than context_len+1 must not produce sequences."""
        p = tmp_path / "short.pkl"
        make_synthetic_pkl(p, n_trajectories=5, traj_len=self.CONTEXT)  # exactly context_len — skipped
        ds = CrafterDataset(str(p), context_len=self.CONTEXT)
        assert len(ds) == 0

    def test_window_stride(self, tmp_path):
        """Dataset uses 50% stride: a traj of length L yields ~2L/T sequences."""
        traj_len = 20
        p = tmp_path / "stride.pkl"
        make_synthetic_pkl(p, n_trajectories=1, traj_len=traj_len)
        ds = CrafterDataset(str(p), context_len=self.CONTEXT)
        # stride = T//2 = 2; windows starting at 0,2,4,...,14 (end=start+4 <= 20)
        expected = len(range(0, traj_len - self.CONTEXT, self.CONTEXT // 2))
        assert len(ds) == expected

    def test_teacher_pkl_compatible(self, synthetic_pkl_teacher):
        """Teacher-collected pkl (different collection_method key) must load fine."""
        ds = CrafterDataset(str(synthetic_pkl_teacher), context_len=self.CONTEXT)
        assert len(ds) > 0
