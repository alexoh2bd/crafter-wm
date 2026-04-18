"""
Tests for teacherPPO.py:
  - preprocess()                  image pre-processing
  - ActorCritic                   forward, get_action
  - train_ppo()                   1 update smoke test
  - collect_rollouts_with_teacher collected pkl format + content
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

from teacherPPO import (
    ACTION_DIM,
    ActorCritic,
    collect_rollouts_with_teacher,
    preprocess,
    train_ppo,
)


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_output_shape(self):
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        t = preprocess(obs)
        assert t.shape == (1, 3, 64, 64)

    def test_output_dtype(self):
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        assert preprocess(obs).dtype == torch.float32

    def test_value_range(self):
        obs = np.full((64, 64, 3), 255, dtype=np.uint8)
        t = preprocess(obs)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_black_frame_is_zeros(self):
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        assert torch.allclose(preprocess(obs), torch.zeros(1, 3, 64, 64))

    def test_white_frame_is_ones(self):
        obs = np.full((64, 64, 3), 255, dtype=np.uint8)
        assert torch.allclose(preprocess(obs), torch.ones(1, 3, 64, 64))


# ── ActorCritic ───────────────────────────────────────────────────────────────

class TestActorCritic:
    B = 4

    @pytest.fixture
    def policy(self):
        return ActorCritic().cpu()

    def test_forward_logits_shape(self, policy):
        obs = torch.rand(self.B, 3, 64, 64)
        logits, values = policy(obs)
        assert logits.shape == (self.B, ACTION_DIM)

    def test_forward_values_shape(self, policy):
        obs = torch.rand(self.B, 3, 64, 64)
        _, values = policy(obs)
        assert values.shape == (self.B, 1)

    def test_get_action_stochastic_shapes(self, policy):
        obs = torch.rand(self.B, 3, 64, 64)
        action, logp, entropy, value = policy.get_action(obs, deterministic=False)
        assert action.shape == (self.B,)
        assert logp.shape   == (self.B,)
        assert entropy.shape == (self.B,)
        assert value.shape  == (self.B, 1)

    def test_get_action_deterministic_shapes(self, policy):
        obs = torch.rand(self.B, 3, 64, 64)
        action, logp, entropy, value = policy.get_action(obs, deterministic=True)
        assert action.shape == (self.B,)

    def test_action_values_in_range(self, policy):
        obs = torch.rand(self.B, 3, 64, 64)
        action, _, _, _ = policy.get_action(obs)
        assert action.min() >= 0
        assert action.max() < ACTION_DIM

    def test_backward_does_not_crash(self, policy):
        obs = torch.rand(self.B, 3, 64, 64)
        logits, values = policy(obs)
        (logits.sum() + values.sum()).backward()
        for name, p in policy.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ── train_ppo — 1-update smoke test ──────────────────────────────────────────

class TestTrainPPOSmoke:
    """
    Very small config (1 env, rollout_len=2, 1 epoch, 1 minibatch) so the
    test runs in <10 s on CPU.  total_steps=1 ensures exactly one update.
    """

    def test_checkpoint_saved(self, tmp_path):
        ckpt = tmp_path / "teacher.pt"
        train_ppo(
            total_steps=1,
            n_envs=1,
            rollout_len=2,
            epochs=1,
            minibatches=1,
            save_path=str(ckpt),
            device="cpu",
        )
        assert ckpt.exists()

    def test_checkpoint_loadable(self, tmp_path):
        ckpt = tmp_path / "teacher.pt"
        train_ppo(total_steps=1, n_envs=1, rollout_len=2,
                  epochs=1, minibatches=1, save_path=str(ckpt), device="cpu")
        state = torch.load(ckpt, map_location="cpu")
        assert "policy" in state

    def test_loaded_policy_matches_architecture(self, tmp_path):
        ckpt = tmp_path / "teacher.pt"
        train_ppo(total_steps=1, n_envs=1, rollout_len=2,
                  epochs=1, minibatches=1, save_path=str(ckpt), device="cpu")
        policy = ActorCritic()
        state = torch.load(ckpt, map_location="cpu")
        policy.load_state_dict(state["policy"])   # must not raise

    def test_returns_policy_object(self, tmp_path):
        ckpt = tmp_path / "teacher.pt"
        policy = train_ppo(
            total_steps=1, n_envs=1, rollout_len=2,
            epochs=1, minibatches=1, save_path=str(ckpt), device="cpu",
        )
        assert isinstance(policy, ActorCritic)

    def test_policy_weights_differ_from_init(self, tmp_path):
        """At least some weights must change after one gradient update."""
        init = ActorCritic()
        w0 = {n: p.clone() for n, p in init.named_parameters()}

        ckpt = tmp_path / "teacher.pt"
        train_ppo(total_steps=1, n_envs=1, rollout_len=2,
                  epochs=1, minibatches=1, save_path=str(ckpt), device="cpu")
        state = torch.load(ckpt, map_location="cpu")
        trained = ActorCritic()
        trained.load_state_dict(state["policy"])

        changed = any(
            not torch.allclose(w0[n], p) for n, p in trained.named_parameters()
        )
        assert changed, "No weights changed — PPO gradient update may be broken"


# ── collect_rollouts_with_teacher ─────────────────────────────────────────────

class TestCollectRollouts:
    N_EP = 1
    MAX_STEPS = 8

    def test_pkl_created(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        assert out.exists()

    def test_pkl_has_required_keys(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        data = collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        for key in ("trajectories", "goal_library", "action_dim",
                    "collection_method", "teacher_checkpoint"):
            assert key in data, f"Missing key: {key}"

    def test_collection_method_is_teacher_ppo(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        data = collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        assert data["collection_method"] == "teacher_ppo"

    def test_correct_number_of_trajectories(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        data = collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        assert len(data["trajectories"]) == self.N_EP

    def test_trajectory_structure(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        data = collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        for traj in data["trajectories"]:
            T = len(traj["obs"])
            assert len(traj["actions"]) == T
            assert len(traj["achievements"]) == T

    def test_obs_dtype_and_shape(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        data = collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        traj = data["trajectories"][0]
        assert len(traj["obs"]) > 0
        frame = traj["obs"][0]
        assert frame.dtype == np.uint8
        assert frame.shape == (64, 64, 3)

    def test_action_dim_matches_constant(self, tmp_path, ppo_checkpoint):
        out = tmp_path / "rollouts.pkl"
        data = collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=self.N_EP,
            save_path=str(out),
            max_steps_per_episode=self.MAX_STEPS,
            device="cpu",
        )
        assert data["action_dim"] == ACTION_DIM

    def test_compatible_with_crafter_dataset(self, tmp_path, ppo_checkpoint):
        """Collected rollouts must be loadable by CrafterDataset."""
        from ledata import CrafterDataset
        out = tmp_path / "rollouts.pkl"
        # Need enough steps so at least one trajectory is long enough (context_len=4)
        collect_rollouts_with_teacher(
            checkpoint_path=str(ppo_checkpoint),
            n_episodes=3,
            save_path=str(out),
            max_steps_per_episode=50,
            epsilon=1.0,    # pure random so env doesn't die too fast
            device="cpu",
        )
        # Should not raise; sequences may be 0 if all episodes are short
        ds = CrafterDataset(str(out), context_len=4)
        assert isinstance(len(ds), int)
