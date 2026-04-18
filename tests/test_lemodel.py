"""
Tests for lemodel.py:
  - SIGReg regulariser
  - ViTEncoder (encode)
  - LeWM forward pass, loss keys, shapes, differentiability
  - LeWM.rollout (autoregressive planning)
"""

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lemodel import LeWM, SIGReg

# ── Fixtures re-used from conftest are injected via pytest ───────────────────

B, T = 2, 4          # batch, context length
ACTION_DIM = 17


def make_batch(B=B, T=T):
    obs_seq = torch.rand(B, T, 3, 64, 64)    # (B, T, C, H, W) [0,1]
    a_seq   = torch.zeros(B, T, ACTION_DIM)
    a_seq.scatter_(2, torch.randint(0, ACTION_DIM, (B, T, 1)), 1.0)  # one-hot
    return obs_seq, a_seq


# ── SIGReg ────────────────────────────────────────────────────────────────────

class TestSIGReg:
    def test_scalar_output(self):
        reg = SIGReg(embed_dim=32, M=8)
        Z = torch.randn(B, 32)
        loss = reg(Z)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_non_negative(self):
        reg = SIGReg(embed_dim=32, M=8)
        Z = torch.randn(B * T, 32)
        assert reg(Z).item() >= 0.0

    def test_3d_input(self):
        """Accepts (N, B, D) input shape."""
        reg = SIGReg(embed_dim=32, M=8)
        Z = torch.randn(T, B, 32)
        loss = reg(Z)
        assert loss.shape == ()

    def test_differentiable(self):
        reg = SIGReg(embed_dim=32, M=8)
        Z = torch.randn(B, 32, requires_grad=True)
        reg(Z).backward()
        assert Z.grad is not None


# ── LeWM forward ─────────────────────────────────────────────────────────────

class TestLeWMForward:
    def test_returns_required_keys(self, tiny_model):
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        for key in ("loss", "pred_loss", "sigreg_loss", "z_seq", "z_hat"):
            assert key in out, f"Missing key: {key}"

    def test_loss_is_scalar_tensor(self, tiny_model):
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        assert out["loss"].shape == ()

    def test_pred_loss_sigreg_loss_are_floats(self, tiny_model):
        """pred_loss and sigreg_loss are detached floats (for logging)."""
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        assert isinstance(out["pred_loss"], float)
        assert isinstance(out["sigreg_loss"], float)

    def test_z_seq_shape(self, tiny_model, tiny_lewm_kwargs):
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        assert out["z_seq"].shape == (B, T, tiny_lewm_kwargs["latent_dim"])

    def test_z_hat_shape(self, tiny_model, tiny_lewm_kwargs):
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        assert out["z_hat"].shape == (B, T, tiny_lewm_kwargs["latent_dim"])

    def test_loss_is_positive(self, tiny_model):
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        assert out["loss"].item() > 0.0

    def test_backward_does_not_crash(self, tiny_model):
        obs_seq, a_seq = make_batch()
        out = tiny_model(obs_seq, a_seq)
        out["loss"].backward()
        # All parameters should have gradients
        for name, p in tiny_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_eval_mode_no_crash(self, tiny_model):
        tiny_model.eval()
        obs_seq, a_seq = make_batch()
        with torch.no_grad():
            out = tiny_model(obs_seq, a_seq)
        assert "loss" in out


# ── LeWM.encode ──────────────────────────────────────────────────────────────

class TestLeWMEncode:
    def test_output_shape(self, tiny_model, tiny_lewm_kwargs):
        obs = torch.rand(B, 3, 64, 64)
        z = tiny_model.encode(obs)
        assert z.shape == (B, tiny_lewm_kwargs["latent_dim"])

    def test_batch_size_1(self, tiny_model, tiny_lewm_kwargs):
        # BatchNorm1d requires eval() when B=1 (no per-batch statistics)
        tiny_model.eval()
        with torch.no_grad():
            obs = torch.rand(1, 3, 64, 64)
            z = tiny_model.encode(obs)
        assert z.shape == (1, tiny_lewm_kwargs["latent_dim"])

    def test_no_grad_inference(self, tiny_model, tiny_lewm_kwargs):
        tiny_model.eval()
        obs = torch.rand(B, 3, 64, 64)
        with torch.no_grad():
            z = tiny_model.encode(obs)
        assert z.shape == (B, tiny_lewm_kwargs["latent_dim"])


# ── LeWM.rollout ─────────────────────────────────────────────────────────────

class TestLeWMRollout:
    def test_output_shape(self, tiny_model, tiny_lewm_kwargs):
        D = tiny_lewm_kwargs["latent_dim"]
        H = 3
        z0    = torch.randn(B, D)
        a_seq = torch.zeros(B, H, ACTION_DIM)
        a_seq.scatter_(2, torch.randint(0, ACTION_DIM, (B, H, 1)), 1.0)
        z_pred = tiny_model.rollout(z0, a_seq)
        assert z_pred.shape == (B, H + 1, D)

    def test_first_step_equals_z0(self, tiny_model, tiny_lewm_kwargs):
        D = tiny_lewm_kwargs["latent_dim"]
        z0    = torch.randn(B, D)
        a_seq = torch.zeros(B, 2, ACTION_DIM)
        z_pred = tiny_model.rollout(z0, a_seq)
        assert torch.allclose(z_pred[:, 0], z0)
