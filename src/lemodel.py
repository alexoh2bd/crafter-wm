"""
LeWorldModel (LeWM) - faithful implementation from the paper.
Encoder: ViT-Tiny (CLS token -> BN projector -> z_t)
Predictor: Causal Transformer with AdaLN action conditioning
Loss: MSE + SIGReg(lambda=0.1)
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── SIGReg ──────────────────────────────────────────────────────────────────

class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.
    Projects embeddings onto M random directions, applies Epps-Pulley
    normality test to each 1D projection, averages the statistics.
    Forces latent distribution toward isotropic Gaussian.
    """
    def __init__(self, embed_dim: int, M: int = 1024, num_knots: int = 10):
        super().__init__()
        self.M = M
        self.num_knots = num_knots
        # Random projection matrix — fixed, not learned
        proj = torch.randn(M, embed_dim)
        proj = F.normalize(proj, dim=1)
        self.register_buffer('proj', proj)  # (M, D)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (N, B, D) — N timesteps, B batch, D embed dim
           OR (B, D) for a flat batch
        Returns scalar loss.
        """
        if Z.dim() == 3:
            N, B, D = Z.shape
            Z_flat = Z.reshape(N * B, D)  # (N*B, D)
        else:
            Z_flat = Z  # (B, D)

        # Project: (N*B, D) @ (D, M) -> (N*B, M)
        projections = Z_flat @ self.proj.T  # (N*B, M)

        # Normalize each projection to zero mean, unit variance
        mu = projections.mean(dim=0, keepdim=True)      # (1, M)
        sigma = projections.std(dim=0, keepdim=True) + 1e-8  # (1, M)
        proj_norm = (projections - mu) / sigma           # (N*B, M)

        # Epps-Pulley test: compare empirical characteristic function
        # to standard Gaussian's characteristic function at test points
        n = proj_norm.shape[0]
        t_vals = torch.linspace(0.1, 2.0, self.num_knots,
                                device=Z.device)  # (K,)

        # Empirical char. function: E[exp(i*t*x)] = E[cos(tx)] + i*E[sin(tx)]
        # |ECF(t)|^2 = E[cos(tx)]^2 + E[sin(tx)]^2
        # Gaussian: |CF(t)|^2 = exp(-t^2)
        # Loss = sum_t |ECF(t)|^2 - exp(-t^2)|
        loss = torch.tensor(0.0, device=Z.device)
        for t in t_vals:
            tx = t * proj_norm                           # (N*B, M)
            ecf_real = torch.cos(tx).mean(dim=0)        # (M,)
            ecf_imag = torch.sin(tx).mean(dim=0)        # (M,)
            ecf_sq = ecf_real ** 2 + ecf_imag ** 2      # (M,)
            gauss_sq = torch.exp(-t ** 2)               # scalar
            loss = loss + (ecf_sq - gauss_sq).abs().mean()

        return loss / self.num_knots


# ─── ViT Encoder ─────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        # Crafter is 64x64 — use patch_size=8 (not 14 as in paper,
        # which assumed 224x224). 8x8 patches -> 64 tokens, reasonable.
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)               # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class ViTEncoder(nn.Module):
    """
    ViT-Tiny: 12 layers, 3 heads, hidden_dim=192.
    Adapted for 64x64 Crafter frames with patch_size=8.
    CLS token -> BatchNorm projector -> z_t.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3,
                 mlp_ratio=4.0, latent_dim=256, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True,
                norm_first=True,   # Pre-LN for stability
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Projector: MLP + BatchNorm (paper spec)
        # BatchNorm applied BEFORE projection so gradients flow through BN
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)           # (B, N+1, D)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]        # (B, D) — CLS token only
        z = self.projector(cls_out)  # (B, latent_dim)
        return z


# ─── AdaLN Predictor ─────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization: scale/shift conditioned on action embedding.
    Initialized to zero so action conditioning is inactive at start.
    """
    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        # Zero-init: starts as identity, learns to modulate
        self.scale = nn.Linear(cond_dim, embed_dim)
        self.shift = nn.Linear(cond_dim, embed_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x, cond):
        # x: (B, T, D), cond: (B, T, C)
        x_norm = self.norm(x)
        scale = self.scale(cond)  # (B, T, D)
        shift = self.shift(cond)  # (B, T, D)
        return x_norm * (1 + scale) + shift


class PredictorBlock(nn.Module):
    """Single causal transformer block with AdaLN action conditioning."""
    def __init__(self, embed_dim: int, num_heads: int,
                 cond_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.adaln1 = AdaLN(embed_dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLN(embed_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond, causal_mask=None):
        # Self-attention with AdaLN
        x_norm = self.adaln1(x, cond)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=causal_mask,
                                is_causal=True if causal_mask is None else False)
        x = x + self.drop(attn_out)
        # FFN with AdaLN
        x = x + self.mlp(self.adaln2(x, cond))
        return x


class Predictor(nn.Module):
    """
    Causal Transformer predictor: 6 layers, 16 heads, 10% dropout.
    Action conditioning via AdaLN at each layer.
    Predicts z_{t+1} from history of z_{1:t} and a_{1:t}.
    Paper: ~10M params
    """
    def __init__(self, latent_dim: int = 256, action_dim: int = 17,
                 depth: int = 6, num_heads: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 context_len: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_len = context_len

        # Action embedding -> conditioning signal
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Positional embedding for sequence positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, context_len, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock(latent_dim, num_heads, latent_dim,
                           mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output projector: MLP + BatchNorm (paper spec)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, z_seq, a_seq):
        """
        z_seq: (B, T, latent_dim) — sequence of latent states
        a_seq: (B, T, action_dim) — sequence of actions
        Returns: z_hat: (B, T, latent_dim) — predicted next states
        """
        B, T, D = z_seq.shape

        # Embed actions -> conditioning
        cond = self.action_embed(a_seq)  # (B, T, D)

        x = z_seq + self.pos_embed[:, :T, :]

        # Build causal mask: position i can only attend to positions <= i
        causal_mask = torch.triu(
            torch.ones(T, T, device=z_seq.device), diagonal=1
        ).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

        for block in self.blocks:
            x = block(x, cond, causal_mask=causal_mask)

        # Flatten for BatchNorm then reshape
        x_flat = x.reshape(B * T, D)
        z_hat_flat = self.projector(x_flat)
        z_hat = z_hat_flat.reshape(B, T, D)

        return z_hat


# ─── Full LeWM ───────────────────────────────────────────────────────────────

class LeWM(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        latent_dim: int = 256,
        action_dim: int = 17,       # Crafter has 17 discrete actions
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        context_len: int = 16,
        sigreg_M: int = 1024,
        sigreg_lambda: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sigreg_lambda = sigreg_lambda

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=192,          # ViT-Tiny hidden dim
            depth=encoder_depth,
            num_heads=encoder_heads,
            latent_dim=latent_dim,
        )

        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            context_len=context_len,
            dropout=dropout,
        )

        self.sigreg = SIGReg(embed_dim=latent_dim, M=sigreg_M)

        # Empirical mean norm of encoder outputs used to rescale predicted
        # latents during open-loop rollout, keeping them on the same scale as
        # real encoded frames (computed from data/latents.npz: ~20.92).
        self.register_buffer(
            'latent_scale',
            torch.tensor(20.92, dtype=torch.float32),
        )

    def encode(self, obs):
        """obs: (B, C, H, W) -> z: (B, latent_dim)"""
        return self.encoder(obs)

    def predict(self, z_seq, a_seq):
        """
        z_seq: (B, T, latent_dim)
        a_seq: (B, T, action_dim) — one-hot or continuous
        -> z_hat: (B, T, latent_dim)
        """
        return self.predictor(z_seq, a_seq)

    def forward(
        self,
        obs_seq,
        a_seq,
        rollout_steps: int = 0,
        rollout_loss_weight: float = 0.1,
        pred_action_weights: Optional[torch.Tensor] = None,
    ):
        """
        obs_seq: (B, T, C, H, W)
        a_seq:   (B, T, action_dim)
        rollout_steps:       Number of open-loop steps for the rollout loss
                             (0 = disabled, backward-compatible default).
        rollout_loss_weight: Weight applied to rollout_loss in total_loss.
        pred_action_weights: Optional (action_dim,) tensor of non-negative weights.
                             When set, prediction MSE is averaged over time with
                             per-step weights w[a_t] where a_t is the action index
                             at each teacher-forcing step (same indexing as unweighted loss).
        Returns dict with loss, pred_loss, sigreg_loss[, rollout_loss].
        pred_loss / sigreg_loss are scalar tensors (not .item()) so torch.compile
        does not graph-break; callers should float(...) or .item() when logging.
        """
        B, T, C, H, W = obs_seq.shape

        # Encode all frames independently
        obs_flat = obs_seq.reshape(B * T, C, H, W)
        z_flat = self.encoder(obs_flat)         # (B*T, D)
        z_seq = z_flat.reshape(B, T, -1)        # (B, T, D)

        # Predict next embeddings (teacher forcing)
        z_hat = self.predictor(z_seq, a_seq)    # (B, T, D)

        # Prediction loss: z_hat[:, :-1] vs z_seq[:, 1:]
        if pred_action_weights is None:
            pred_loss = F.mse_loss(
                z_hat[:, :-1],    # predicted z_{2:T}
                z_seq[:, 1:].detach()  # target z_{2:T}
            )
        else:
            err = (z_hat[:, :-1] - z_seq[:, 1:].detach()) ** 2  # (B, T-1, D)
            per_step = err.mean(dim=-1)                         # (B, T-1)
            a_idx = a_seq[:, :-1].argmax(dim=-1).long()         # (B, T-1)
            w = pred_action_weights[a_idx].clamp(min=0.0)       # (B, T-1)
            pred_loss = (per_step * w).mean()

        # SIGReg on all embeddings (step-wise, per paper Alg.1)
        # Apply per-timestep: average SIGReg over T steps
        sigreg_loss = torch.tensor(0.0, device=obs_seq.device)
        for t in range(T):
            sigreg_loss = sigreg_loss + self.sigreg(z_seq[:, t])
        sigreg_loss = sigreg_loss / T

        total_loss = pred_loss + self.sigreg_lambda * sigreg_loss

        # Optional multi-step open-loop rollout loss.
        # Unrolls the predictor open-loop for rollout_steps steps starting
        # from a random position, applying the same norm rescaling used at
        # inference, supervised against ground-truth encoded latents.
        # This closes the teacher-forcing / open-loop distribution gap.
        rollout_loss_val = 0.0
        if rollout_steps > 0 and T > rollout_steps + 1:
            s = random.randint(0, T - rollout_steps - 1)
            z_r = z_seq[:, s]                                     # (B, D)
            acc = torch.tensor(0.0, device=obs_seq.device)
            for k in range(rollout_steps):
                z_r = self.predictor(
                    z_r.unsqueeze(1),
                    a_seq[:, s + k].unsqueeze(1),
                )[:, -1]                                          # (B, D)
                z_r = F.normalize(z_r, dim=-1) * self.latent_scale
                acc = acc + F.mse_loss(z_r, z_seq[:, s + k + 1].detach())
            rollout_loss = acc / rollout_steps
            total_loss = total_loss + rollout_loss_weight * rollout_loss
            rollout_loss_val = rollout_loss.detach()

        out = {
            'loss': total_loss,
            'pred_loss': pred_loss.detach(),
            'sigreg_loss': sigreg_loss.detach(),
            'z_seq': z_seq,
            'z_hat': z_hat,
        }
        if rollout_steps > 0:
            out['rollout_loss'] = rollout_loss_val
        return out

    @torch.no_grad()
    def rollout(self, z0, a_seq):
        """
        Autoregressive rollout for planning.
        z0:    (B, latent_dim) — initial latent
        a_seq: (B, H, action_dim) — action sequence to evaluate
        Returns z_pred: (B, H+1, latent_dim)

        Fixes vs original:
        - Sliding context window: history fed to the predictor is clamped to
          the last context_len frames so positional embeddings are never
          addressed out of range.
        - Norm stabilisation: each predicted latent is rescaled to
          latent_scale (empirical mean encoder-output norm) before being
          appended to the history, preventing the scale explosion observed in
          open-loop evaluation (open-loop MSE ~4B vs teacher-forced ~0.2).
        """
        B, H, _ = a_seq.shape
        ctx = self.predictor.context_len
        z_history = [z0]

        for t in range(H):
            # Clamp history to the last ctx frames to respect pos embeddings
            window = z_history[-ctx:]
            z_stack = torch.stack(window, dim=1)             # (B, W, D)
            a_start = max(0, t + 1 - ctx)
            a_stack = a_seq[:, a_start:t + 1]                # (B, W, A)
            z_hat = self.predictor(z_stack, a_stack)         # (B, W, D)
            z_next = z_hat[:, -1]                            # (B, D)
            # Rescale to encoder-output norm to prevent scale drift
            z_next = F.normalize(z_next, dim=-1) * self.latent_scale
            z_history.append(z_next)

        return torch.stack(z_history, dim=1)  # (B, H+1, D)