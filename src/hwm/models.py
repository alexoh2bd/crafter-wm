"""HWM-specific model components.

New modules only — does not modify lemodel.py.

Classes
-------
ActionEncoder
    Maps a variable-length primitive action sequence to a single macro-action
    vector in R^{LATENT_DIM}.

HighLevelPredictor
    Single-step latent predictor conditioned on macro-actions (l ∈ R^{LATENT_DIM})
    via AdaLN, identical in architecture to the low-level Predictor in lemodel.py
    but with a macro-action conditioning interface.

Helpers
-------
load_lewm(device)
    Reconstruct LeWM from saved args in best.pt and return (model, ckpt_args).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make src/ importable when run directly from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import ACTION_DIM, CHECKPOINT, CONTEXT_LEN, LATENT_DIM, MACRO_DIM, NAV_ACTION_THRESHOLD
from lemodel import AdaLN, LeWM, PredictorBlock, SIGReg


# ── ActionEncoder ─────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """Encode a primitive action sequence into a single macro-action vector.

    Architecture: learned CLS token + 2-layer bidirectional TransformerEncoder
    (no causal mask) + linear projection to macro_dim.

    Args:
        action_dim: Dimension of the one-hot action input (default 17).
        hidden_dim: Internal transformer width (default 192, matching ViT-Tiny).
        latent_dim: LeWM latent dimension — kept for pos-embed sizing (default 256).
        macro_dim:  Output macro-action dimension (default MACRO_DIM=32).
                    Smaller than latent_dim makes CEM search tractable.
        depth:      Number of transformer layers (default 2).
        num_heads:  Number of attention heads (default 3).
        max_len:    Maximum expected sub-trajectory length for positional encoding.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 192,
        latent_dim: int = LATENT_DIM,
        macro_dim: int = MACRO_DIM,
        depth: int = 2,
        num_heads: int = 3,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.macro_dim  = macro_dim

        # Project one-hot actions into hidden space
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Learned CLS token — prepended to the sequence before encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding for positions 0..max_len (0 = CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Project CLS output to macro-action space (macro_dim, not latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, macro_dim),
            nn.BatchNorm1d(macro_dim),
        )

    def forward(self, a_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_seq: (B, L, action_dim) float one-hot action sequence.

        Returns:
            l: (B, macro_dim) macro-action vector.
        """
        B, L, _ = a_seq.shape

        # Project actions into hidden space
        x = self.action_proj(a_seq)              # (B, L, H)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        x = torch.cat([cls, x], dim=1)           # (B, L+1, H)

        # Add positional embeddings (truncate if L+1 > max_len+1)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]

        # Bidirectional transformer (no mask)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        cls_out = x[:, 0]              # (B, H) — CLS token
        l = self.projector(cls_out)    # (B, latent_dim)
        return l


# ── SegmentedActionEncoder ────────────────────────────────────────────────────

class SegmentedActionEncoder(nn.Module):
    """ActionEncoder with separate nav and interaction CLS tokens.

    Motivation: a Crafter inter-waypoint segment typically consists of a
    navigation run (move actions) followed by an interaction run (do /
    crafting actions).  Encoding the whole sequence as one homogeneous bag
    forces the encoder to solve two unrelated compression problems at once.
    This variant tags each action token with a type embedding (nav=0,
    interact=1) and maintains two separate CLS tokens so that the
    Transformer can route nav content to one readout and interaction content
    to the other.  Their concatenated outputs are projected to macro_dim.

    Interface is identical to ActionEncoder — forward(a_seq) → (B, macro_dim).

    Architecture:
        - action_proj:  Linear(action_dim, hidden_dim)
        - type_embed:   Embedding(2, hidden_dim)      — nav vs. interact tag
        - pos_embed:    (1, max_len, hidden_dim)       — for action tokens only
        - cls_tokens:   (1, 2, hidden_dim)             — [nav_cls, interact_cls]
        - blocks:       2-layer bidirectional TransformerEncoder
        - projector:    Linear(2*hidden_dim, macro_dim) + BN
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 192,
        latent_dim: int = LATENT_DIM,
        macro_dim: int = MACRO_DIM,
        depth: int = 2,
        num_heads: int = 3,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.macro_dim  = macro_dim

        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Two CLS tokens: index 0 = nav aggregator, index 1 = interact aggregator
        self.cls_tokens = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        # Binary type embedding added to every action token before attention
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Positional encoding for action token positions only (0..max_len-1)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Input is 2*hidden_dim (nav_cls ∥ interact_cls)
        self.projector = nn.Sequential(
            nn.Linear(2 * hidden_dim, macro_dim),
            nn.BatchNorm1d(macro_dim),
        )

    def forward(self, a_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_seq: (B, L, action_dim) float one-hot action sequence.

        Returns:
            l: (B, macro_dim) macro-action vector.
        """
        B, L, _ = a_seq.shape

        # Derive type ids from the argmax of the one-hot vector.
        # Padded (all-zero) positions have argmax=0 (noop → nav), which is fine
        # because padded steps are masked out of the loss anyway.
        action_idx = a_seq.argmax(dim=-1)                          # (B, L)
        type_ids   = (action_idx >= NAV_ACTION_THRESHOLD).long()   # (B, L) 0=nav,1=interact

        # Project actions and add type + position embeddings
        x = self.action_proj(a_seq)            # (B, L, H)
        x = x + self.type_embed(type_ids)      # (B, L, H)
        x = x + self.pos_embed[:, :L, :]       # (B, L, H)

        # Prepend both CLS tokens (no positional embedding — global aggregators)
        cls = self.cls_tokens.expand(B, -1, -1)   # (B, 2, H)
        x = torch.cat([cls, x], dim=1)            # (B, L+2, H)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # nav_cls at position 0, interact_cls at position 1
        combined = torch.cat([x[:, 0], x[:, 1]], dim=-1)  # (B, 2*H)
        return self.projector(combined)                     # (B, macro_dim)


# ── HighLevelPredictor ────────────────────────────────────────────────────────

class HighLevelPredictor(nn.Module):
    """Single-step latent-space predictor conditioned on macro-actions.

    Shares the same PredictorBlock / AdaLN architecture as lemodel.Predictor
    but accepts macro-action vectors l ∈ R^{latent_dim} instead of one-hot
    primitive actions.  The output passes through a Linear+BN projector so it
    lives in the same R^{latent_dim} space as the LeWM encoder output.

    At inference, called one step at a time:
        ẑ_next = high_predictor(l, z_curr)  # (B, latent_dim)

    During training, can be chained autoregressively to produce a 2-step
    trajectory from triplet waypoints.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        macro_dim: int = MACRO_DIM,
        depth: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        context_len: int = 3,   # H_hi = 3 macro-steps
    ) -> None:
        super().__init__()
        self.latent_dim  = latent_dim
        self.macro_dim   = macro_dim
        self.context_len = context_len

        # Macro-action → conditioning signal.
        # First linear projects from macro_dim → latent_dim so that CEM can
        # search in the smaller R^macro_dim space while the predictor
        # processes everything internally in R^latent_dim.
        self.action_embed = nn.Sequential(
            nn.Linear(macro_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, context_len, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock(latent_dim, num_heads, latent_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output projector — must match LeWM encoder output space
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        l_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_seq: (B, T, latent_dim)  — sequence of latent states
            l_seq: (B, T, latent_dim)  — corresponding macro-action vectors

        Returns:
            z_hat: (B, T, latent_dim)  — predicted next latent states
        """
        B, T, D = z_seq.shape
        cond = self.action_embed(l_seq)          # (B, T, D)

        x = z_seq + self.pos_embed[:, :T, :]

        # Causal mask so position i only attends to positions ≤ i
        causal_mask = torch.triu(
            torch.ones(T, T, device=z_seq.device), diagonal=1
        ).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

        for block in self.blocks:
            x = block(x, cond, causal_mask=causal_mask)

        x_flat = x.reshape(B * T, D)
        z_hat_flat = self.projector(x_flat)
        z_hat = z_hat_flat.reshape(B, T, D)
        return z_hat

    @torch.no_grad()
    def rollout(
        self,
        z0: torch.Tensor,
        l_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive rollout for high-level planning.

        Args:
            z0:    (B, latent_dim)     — initial latent state
            l_seq: (B, H, latent_dim)  — macro-action sequence

        Returns:
            z_traj: (B, H+1, latent_dim)  — trajectory including z0 at t=0
        """
        B, H, _ = l_seq.shape
        z_history = [z0]

        for t in range(H):
            z_stack = torch.stack(z_history, dim=1)  # (B, t+1, D)
            l_stack = l_seq[:, :t + 1]               # (B, t+1, D)
            z_hat = self.forward(z_stack, l_stack)   # (B, t+1, D)
            z_history.append(z_hat[:, -1])            # take last prediction

        return torch.stack(z_history, dim=1)          # (B, H+1, D)


# ── load_lewm helper ──────────────────────────────────────────────────────────

def load_lewm(
    checkpoint_path: str = CHECKPOINT,
    device: str | torch.device = "cpu",
) -> Tuple[LeWM, dict]:
    """Reconstruct and return a frozen LeWM from a saved checkpoint.

    Reads architecture hyperparameters from `ckpt['args']` so the model always
    matches the checkpoint regardless of current config defaults.

    Returns:
        model: LeWM with weights loaded, set to eval() mode.
        args:  The dict of saved training args.
    """
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("args", {})

    model = LeWM(
        img_size       = saved.get("img_size",        64),
        patch_size     = saved.get("patch_size",       8),
        latent_dim     = saved.get("latent_dim",     LATENT_DIM),
        action_dim     = ACTION_DIM,
        encoder_depth  = saved.get("encoder_depth",  12),
        encoder_heads  = saved.get("encoder_heads",   3),
        predictor_depth= saved.get("predictor_depth", 6),
        predictor_heads= saved.get("predictor_heads", 16),
        context_len    = saved.get("context_len",   CONTEXT_LEN),
        sigreg_M       = saved.get("sigreg_M",      1024),
        sigreg_lambda  = saved.get("sigreg_lambda",  0.1),
        dropout        = saved.get("dropout",         0.1),
    ).to(device)

    # strict=False: latent_scale was added as a registered buffer after some
    # checkpoints were saved; if it is absent the constructor default (20.92)
    # is already correct.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Freeze all parameters — this model is never trained further
    for p in model.parameters():
        p.requires_grad_(False)

    return model, saved


def one_hot_actions(actions: torch.Tensor, action_dim: int = ACTION_DIM) -> torch.Tensor:
    """Convert integer action tensor to one-hot float tensor.

    Args:
        actions: (...) int64 tensor of action indices.
        action_dim: Number of action classes.

    Returns:
        (..., action_dim) float32 one-hot tensor.
    """
    return F.one_hot(actions.long(), num_classes=action_dim).float()
