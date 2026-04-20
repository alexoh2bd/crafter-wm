"""Train ActionEncoder + HighLevelPredictor on human playthrough trajectories.

The LeWM encoder is loaded frozen from logs/lewm_teacher_deep/best.pt.
All frames are pre-encoded once and cached in data/latents.npz to avoid
re-running the expensive ViT encoder every epoch.

Training objective (waypoint triplet loss):
    Sample (t1, t2, t3) with t3 - t1 ≤ 64 from each episode.
    l1    = ActionEncoder(one_hot(actions[t1:t2]))
    ẑ_t2  = HighLevelPredictor(l1, Z[t1])           # single step
    l2    = ActionEncoder(one_hot(actions[t2:t3]))
    ẑ_t3  = HighLevelPredictor(l2, ẑ_t2)            # chained step
    loss  = L1(ẑ_t2, Z[t2]) + L1(ẑ_t3, Z[t3]) + λ * SIGReg(ẑ_t2)

Usage
-----
    python src/hwm/train_hwm_high.py
    python src/hwm/train_hwm_high.py --epochs 50 --batch_size 256 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import (
    ACTION_DIM,
    CHECKPOINT,
    LATENT_DIM,
    LATENTS_CACHE,
    PROJECT_ROOT,
    TRAJ_DATASET,
    HWM_HIGH_CKPT,
)
from hwm.models import ActionEncoder, HighLevelPredictor, load_lewm, one_hot_actions
from lemodel import SIGReg


# ── Pre-encode all frames with the frozen LeWM encoder ───────────────────────

@torch.no_grad()
def precompute_latents(
    traj_dataset_path: str,
    latents_path: str,
    lewm_ckpt: str,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode every frame in trajectory_dataset.npz using the frozen LeWM encoder.

    Saves latents.npz with arrays Z, actions, trajectory_boundaries.
    Returns (Z, actions, trajectory_boundaries) as numpy arrays.
    """
    cache = Path(latents_path)
    if cache.exists():
        print(f"Loading cached latents from {cache}")
        d = np.load(latents_path)
        return d["Z"], d["actions"], d["trajectory_boundaries"]

    print("Pre-encoding frames with frozen LeWM encoder...")
    model, _ = load_lewm(lewm_ckpt, device=device)
    model.eval()

    d = np.load(traj_dataset_path)
    obs         = d["obs"]                  # (N, 64, 64, 3) uint8
    actions     = d["actions"]              # (N,) int64
    boundaries  = d["trajectory_boundaries"]  # (K,) int64
    N = len(obs)

    Z_list = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(obs[start:end]).float() / 255.0  # (B, 64, 64, 3)
        batch = batch.permute(0, 3, 1, 2).to(device)              # (B, 3, 64, 64)
        z = model.encode(batch).cpu().numpy()                      # (B, 256)
        Z_list.append(z)
        if (start // batch_size) % 20 == 0:
            print(f"  encoded {end}/{N} frames")

    Z = np.concatenate(Z_list, axis=0)  # (N, 256)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(latents_path, Z=Z, actions=actions, trajectory_boundaries=boundaries)
    print(f"Cached latents → {cache}  shape={Z.shape}")
    return Z, actions, boundaries


# ── Dataset ───────────────────────────────────────────────────────────────────

class WaypointTripletDataset(torch.utils.data.Dataset):
    """Samples waypoint triplets (t1, t2, t3) from pre-encoded trajectories.

    Each item: (Z[t1], l1_actions, Z[t2], l2_actions, Z[t3])
      where l1_actions = actions[t1:t2]  (variable length)
            l2_actions = actions[t2:t3]  (variable length)
    Sequences are zero-padded to max_subseq_len for batching.
    """

    def __init__(
        self,
        Z: np.ndarray,
        actions: np.ndarray,
        boundaries: np.ndarray,
        n_triplets_per_episode: int = 16,
        max_window: int = 64,
        min_gap: int = 2,
        max_subseq_len: int = 32,
        seed: int = 0,
    ) -> None:
        self.Z = Z
        self.actions = actions
        self.max_subseq_len = max_subseq_len
        self.action_dim = ACTION_DIM

        rng = np.random.default_rng(seed)
        self.triplets: list[tuple[int, int, int]] = []

        n_eps = len(boundaries)
        ep_ends = np.concatenate([boundaries[1:], [len(Z)]])

        for ep_i in range(n_eps):
            start = int(boundaries[ep_i])
            end   = int(ep_ends[ep_i])
            T = end - start
            if T < max_window + 1:
                continue

            for _ in range(n_triplets_per_episode):
                t1 = rng.integers(start, end - max_window)
                t2 = rng.integers(t1 + min_gap, t1 + max_window // 2)
                t3 = rng.integers(t2 + min_gap, t1 + max_window)
                t2 = min(t2, end - 1)
                t3 = min(t3, end - 1)
                if t3 > t2 > t1:
                    self.triplets.append((t1, t2, t3))

        print(f"WaypointTripletDataset: {len(self.triplets)} triplets from {n_eps} episodes")

    def __len__(self) -> int:
        return len(self.triplets)

    def _pad_actions(self, actions_slice: np.ndarray) -> torch.Tensor:
        """Pad/truncate action slice to max_subseq_len and convert to one-hot."""
        L = min(len(actions_slice), self.max_subseq_len)
        a = actions_slice[:L].astype(np.int64)
        a_oh = np.eye(self.action_dim, dtype=np.float32)[a]        # (L, 17)
        pad = np.zeros((self.max_subseq_len - L, self.action_dim), dtype=np.float32)
        return torch.from_numpy(np.concatenate([a_oh, pad], axis=0))  # (max_subseq_len, 17)

    def __getitem__(self, idx: int):
        t1, t2, t3 = self.triplets[idx]
        z1 = torch.from_numpy(self.Z[t1].copy())
        z2 = torch.from_numpy(self.Z[t2].copy())
        z3 = torch.from_numpy(self.Z[t3].copy())
        l1_actions = self._pad_actions(self.actions[t1:t2])
        l2_actions = self._pad_actions(self.actions[t2:t3])
        return z1, l1_actions, z2, l2_actions, z3


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Pre-encode ───────────────────────────────────────────────────────────
    if getattr(args, "force_reencode", False) and Path(args.latents_cache).exists():
        print(f"--force_reencode: removing stale cache {args.latents_cache}")
        Path(args.latents_cache).unlink()

    Z, actions, boundaries = precompute_latents(
        traj_dataset_path=args.traj_dataset,
        latents_path=args.latents_cache,
        lewm_ckpt=args.checkpoint,
        device=device,
        batch_size=args.encode_batch,
    )

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    dataset = WaypointTripletDataset(
        Z, actions, boundaries,
        n_triplets_per_episode=args.triplets_per_episode,
        max_window=args.max_window,
        max_subseq_len=args.max_subseq_len,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    action_enc  = ActionEncoder(
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        max_len=args.max_subseq_len,
    ).to(device)

    high_pred = HighLevelPredictor(
        latent_dim=LATENT_DIM,
        depth=6,
        num_heads=16,
        dropout=0.1,
        context_len=3,
    ).to(device)

    sigreg = SIGReg(embed_dim=LATENT_DIM, M=1024).to(device)

    params = list(action_enc.parameters()) + list(high_pred.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    best_val_loss = float("inf")

    use_wandb = getattr(args, "wandb", False)
    wandb_run = None
    if use_wandb:
        if _WANDB_AVAILABLE:
            run_name = getattr(args, "wandb_run_name", None) or "hwm-high-train"
            wandb_run = _wandb.init(
                project=getattr(args, "wandb_project", "lewm-crafter"),
                name=run_name,
                config=vars(args),
            )
        else:
            print("Warning: --wandb set but wandb is not installed; "
                  "install with `pip install wandb`")

    print(f"Training ActionEncoder + HighLevelPredictor for {args.epochs} epochs")
    print(f"  Batches/epoch: {len(loader)}   Batch size: {args.batch_size}")

    global_step = 0
    for epoch in range(args.epochs):
        action_enc.train()
        high_pred.train()
        epoch_loss      = 0.0
        epoch_pred_loss = 0.0
        epoch_sr_loss   = 0.0
        t0 = time.time()

        for z1, l1_a, z2, l2_a, z3 in loader:
            z1  = z1.to(device)   # (B, D)
            z2  = z2.to(device)
            z3  = z3.to(device)
            l1_a = l1_a.to(device)  # (B, L, 17)
            l2_a = l2_a.to(device)

            # Encode action subsequences into macro-actions
            l1 = action_enc(l1_a)  # (B, D)
            l2 = action_enc(l2_a)  # (B, D)

            # High-level prediction: one step from z1 conditioned on l1
            # Shape: z_seq (B,1,D), l_seq (B,1,D) -> z_hat (B,1,D)
            z_hat_t2 = high_pred(
                z1.unsqueeze(1), l1.unsqueeze(1)
            ).squeeze(1)  # (B, D)

            # Chained: one step from ẑ_t2 conditioned on l2
            z_hat_t3 = high_pred(
                z_hat_t2.unsqueeze(1), l2.unsqueeze(1)
            ).squeeze(1)  # (B, D)

            # Prediction losses
            pred_loss = (
                F.l1_loss(z_hat_t2, z2.detach())
                + F.l1_loss(z_hat_t3, z3.detach())
            )

            # SIGReg on ẑ_t2 only, per-step (B, D) slice
            sr_loss = sigreg(z_hat_t2)

            loss = pred_loss + args.sigreg_lambda * sr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss      += loss.item()
            epoch_pred_loss += pred_loss.item()
            epoch_sr_loss   += sr_loss.item()

            if wandb_run is not None:
                _wandb.log(
                    {
                        "train/loss":      loss.item(),
                        "train/pred_loss": pred_loss.item(),
                        "train/sr_loss":   sr_loss.item(),
                        "train/lr":        scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
            global_step += 1

        n_batches = len(loader)
        epoch_loss      /= n_batches
        epoch_pred_loss /= n_batches
        epoch_sr_loss   /= n_batches
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log_entry = {
            "epoch":     epoch,
            "loss":      epoch_loss,
            "pred_loss": epoch_pred_loss,
            "sr_loss":   epoch_sr_loss,
            "lr":        lr_now,
            "time":      elapsed,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if wandb_run is not None:
            _wandb.log(
                {
                    "epoch/loss":      epoch_loss,
                    "epoch/pred_loss": epoch_pred_loss,
                    "epoch/sr_loss":   epoch_sr_loss,
                    "epoch/lr":        lr_now,
                    "epoch/time_s":    elapsed,
                },
                step=global_step,
            )

        if (epoch + 1) % args.log_every == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={epoch_loss:.4f}  "
                  f"lr={lr_now:.2e}  t={elapsed:.1f}s")

        # Save best checkpoint
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "action_encoder": action_enc.state_dict(),
                    "high_predictor":  high_pred.state_dict(),
                    "args": vars(args),
                },
                log_dir / "best.pt",
            )

    # Always save final checkpoint
    torch.save(
        {
            "epoch": args.epochs - 1,
            "loss": epoch_loss,
            "action_encoder": action_enc.state_dict(),
            "high_predictor":  high_pred.state_dict(),
            "args": vars(args),
        },
        log_dir / "latest.pt",
    )

    # Compute empirical macro-action distribution for cem_high initialisation.
    # A single post-training pass over the full dataset collects all ActionEncoder
    # outputs.  mean/std are saved into both checkpoints so cem_high can seed its
    # Gaussian prior from the actual output distribution instead of N(0, I).
    print("Computing empirical macro-action mean/std (post-training pass)...")
    action_enc.eval()
    all_l: list[torch.Tensor] = []
    with torch.no_grad():
        for z1, l1_a, z2, l2_a, z3 in loader:
            l1_a = l1_a.to(device)
            l2_a = l2_a.to(device)
            all_l.append(action_enc(l1_a).cpu())
            all_l.append(action_enc(l2_a).cpu())
    all_l_cat = torch.cat(all_l, dim=0)                         # (N_total, D)
    macro_mean = all_l_cat.mean(dim=0)                          # (D,)
    macro_std  = all_l_cat.std(dim=0).clamp(min=1e-4)          # (D,)

    for ckpt_path in [log_dir / "best.pt", log_dir / "latest.pt"]:
        if ckpt_path.exists():
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            ckpt_data["macro_action_mean"] = macro_mean
            ckpt_data["macro_action_std"]  = macro_std
            torch.save(ckpt_data, ckpt_path)
            print(f"  Patched {ckpt_path.name} with macro-action stats")

    if wandb_run is not None:
        _wandb.summary["best_loss"] = best_val_loss
        _wandb.finish()

    print(f"\nDone. Best loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {log_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HWM high-level modules (ActionEncoder + HighLevelPredictor)"
    )
    parser.add_argument("--checkpoint",           default=CHECKPOINT)
    parser.add_argument("--traj_dataset",         default=TRAJ_DATASET)
    parser.add_argument("--latents_cache",        default=LATENTS_CACHE)
    parser.add_argument("--logdir",               default=str(
        PROJECT_ROOT / "data" / "crafter" / "world_model" / "hwm_high"))
    parser.add_argument("--epochs",               type=int,   default=30)
    parser.add_argument("--batch_size",           type=int,   default=256)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--weight_decay",         type=float, default=1e-4)
    parser.add_argument("--sigreg_lambda",        type=float, default=0.2)
    parser.add_argument("--encode_batch",         type=int,   default=512)
    parser.add_argument("--triplets_per_episode", type=int,   default=16)
    parser.add_argument("--max_window",           type=int,   default=64)
    parser.add_argument("--max_subseq_len",       type=int,   default=32)
    parser.add_argument("--log_every",            type=int,   default=5)
    parser.add_argument("--seed",                 type=int,   default=0)
    parser.add_argument("--wandb",                action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",        default="lewm-crafter")
    parser.add_argument("--wandb_run_name",       default=None)
    parser.add_argument("--force_reencode",       action="store_true",
                        help="Delete cached latents.npz and re-encode from "
                             "trajectory_dataset.npz (use after rebuilding the dataset "
                             "with a new train/eval split)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
