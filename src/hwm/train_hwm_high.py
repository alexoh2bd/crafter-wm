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
    MACRO_DIM,
    PROJECT_ROOT,
    TRAJ_DATASET,
    HWM_HIGH_CKPT,
)
from hwm.models import ActionEncoder, HighLevelPredictor, SegmentedActionEncoder, load_lewm, one_hot_actions
from lemodel import SIGReg


def chain_forward_loss(
    batch: tuple,
    device: torch.device,
    action_enc: ActionEncoder,
    high_pred: HighLevelPredictor,
    sigreg: SIGReg,
    sigreg_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single chain batch: total loss, pred_loss, sr_loss (same as train loop).

    Growing-T forward pass: at step k the predictor receives the full history
    z_start, z_hat_0, ..., z_hat_{k-1} (T = k+1 tokens) with the corresponding
    k+1 macro-actions.  This matches how rollout() is called at eval time and
    removes the train/eval distribution shift present in the old T=1 formulation.
    """
    z_start, actions_padded, z_targets_padded, chain_len = batch
    z_start = z_start.to(device)
    actions_padded = actions_padded.to(device)
    z_targets_padded = z_targets_padded.to(device)
    chain_len = chain_len.to(device)

    B, Kp1, L, A = actions_padded.shape
    macro_dim = action_enc.macro_dim
    l_all = action_enc(actions_padded.view(B * Kp1, L, A)).view(B, Kp1, macro_dim)

    # Growing-T: feed history of length k+1 at step k (mirrors rollout()).
    z_history = [z_start]   # list of (B, D) tensors
    z_preds = []
    for k in range(Kp1):
        z_stack = torch.stack(z_history, dim=1)  # (B, k+1, D)
        l_stack = l_all[:, :k + 1]               # (B, k+1, macro_dim)
        z_hat_stack = high_pred(z_stack, l_stack) # (B, k+1, D)
        z_hat_k = z_hat_stack[:, -1]              # (B, D)
        z_preds.append(z_hat_k)
        z_history.append(z_hat_k)
    z_hat_all = torch.stack(z_preds, dim=1)

    valid = (
        torch.arange(Kp1, device=device).unsqueeze(0)
        < chain_len.unsqueeze(1)
    )
    mask = valid.unsqueeze(-1).float()
    n_valid = mask.sum().clamp(min=1.0)

    pred_loss = (
        F.l1_loss(
            z_hat_all * mask,
            z_targets_padded.detach() * mask,
            reduction="sum",
        )
        / n_valid
    )
    sr_loss = sigreg(z_hat_all[valid])
    loss = pred_loss + sigreg_lambda * sr_loss
    return loss, pred_loss, sr_loss


def split_chain_train_val(
    dataset: torch.utils.data.Dataset,
    val_frac: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    """Random split of chain samples into train / val. Returns (train, None) if disabled."""
    from torch.utils.data import Subset

    n = len(dataset)
    if val_frac <= 0 or n < 2:
        return dataset, None

    rng_split = np.random.default_rng(seed + 42_001)
    perm = rng_split.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)
    if n - n_val < batch_size:
        print(
            f"  val split skipped: need at least {batch_size} train chains "
            f"(n={n}); use smaller --batch_size or --val_frac"
        )
        return dataset, None

    val_ix = perm[:n_val].tolist()
    train_ix = perm[n_val:].tolist()
    print(f"  Chain train/val split: {len(train_ix)} train, {len(val_ix)} val ({val_frac:.0%})")
    return Subset(dataset, train_ix), Subset(dataset, val_ix)


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


# ── Chain dataset ─────────────────────────────────────────────────────────────

class WaypointChainDataset(torch.utils.data.Dataset):
    """Samples variable-length waypoint chains from pre-encoded trajectories.

    Each item contains a start state plus K intermediate+final target states,
    where K is drawn uniformly from [n_intermediates_min, n_intermediates_max].
    The K action segments between consecutive waypoints are padded to
    (max_intermediates + 1) steps so batching requires no custom collation.

    **max_window** (when ``full_episode`` is False): bounds how far ``t_start``
    and ``t_end`` can be apart, so each example is a *local* segment of an
    episode—easier targets and similar loss scale to short horizons. Not a model
    architecture limit; the ActionEncoder still sees at most ``max_subseq_len``
    steps *per segment*.

    **full_episode** (``--chain_full_episode``): ignores ``max_window`` and
    anchors ``t_start`` / ``t_end`` at the episode boundaries, sampling K
    random interior waypoints across the full trajectory (episodes must still
    be long enough for ``K`` gaps).

    __getitem__ returns:
        z_start          (D,)                          — anchor latent at t_start
        actions_padded   (max_K+1, max_subseq_len, A) — zero-padded action segs
        z_targets_padded (max_K+1, D)                 — zero-padded target states
        chain_len        int scalar                    — actual valid steps (K+1)
    """

    def __init__(
        self,
        Z: np.ndarray,
        actions: np.ndarray,
        boundaries: np.ndarray,
        n_chains_per_episode: int = 16,
        max_window: int = 64,
        min_gap: int = 2,
        max_subseq_len: int = 32,
        n_intermediates_min: int = 4,
        n_intermediates_max: int = 10,
        full_episode: bool = False,
        seed: int = 0,
    ) -> None:
        assert n_intermediates_min >= 1, "n_intermediates_min must be >= 1"
        assert n_intermediates_max >= n_intermediates_min

        self.Z = Z
        self.actions = actions
        self.max_subseq_len = max_subseq_len
        self.action_dim = ACTION_DIM
        self.max_K = n_intermediates_max  # K = number of intermediate waypoints
        self._full_episode = full_episode

        rng = np.random.default_rng(seed)
        # Each chain stored as (t_start, [t_i1, ..., t_iK], t_end)
        self.chains: list[tuple[int, list[int], int]] = []

        n_eps = len(boundaries)
        ep_ends = np.concatenate([boundaries[1:], [len(Z)]])

        for ep_i in range(n_eps):
            start = int(boundaries[ep_i])
            end   = int(ep_ends[ep_i])
            T = end - start
            # Need enough room for max_K+2 points with min_gap between each
            min_required = (n_intermediates_max + 2) * min_gap + 1
            if T < min_required:
                continue
            if not full_episode and T <= max_window:
                continue

            for _ in range(n_chains_per_episode):
                K = int(rng.integers(n_intermediates_min, n_intermediates_max + 1))
                if full_episode:
                    t_start = start
                    t_end = end - 1
                else:
                    # Local segment: t_start uniform, t_end within max_window span
                    t_start = int(rng.integers(start, end - max_window))
                    t_end = int(rng.integers(
                        t_start + (K + 1) * min_gap,
                        min(t_start + max_window, end - 1) + 1,
                    ))
                    t_end = min(t_end, end - 1)

                # Candidate pool for intermediate waypoints (excluding endpoints)
                pool = np.arange(t_start + min_gap, t_end - min_gap + 1)
                if len(pool) < K:
                    continue
                t_mids = np.sort(rng.choice(pool, size=K, replace=False)).tolist()

                if all(
                    b - a >= min_gap
                    for a, b in zip([t_start] + t_mids, t_mids + [t_end])
                ):
                    self.chains.append((t_start, t_mids, t_end))

        mode = "full_episode" if full_episode else f"max_window={max_window}"
        print(
            f"WaypointChainDataset: {len(self.chains)} chains from {n_eps} episodes "
            f"({mode}, K in [{n_intermediates_min}, {n_intermediates_max}])"
        )

    def __len__(self) -> int:
        return len(self.chains)

    def _pad_actions(self, actions_slice: np.ndarray) -> torch.Tensor:
        L = min(len(actions_slice), self.max_subseq_len)
        a = actions_slice[:L].astype(np.int64)
        a_oh = np.eye(self.action_dim, dtype=np.float32)[a]
        pad = np.zeros((self.max_subseq_len - L, self.action_dim), dtype=np.float32)
        return torch.from_numpy(np.concatenate([a_oh, pad], axis=0))

    def __getitem__(self, idx: int):
        t_start, t_mids, t_end = self.chains[idx]
        waypoints = [t_start] + t_mids + [t_end]
        K = len(t_mids)       # actual intermediates
        chain_len = K + 1     # number of action segments / target states

        z_start = torch.from_numpy(self.Z[t_start].copy())

        # Build action segments and target states for each step
        segs_list = []
        targets_list = []
        for i in range(chain_len):
            ta, tb = waypoints[i], waypoints[i + 1]
            segs_list.append(self._pad_actions(self.actions[ta:tb]))
            targets_list.append(torch.from_numpy(self.Z[tb].copy()))

        # Pad to max_K+1 steps with zeros
        max_steps = self.max_K + 1
        zero_act = torch.zeros(self.max_subseq_len, self.action_dim)
        zero_z   = torch.zeros(self.Z.shape[1])
        while len(segs_list) < max_steps:
            segs_list.append(zero_act)
            targets_list.append(zero_z)

        actions_padded  = torch.stack(segs_list, dim=0)   # (max_K+1, L, A)
        z_targets_padded = torch.stack(targets_list, dim=0)  # (max_K+1, D)

        return z_start, actions_padded, z_targets_padded, torch.tensor(chain_len, dtype=torch.long)


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
    use_chain = getattr(args, "n_intermediates_max", 1) > 1
    val_dataset = None
    if use_chain:
        full_chain_ds = WaypointChainDataset(
            Z, actions, boundaries,
            n_chains_per_episode=args.triplets_per_episode,
            max_window=args.max_window,
            max_subseq_len=args.max_subseq_len,
            n_intermediates_min=args.n_intermediates_min,
            n_intermediates_max=args.n_intermediates_max,
            full_episode=getattr(args, "chain_full_episode", False),
            seed=args.seed,
        )
        train_dataset, val_dataset = split_chain_train_val(
            full_chain_ds,
            getattr(args, "val_frac", 0.0),
            args.batch_size,
            args.seed,
        )
    else:
        train_dataset = WaypointTripletDataset(
            Z, actions, boundaries,
            n_triplets_per_episode=args.triplets_per_episode,
            max_window=args.max_window,
            max_subseq_len=args.max_subseq_len,
            seed=args.seed,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=2, pin_memory=True, drop_last=False,
        )

    # ── Models ───────────────────────────────────────────────────────────────
    macro_dim = getattr(args, "macro_dim", MACRO_DIM)
    use_segmented = getattr(args, "segmented_encoder", False)

    EncCls = SegmentedActionEncoder if use_segmented else ActionEncoder
    action_enc = EncCls(
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        macro_dim=macro_dim,
        max_len=args.max_subseq_len,
    ).to(device)

    high_pred = HighLevelPredictor(
        latent_dim=LATENT_DIM,
        macro_dim=macro_dim,
        depth=6,
        num_heads=16,
        dropout=0.1,
        context_len=getattr(args, "context_len", 3),
    ).to(device)

    sigreg = SIGReg(embed_dim=LATENT_DIM, M=1024).to(device)

    params = list(action_enc.parameters()) + list(high_pred.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    best_score = float("inf")  # min val loss if val_loader else min train epoch loss
    use_val_for_ckpt = val_loader is not None

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
    print(f"  Batches/epoch: {len(train_loader)}   Batch size: {args.batch_size}")
    if val_loader is not None:
        print(f"  Val batches/epoch: {len(val_loader)}")

    global_step = 0
    for epoch in range(args.epochs):
        action_enc.train()
        high_pred.train()
        epoch_loss      = 0.0
        epoch_pred_loss = 0.0
        epoch_sr_loss   = 0.0
        t0 = time.time()

        for batch in train_loader:
            if use_chain:
                loss, pred_loss, sr_loss = chain_forward_loss(
                    batch, device, action_enc, high_pred, sigreg, args.sigreg_lambda
                )
            else:
                # ── Triplet branch (original) ─────────────────────────────────
                z1, l1_a, z2, l2_a, z3 = batch
                z1   = z1.to(device)
                z2   = z2.to(device)
                z3   = z3.to(device)
                l1_a = l1_a.to(device)
                l2_a = l2_a.to(device)

                l1 = action_enc(l1_a)
                l2 = action_enc(l2_a)

                # Growing-T: mirror rollout() — step 2 receives history [z1, z_hat_t2].
                z_hat_t2 = high_pred(
                    z1.unsqueeze(1), l1.unsqueeze(1)
                )[:, -1]
                z_hat_t3 = high_pred(
                    torch.stack([z1, z_hat_t2], dim=1),   # (B, 2, D)
                    torch.stack([l1, l2],        dim=1),   # (B, 2, macro_dim)
                )[:, -1]

                pred_loss = (
                    F.l1_loss(z_hat_t2, z2.detach())
                    + F.l1_loss(z_hat_t3, z3.detach())
                )
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

        n_batches = len(train_loader)
        epoch_loss      /= n_batches
        epoch_pred_loss /= n_batches
        epoch_sr_loss   /= n_batches
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        val_loss = val_pred = val_sr = None
        if val_loader is not None:
            action_enc.eval()
            high_pred.eval()
            v_loss = v_pred = v_sr = 0.0
            nv = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vl, vp, vs = chain_forward_loss(
                        vbatch, device, action_enc, high_pred, sigreg,
                        args.sigreg_lambda,
                    )
                    v_loss += vl.item()
                    v_pred += vp.item()
                    v_sr += vs.item()
                    nv += 1
            nv = max(nv, 1)
            val_loss = v_loss / nv
            val_pred = v_pred / nv
            val_sr = v_sr / nv
            action_enc.train()
            high_pred.train()

        log_entry: dict = {
            "epoch":     epoch,
            "loss":      epoch_loss,
            "pred_loss": epoch_pred_loss,
            "sr_loss":   epoch_sr_loss,
            "lr":        lr_now,
            "time":      elapsed,
        }
        if val_loss is not None:
            log_entry["val_loss"] = val_loss
            log_entry["val_pred_loss"] = val_pred
            log_entry["val_sr_loss"] = val_sr

        with open(metrics_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if wandb_run is not None:
            wb_payload = {
                "epoch/loss":      epoch_loss,
                "epoch/pred_loss": epoch_pred_loss,
                "epoch/sr_loss":   epoch_sr_loss,
                "epoch/lr":        lr_now,
                "epoch/time_s":    elapsed,
            }
            if val_loss is not None:
                wb_payload["epoch/val_loss"] = val_loss
                wb_payload["epoch/val_pred_loss"] = val_pred
                wb_payload["epoch/val_sr_loss"] = val_sr
            _wandb.log(wb_payload, step=global_step)

        if (epoch + 1) % args.log_every == 0:
            msg = (
                f"Epoch {epoch+1:3d}/{args.epochs}  loss={epoch_loss:.4f}  "
                f"lr={lr_now:.2e}  t={elapsed:.1f}s"
            )
            if val_loss is not None:
                msg += f"  val_loss={val_loss:.4f}"
            print(msg)

        score = val_loss if val_loss is not None else epoch_loss
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "val_loss": val_loss,
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
        for batch in train_loader:
            if use_chain:
                _, actions_padded, _, chain_len = batch
                actions_padded = actions_padded.to(device)
                chain_len      = chain_len.to(device)
                B, Kp1, L, A  = actions_padded.shape
                l_flat = action_enc(
                    actions_padded.view(B * Kp1, L, A)
                ).view(B, Kp1, action_enc.macro_dim)
                # Only collect valid (non-padded) segments
                valid = (
                    torch.arange(Kp1, device=device).unsqueeze(0)
                    < chain_len.unsqueeze(1)
                )
                all_l.append(l_flat[valid].cpu())
            else:
                z1, l1_a, z2, l2_a, z3 = batch
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
        _wandb.summary["best_score"] = best_score
        _wandb.summary["best_is_val"] = use_val_for_ckpt
        _wandb.finish()

    print(f"\nDone. Best {'val' if use_val_for_ckpt else 'train'} loss: {best_score:.4f}")
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
    parser.add_argument("--max_window",           type=int,   default=128,
                        help="Waypoint window in env steps (default 128; "
                             "covers navigation-then-interaction sequences)")
    parser.add_argument("--max_subseq_len",       type=int,   default=32)
    parser.add_argument("--macro_dim",            type=int,   default=MACRO_DIM,
                        help="ActionEncoder output / CEM search dimension "
                             f"(default {MACRO_DIM}; lower = tractable CEM)")
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
    # ── Chain training args ───────────────────────────────────────────────────
    parser.add_argument("--n_intermediates_min",  type=int,   default=1,
                        help="Minimum number of intermediate waypoints per chain "
                             "(default 1 = classic triplet behaviour)")
    parser.add_argument("--n_intermediates_max",  type=int,   default=1,
                        help="Maximum number of intermediate waypoints per chain "
                             "(default 1 = classic triplet behaviour; set > 1 to "
                             "activate WaypointChainDataset)")
    parser.add_argument("--context_len",          type=int,   default=3,
                        help="Positional-embedding context length for "
                             "HighLevelPredictor (default 3 matches original "
                             "triplet training; set to n_intermediates_max+2 "
                             "when using chain training, e.g. 12 for K=4-10)")
    parser.add_argument("--chain_full_episode",   action="store_true",
                        help="Chain dataset: span each sample from episode start "
                             "to end (K random interior waypoints). Ignores "
                             "--max_window for placement; use long episodes only.")
    parser.add_argument("--val_frac",             type=float, default=0.1,
                        help="Chain training only: fraction of chain samples held "
                             "out for validation loss and best.pt selection. "
                             "0 disables. Default 0.1.")
    parser.add_argument("--segmented_encoder",    action="store_true",
                        help="Use SegmentedActionEncoder instead of ActionEncoder. "
                             "Splits each inter-waypoint action sequence into nav "
                             "(move/noop) and interact (do/craft) segments via "
                             "separate CLS tokens and type embeddings.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
