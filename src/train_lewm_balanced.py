"""
Train LeWM with optional per-action prediction loss weighting and balanced
sequence sampling (oversample rare / no-op crafting transitions).

Designed for PPO-teacher or GPU-collected rollout pickles (same schema as
``ledata.collect_crafter_data`` / ``CrafterDataset``).

Does *not* add a heteroscedastic head (that would require changing ``Predictor``
in ``lemodel.py``); use plain MSE + SIGReg like ``letrain.py``.

Weighted loss is implemented via ``LeWM.forward(..., pred_action_weights=...)``.

``torch.compile`` is applied on a single GPU when ``compile_model`` is true; it is
skipped under ``DataParallel`` (multi-GPU) for compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

from ledata import ACTION_DIM
from lemodel import LeWM


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    flat: dict = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def load_pickle_trajectories(path: str | Path) -> list[dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return list(data["trajectories"])


def merge_trajectories(paths: list[str | Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        out.extend(load_pickle_trajectories(p))
    return out


def build_sliding_sequences(
    trajectories: list[dict],
    context_len: int,
    stride: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (obs_chunk (T,H,W,C) uint8, act_chunk (T,) int64)."""
    if stride is None:
        stride = max(1, context_len // 2)
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    for traj in trajectories:
        obs = traj["obs"]
        actions = traj["actions"]
        tlen = len(obs)
        if tlen < context_len + 1:
            continue
        for start in range(0, tlen - context_len, stride):
            end = start + context_len
            obs_chunk = np.stack(obs[start:end])
            act_chunk = np.asarray(actions[start:end], dtype=np.int64)
            seqs.append((obs_chunk, act_chunk))
    return seqs


def count_actions_in_sequences(
    sequences: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    counts = np.zeros(ACTION_DIM, dtype=np.int64)
    for _, act_chunk in sequences:
        for a in act_chunk:
            counts[int(a)] += 1
    return counts


def compute_action_weights(
    counts: np.ndarray,
    strategy: str,
    cap: float,
    beta: float = 0.999,
) -> np.ndarray:
    """Per-action multipliers; normalized to mean 1, then capped."""
    total = float(counts.sum())
    freqs = counts.astype(np.float64) / max(total, 1.0)
    a = ACTION_DIM
    if strategy == "none":
        w = np.ones(a, dtype=np.float64)
    elif strategy == "inv_freq":
        w = 1.0 / (freqs + 1e-8)
    elif strategy == "inv_sqrt":
        w = 1.0 / np.sqrt(freqs + 1e-8)
    elif strategy == "effective_num":
        n_c = counts.astype(np.float64)
        effective_num = (1.0 - beta**n_c) / max(1.0 - beta, 1e-8)
        w = 1.0 / (effective_num + 1e-8)
    else:
        raise ValueError(f"Unknown weight_strategy: {strategy}")
    w = w / w.mean()
    w = np.minimum(w, cap)
    return w.astype(np.float32)


def transition_is_noop_mse(obs_hwc: np.ndarray, t: int, thresh: float) -> bool:
    o0 = obs_hwc[t].astype(np.float32) / 255.0
    o1 = obs_hwc[t + 1].astype(np.float32) / 255.0
    return float(np.mean((o0 - o1) ** 2)) < thresh


def sequence_category(
    obs_chunk: np.ndarray,
    act_chunk: np.ndarray,
    oversample_targets: set[int],
    noop_thresh: float,
) -> int:
    """0 = normal, 1 = has target action on a visual no-op, 2 = target active only."""
    has_noop_target = False
    has_active_target = False
    for t in range(len(act_chunk) - 1):
        a = int(act_chunk[t])
        if a not in oversample_targets:
            continue
        if transition_is_noop_mse(obs_chunk, t, noop_thresh):
            has_noop_target = True
        else:
            has_active_target = True
    if has_noop_target:
        return 1
    if has_active_target:
        return 2
    return 0


class CrafterSequenceDataset(Dataset):
    """One item: (obs_seq (T,C,H,W) float32, a_seq (T, A) one-hot)."""

    def __init__(self, sequences: list[tuple[np.ndarray, np.ndarray]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        obs_chunk, act_chunk = self.sequences[idx]
        obs_t = torch.from_numpy(obs_chunk).float() / 255.0
        obs_t = obs_t.permute(0, 3, 1, 2)
        act_t = torch.zeros(len(act_chunk), ACTION_DIM)
        for i, a in enumerate(act_chunk):
            act_t[i, int(a)] = 1.0
        return obs_t, act_t


def sample_balanced_batch(
    train_indices: np.ndarray,
    categories: np.ndarray,
    batch_size: int,
    oversample_targets: list[int],
    oversample_factor: float,
    noop_ratio: float,
    rng: np.random.Generator,
) -> list[int]:
    normal = train_indices[categories[train_indices] == 0].tolist()
    t_noop = train_indices[categories[train_indices] == 1].tolist()
    t_act = train_indices[categories[train_indices] == 2].tolist()

    p_target = min(oversample_factor * len(oversample_targets) / ACTION_DIM, 0.5)
    n_target = int(round(batch_size * p_target))
    n_normal = batch_size - n_target
    n_noop = int(round(n_target * noop_ratio))
    n_active = n_target - n_noop

    def pick(pool: list[int], k: int) -> list[int]:
        if k <= 0 or len(pool) == 0:
            return []
        return rng.choice(pool, size=k, replace=True).tolist()

    batch = pick(normal, n_normal) + pick(t_noop, n_noop) + pick(t_act, n_active)
    if len(batch) < batch_size:
        batch.extend(
            pick(train_indices.tolist(), batch_size - len(batch))
        )
    rng.shuffle(batch)
    return batch[:batch_size]


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng_split = np.random.default_rng(args.seed + 11)
    rng_batch = np.random.default_rng(args.seed + 22)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if getattr(args, "cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if getattr(args, "allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    paths = [args.data_path]
    if getattr(args, "extra_data_path", None):
        paths.append(args.extra_data_path)
    trajectories = merge_trajectories(paths)
    sequences = build_sliding_sequences(
        trajectories, context_len=args.context_len, stride=args.seq_stride
    )
    if len(sequences) < 10:
        raise RuntimeError(f"Too few sequences ({len(sequences)}); check data paths / context_len.")

    counts = count_actions_in_sequences(sequences)
    freqs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    print(f"Sequences: {len(sequences)}  |  action token counts min/max: {counts.min()}/{counts.max()}")
    print(f"Action freq (first 5): {freqs[:5]}")

    oversample_set = set(int(x) for x in args.oversample_targets)
    cats = np.array(
        [
            sequence_category(obs, act, oversample_set, args.noop_mse_threshold)
            for obs, act in sequences
        ],
        dtype=np.int64,
    )
    print(
        f"Sequence categories: normal={np.sum(cats==0)}  "
        f"target_noop={np.sum(cats==1)}  target_active={np.sum(cats==2)}"
    )

    action_weights_np = compute_action_weights(
        counts,
        strategy=args.weight_strategy,
        cap=args.weight_cap,
        beta=args.effective_num_beta,
    )
    pred_w_tensor = torch.from_numpy(action_weights_np).to(device)

    full_ds = CrafterSequenceDataset(sequences)
    n = len(full_ds)
    n_val = max(1, int(args.val_frac * n))
    n_train = n - n_val
    perm = rng_split.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    steps_per_epoch = max(1, n_train // args.batch_size)
    planned_optimizer_steps = args.epochs * steps_per_epoch
    if getattr(args, "align_total_steps", True):
        if args.total_steps < planned_optimizer_steps:
            print(
                f"align_total_steps: total_steps {args.total_steps} -> {planned_optimizer_steps} "
                f"(epochs={args.epochs} × steps/epoch={steps_per_epoch})"
            )
            args.total_steps = planned_optimizer_steps
    wf = getattr(args, "warmup_frac", None)
    if wf is not None:
        args.warmup_steps = max(1, int(args.total_steps * float(wf)))
        print(f"warmup_frac={wf}: warmup_steps={args.warmup_steps} (total_steps={args.total_steps})")

    val_workers = int(getattr(args, "val_num_workers", 4))
    val_prefetch = int(getattr(args, "val_prefetch_factor", 4))
    val_loader = DataLoader(
        Subset(full_ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=val_workers > 0,
        prefetch_factor=val_prefetch if val_workers > 0 else None,
    )

    model = LeWM(
        img_size=64,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        action_dim=ACTION_DIM,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        context_len=args.context_len,
        sigreg_M=args.sigreg_M,
        sigreg_lambda=args.sigreg_lambda,
        dropout=args.dropout,
    ).to(device)

    resume_path = getattr(args, "resume", None)
    if resume_path and Path(resume_path).exists():
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        sd = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=False)

    if getattr(args, "freeze_encoder", False):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("Encoder frozen")

    n_gpu = torch.cuda.device_count()
    use_compile = (
        getattr(args, "compile_model", True)
        and device.type == "cuda"
        and n_gpu <= 1
    )
    if use_compile:
        mode = getattr(args, "compile_mode", "default")
        if mode not in ("default", "reduce-overhead", "max-autotune"):
            mode = "default"
        try:
            model = torch.compile(model, mode=mode)  # type: ignore[assignment]
            print(f"torch.compile(mode={mode})")
        except Exception as ex:
            print(f"torch.compile skipped: {ex}")

    if n_gpu > 1:
        if getattr(args, "compile_model", True) and device.type == "cuda":
            print("torch.compile disabled: DataParallel multi-GPU")
        print(f"Using {n_gpu} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(
            1, args.total_steps - args.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    log_file = open(logdir / "metrics.jsonl", "w", encoding="utf-8")

    use_wandb = getattr(args, "use_wandb", False)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed; continuing without it")
        else:
            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", "lewm-crafter"),
                name=getattr(args, "wandb_run_name", None) or logdir.name,
                config={k: v for k, v in vars(args).items() if isinstance(v, (bool, int, float, str))},
                dir=str(logdir),
            )

    def log(metrics: dict[str, Any], step: int | None = None) -> None:
        line = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(line)
        log_file.write(json.dumps(metrics, default=str) + "\n")
        log_file.flush()
        if wandb_run is not None:
            import wandb

            payload = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k not in ("epoch",)
            }
            wstep = step if step is not None else metrics.get("step", 0)
            wandb.log(payload, step=int(wstep))

    global_step = 0
    best_val = float("inf")

    train_indices_np = train_idx.astype(np.int64)

    def collate_cpu_batch(ix: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        obs_list, a_list = zip(*(full_ds[i] for i in ix))
        return torch.stack(obs_list, dim=0), torch.stack(a_list, dim=0)

    def draw_train_indices() -> list[int]:
        if args.use_balanced_sampling:
            return sample_balanced_batch(
                train_indices_np,
                cats,
                args.batch_size,
                list(oversample_set),
                args.oversample_factor,
                args.noop_condition_ratio,
                rng_batch,
            )
        return rng_batch.choice(
            train_indices_np, size=args.batch_size, replace=True
        ).tolist()

    prefetch = bool(getattr(args, "prefetch_batches", True))
    batch_pool: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="collate")
        if prefetch
        else None
    )

    try:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()

            if batch_pool is not None:
                first_ix = draw_train_indices()
                pending = batch_pool.submit(collate_cpu_batch, first_ix)
                for step_i in range(steps_per_epoch):
                    obs_seq, a_seq = pending.result()
                    if step_i < steps_per_epoch - 1:
                        next_ix = draw_train_indices()
                        pending = batch_pool.submit(collate_cpu_batch, next_ix)
                    obs_seq = obs_seq.to(device, non_blocking=True)
                    a_seq = a_seq.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    rollout_steps = getattr(args, "rollout_steps", 0)
                    rollout_loss_weight = getattr(args, "rollout_loss_weight", 0.1)
                    p_w = pred_w_tensor if args.use_loss_weighting else None

                    with autocast():
                        m = model.module if isinstance(model, nn.DataParallel) else model
                        out = m(
                            obs_seq,
                            a_seq,
                            rollout_steps=rollout_steps,
                            rollout_loss_weight=rollout_loss_weight,
                            pred_action_weights=p_w,
                        )
                        loss = out["loss"]

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1

                    if global_step % args.log_every == 0:
                        log(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "loss": out["loss"].item(),
                                "pred_loss": float(out["pred_loss"]),
                                "sigreg_loss": float(out["sigreg_loss"]),
                                "lr": optimizer.param_groups[0]["lr"],
                                "time": time.time() - t0,
                            },
                            step=global_step,
                        )
                        t0 = time.time()
            else:
                for _ in range(steps_per_epoch):
                    batch_ix = draw_train_indices()
                    obs_seq, a_seq = collate_cpu_batch(batch_ix)
                    obs_seq = obs_seq.to(device, non_blocking=True)
                    a_seq = a_seq.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    rollout_steps = getattr(args, "rollout_steps", 0)
                    rollout_loss_weight = getattr(args, "rollout_loss_weight", 0.1)
                    p_w = pred_w_tensor if args.use_loss_weighting else None

                    with autocast():
                        m = model.module if isinstance(model, nn.DataParallel) else model
                        out = m(
                            obs_seq,
                            a_seq,
                            rollout_steps=rollout_steps,
                            rollout_loss_weight=rollout_loss_weight,
                            pred_action_weights=p_w,
                        )
                        loss = out["loss"]

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1

                    if global_step % args.log_every == 0:
                        log(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "loss": out["loss"].item(),
                                "pred_loss": float(out["pred_loss"]),
                                "sigreg_loss": float(out["sigreg_loss"]),
                                "lr": optimizer.param_groups[0]["lr"],
                                "time": time.time() - t0,
                            },
                            step=global_step,
                        )
                        t0 = time.time()

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for obs_seq, a_seq in val_loader:
                    obs_seq = obs_seq.to(device, non_blocking=True)
                    a_seq = a_seq.to(device, non_blocking=True)
                    with autocast():
                        m = model.module if isinstance(model, nn.DataParallel) else model
                        out = m(
                            obs_seq,
                            a_seq,
                            rollout_steps=0,
                            pred_action_weights=pred_w_tensor if args.use_loss_weighting else None,
                        )
                    val_losses.append(out["loss"].item())
            val_loss = float(np.mean(val_losses))
            log({"epoch": epoch, "val_loss": val_loss}, step=global_step)

            ckpt = {
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict()
                if not isinstance(model, nn.DataParallel)
                else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(ckpt, logdir / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, logdir / "best.pt")
                print(f"  New best val_loss={val_loss:.4f}")

        print(f"Done. Best val_loss={best_val:.4f}  →  {logdir / 'best.pt'}")
    finally:
        if batch_pool is not None:
            batch_pool.shutdown(wait=True)
        log_file.close()
        if wandb_run is not None:
            import wandb

            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="LeWM training with balanced sampling / loss weights")
    parser.add_argument("--config", type=str, default=None, help="YAML with flattened sections")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--extra_data_path",
        type=str,
        default=None,
        help="Optional second pickle; trajectories are concatenated with data_path",
    )
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--seq_stride", type=int, default=None, help="Sliding window stride (default context_len//2)")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--encoder_depth", type=int, default=None)
    parser.add_argument("--encoder_heads", type=int, default=None)
    parser.add_argument("--predictor_depth", type=int, default=None)
    parser.add_argument("--predictor_heads", type=int, default=None)
    parser.add_argument("--sigreg_M", type=int, default=None)
    parser.add_argument("--sigreg_lambda", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument(
        "--warmup_frac",
        type=float,
        default=None,
        help="If set, warmup_steps = floor(total_steps * warmup_frac) after total_steps align",
    )
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument(
        "--align_total_steps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ensure total_steps >= epochs * (n_train // batch_size) for LR cosine",
    )
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile on single-GPU only (skipped for DataParallel)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=None,
        help="DataLoader workers for validation",
    )
    parser.add_argument(
        "--val_prefetch_factor",
        type=int,
        default=None,
        help="DataLoader prefetch batches per val worker (requires val_num_workers>0)",
    )
    parser.add_argument(
        "--prefetch_batches",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlap CPU batch collation with GPU (background thread)",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.backends.cudnn.benchmark (helps fixed conv shapes)",
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow TF32 matmul on Ampere+",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--rollout_loss_weight", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # Balanced-specific
    parser.add_argument("--use_loss_weighting", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--weight_strategy",
        type=str,
        default=None,
        choices=["none", "inv_freq", "inv_sqrt", "effective_num"],
    )
    parser.add_argument("--weight_cap", type=float, default=None)
    parser.add_argument("--effective_num_beta", type=float, default=0.999)
    parser.add_argument("--use_balanced_sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--oversample_targets",
        type=int,
        nargs="*",
        default=None,
        help="Action indices to oversample (default: crafting-related subset)",
    )
    parser.add_argument("--oversample_factor", type=float, default=None)
    parser.add_argument("--noop_condition_ratio", type=float, default=None)
    parser.add_argument("--noop_mse_threshold", type=float, default=None)

    cli = parser.parse_args()

    defaults: dict[str, Any] = {
        "data_path": "data/crafter/ppo_rollouts/crafter_teacher_data.pkl",
        "extra_data_path": None,
        "logdir": "data/crafter/world_model/lewm_balanced",
        "context_len": 16,
        "seq_stride": None,
        "val_frac": 0.05,
        "latent_dim": 256,
        "encoder_depth": 12,
        "encoder_heads": 3,
        "predictor_depth": 6,
        "predictor_heads": 16,
        "sigreg_M": 1024,
        "sigreg_lambda": 0.1,
        "dropout": 0.1,
        "epochs": 100,
        "batch_size": 128,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1000,
        "warmup_frac": None,
        "total_steps": 200_000,
        "align_total_steps": True,
        "compile_model": True,
        "compile_mode": "default",
        "val_num_workers": 4,
        "val_prefetch_factor": 4,
        "prefetch_batches": True,
        "cudnn_benchmark": True,
        "allow_tf32": True,
        "log_every": 50,
        "seed": 0,
        "rollout_steps": 0,
        "rollout_loss_weight": 0.1,
        "resume": None,
        "freeze_encoder": False,
        "use_wandb": False,
        "wandb_project": "lewm-crafter",
        "wandb_run_name": None,
        "use_loss_weighting": True,
        "weight_strategy": "inv_sqrt",
        "weight_cap": 10.0,
        "effective_num_beta": 0.999,
        "use_balanced_sampling": True,
        "oversample_targets": [7, 8, 11, 12, 14],
        "oversample_factor": 5.0,
        "noop_condition_ratio": 0.5,
        "noop_mse_threshold": 1e-5,
    }

    if cli.config and Path(cli.config).exists():
        defaults.update(_load_config(cli.config))

    for key, val in vars(cli).items():
        if key == "config":
            continue
        if val is not None:
            defaults[key] = val

    if defaults.get("seq_stride") is None:
        defaults["seq_stride"] = max(1, int(defaults["context_len"]) // 2)

    if not defaults.get("data_path"):
        raise SystemExit("data_path is required (CLI or YAML)")

    args = argparse.Namespace(**defaults)
    train(args)


if __name__ == "__main__":
    main()
