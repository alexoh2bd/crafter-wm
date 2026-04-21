"""
LeWM v2 training: mixed random + PPO buffers, predictor hidden dim 384 (ViT-S width).

Run from repo root::

    PYTHONPATH=src python src/hwm/train_lewm_v2.py --logdir logs/lewm_v2_run1
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# src/ on path
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hwm.constants import ACTION_DIM
from hwm.data.mixed_sampler import MixedTransitionSampler
from lemodel import LeWM

DEFAULT_CONFIG = {
    "random_buffer": "data/crafter/ppo_rollouts/crafter_data.pkl",
    "ppo_buffer": "data/crafter/ppo_rollouts/crafter_teacher_data.pkl",
    "random_ratio": 0.7,
    "seq_len": 16,
    "latent_dim": 256,
    "predictor_hidden_dim": 384,
    "context_len": 16,
    "batch_size": 64,
    "n_steps": 100_000,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "warmup_steps": 1000,
    "grad_clip": 1.0,
    "sigreg_lambda": 0.1,
    "sigreg_M": 1024,
    "val_frac": 0.05,
    "val_every": 1000,
    "val_batches": 16,
    "checkpoint_every": 5000,
    "diagnostic_every": 5000,
    "seed": 0,
    "precision": "bf16",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _project_root() / path


def split_trajectories(
    trajs: list, val_frac: float, rng: np.random.Generator
) -> tuple[list, list]:
    order = rng.permutation(len(trajs))
    trajs = [trajs[i] for i in order]
    n_val = max(1, int(len(trajs) * val_frac))
    return trajs[n_val:], trajs[:n_val]


def verify_adaln_zero_init(model: LeWM) -> None:
    for block in model.predictor.blocks:
        for adaln in (block.adaln1, block.adaln2):
            assert adaln.scale.weight.abs().max() < 1e-5, "AdaLN scale not zero-init"
            assert adaln.scale.bias.abs().max() < 1e-5
            assert adaln.shift.weight.abs().max() < 1e-5
            assert adaln.shift.bias.abs().max() < 1e-5


def _evaluate_loss_impl(
    model: LeWM,
    val_sampler: MixedTransitionSampler,
    device: torch.device,
    batch_size: int,
    n_batches: int,
    use_amp: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    dtype = torch.bfloat16 if use_amp else torch.float32
    for _ in range(n_batches):
        obs_np, actions_np = val_sampler.sample_batch(batch_size)
        obs = torch.from_numpy(obs_np).float().div_(255.0).permute(0, 1, 4, 2, 3).to(
            device
        )
        actions = torch.from_numpy(actions_np).long().to(device)
        a_oh = F.one_hot(actions, num_classes=ACTION_DIM).float()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(obs, a_oh, rollout_steps=0)
        losses.append(float(out["loss"].item()))
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def run_action_magnitude_diagnostic(
    model: LeWM,
    val_sampler: MixedTransitionSampler,
    device: torch.device,
    batch_size: int,
    N: int = 256,
    use_amp: bool = True,
) -> float:
    """Returns craft/move ratio; logs per-action mean ‖Δz‖₂."""
    model.eval()
    dtype = torch.bfloat16 if use_amp else torch.float32
    first_frames: list[np.ndarray] = []
    while sum(x.shape[0] for x in first_frames) < N:
        bs = min(batch_size, 128)
        obs_np, _ = val_sampler.sample_batch(bs)
        first_frames.append(obs_np[:, 0])
        if sum(x.shape[0] for x in first_frames) > N * 4:
            break
    obs_np = np.concatenate(first_frames, axis=0)[:N]
    states = torch.from_numpy(obs_np).float().div_(255.0).permute(0, 3, 1, 2).to(
        device
    )
    n = states.shape[0]
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        z_curr = model.encode(states)
    deltas = torch.zeros(n, ACTION_DIM, device=device, dtype=torch.float32)
    for a in range(ACTION_DIM):
        a_oh = torch.zeros(n, 1, ACTION_DIM, device=device)
        a_oh[:, 0, a] = 1.0
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            z_next = model.predictor(z_curr.unsqueeze(1), a_oh).squeeze(1)
        delta = (z_next.float() - z_curr.float()).norm(dim=-1)
        deltas[:, a] = delta
    move_delta = deltas[:, 0:5].mean().item()
    craft_delta = deltas[:, 6:].mean().item()
    ratio = craft_delta / max(move_delta, 1e-6)
    per_action = deltas.mean(dim=0)
    spread = float(per_action.std().item() / max(per_action.mean().item(), 1e-6))
    names = [
        "noop",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "do",
        "sleep",
        "place_stone",
        "place_table",
        "place_furnace",
        "place_plant",
        "make_wood_pickaxe",
        "make_stone_pickaxe",
        "make_iron_pickaxe",
        "make_wood_sword",
        "make_stone_sword",
        "make_iron_sword",
    ]
    print("\n[diag] per-action mean ‖Δz‖₂")
    print(f"{'action':<22} {'mean':>10}")
    print("-" * 34)
    for a in range(ACTION_DIM):
        print(f"{names[a]:<22} {per_action[a].item():>10.4f}")
    print(
        f"[diag] movement_delta={move_delta:.4f} craft_delta={craft_delta:.4f} "
        f"ratio={ratio:.4f} spread={spread:.4f}\n"
    )
    model.train()
    return ratio


def lr_at_step(step: int, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Linear warmup then cosine decay to lr_min (global step index ``step``)."""
    if step < warmup_steps:
        return lr_max * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(args: argparse.Namespace) -> None:
    cfg = {**DEFAULT_CONFIG, **vars(args)}
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (
        device.type == "cuda"
        and cfg.get("precision") == "bf16"
        and torch.cuda.is_bf16_supported()
    )

    random_path = _resolve_path(cfg["random_buffer"])
    ppo_path = _resolve_path(cfg["ppo_buffer"])

    with open(random_path, "rb") as f:
        random_data = pickle.load(f)
    with open(ppo_path, "rb") as f:
        ppo_data = pickle.load(f)

    r_train, r_val = split_trajectories(random_data["trajectories"], cfg["val_frac"], rng)
    p_train, p_val = split_trajectories(ppo_data["trajectories"], cfg["val_frac"], rng)

    train_sampler = MixedTransitionSampler(
        random_trajs=r_train,
        ppo_trajs=p_train,
        seq_len=cfg["seq_len"],
        random_ratio=cfg["random_ratio"],
        seed=int(cfg["seed"]),
    )
    val_sampler = MixedTransitionSampler(
        random_trajs=r_val,
        ppo_trajs=p_val,
        seq_len=cfg["seq_len"],
        random_ratio=cfg["random_ratio"],
        seed=int(cfg["seed"]) + 1,
    )

    model = LeWM(
        img_size=64,
        patch_size=8,
        latent_dim=cfg["latent_dim"],
        action_dim=ACTION_DIM,
        encoder_depth=12,
        encoder_heads=3,
        predictor_depth=6,
        predictor_heads=16,
        context_len=cfg["context_len"],
        sigreg_M=cfg["sigreg_M"],
        sigreg_lambda=cfg["sigreg_lambda"],
        dropout=0.1,
        predictor_hidden_dim=cfg["predictor_hidden_dim"],
    ).to(device)

    verify_adaln_zero_init(model)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    logdir = Path(cfg["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    metrics_path = logdir / "metrics.jsonl"
    metrics_f = open(metrics_path, "w", encoding="utf-8")

    def log(row: dict) -> None:
        line = " | ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
        )
        print(line)
        metrics_f.write(json.dumps(row, default=str) + "\n")
        metrics_f.flush()

    ckpt_args = {
        "img_size": 64,
        "patch_size": 8,
        "latent_dim": cfg["latent_dim"],
        "predictor_hidden_dim": cfg["predictor_hidden_dim"],
        "encoder_depth": 12,
        "encoder_heads": 3,
        "predictor_depth": 6,
        "predictor_heads": 16,
        "context_len": cfg["context_len"],
        "sigreg_M": cfg["sigreg_M"],
        "sigreg_lambda": cfg["sigreg_lambda"],
        "dropout": 0.1,
    }

    best_val = float("inf")
    dtype = torch.bfloat16 if use_amp else torch.float32

    t0 = time.perf_counter()
    for step in range(cfg["n_steps"]):
        model.train()
        obs_np, actions_np = train_sampler.sample_batch(cfg["batch_size"])
        obs = torch.from_numpy(obs_np).float().div_(255.0).permute(0, 1, 4, 2, 3).to(
            device
        )
        actions = torch.from_numpy(actions_np).long().to(device)
        a_oh = F.one_hot(actions, num_classes=ACTION_DIM).float()

        lr_now = lr_at_step(
            step,
            cfg["warmup_steps"],
            cfg["n_steps"],
            cfg["lr"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(obs, a_oh, rollout_steps=0)
            loss = out["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        optimizer.step()

        if step % cfg.get("log_every", 100) == 0:
            lr = lr_now
            log(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "pred_loss": float(out["pred_loss"]),
                    "sigreg_loss": float(out["sigreg_loss"]),
                    "lr": lr,
                    "time_s": time.perf_counter() - t0,
                }
            )
            t0 = time.perf_counter()

        if step % cfg["val_every"] == 0:
            val_loss = _evaluate_loss_impl(
                model,
                val_sampler,
                device,
                cfg["batch_size"],
                cfg["val_batches"],
                use_amp,
            )
            log({"step": step, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": ckpt_args,
                        "val_loss": val_loss,
                    },
                    logdir / "best.pt",
                )

        diag_ratio = None
        if step > 0 and step % cfg["diagnostic_every"] == 0:
            diag_ratio = run_action_magnitude_diagnostic(
                model,
                val_sampler,
                device,
                cfg["batch_size"],
                N=256,
                use_amp=use_amp,
            )
            log({"step": step, "diag_ratio": float(diag_ratio)})

        if step > 0 and step % cfg["checkpoint_every"] == 0:
            name = f"step_{step}"
            if diag_ratio is not None:
                name += f"_ratio_{diag_ratio:.1f}"
            payload = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": ckpt_args,
                "val_loss": best_val,
            }
            torch.save(payload, logdir / f"{name}.pt")
            torch.save(payload, logdir / "latest.pt")

    torch.save(
        {
            "step": cfg["n_steps"] - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": ckpt_args,
            "val_loss": best_val,
        },
        logdir / "latest.pt",
    )

    metrics_f.close()
    print(f"Done. Logs: {metrics_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LeWM v2 (mixed buffers + wide predictor)")
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--random-buffer", type=str, default=DEFAULT_CONFIG["random_buffer"])
    p.add_argument("--ppo-buffer", type=str, default=DEFAULT_CONFIG["ppo_buffer"])
    p.add_argument("--random-ratio", type=float, default=DEFAULT_CONFIG["random_ratio"])
    p.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG["seq_len"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--n-steps", type=int, default=DEFAULT_CONFIG["n_steps"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
    p.add_argument("--latent-dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    p.add_argument(
        "--predictor-hidden-dim",
        type=int,
        default=DEFAULT_CONFIG["predictor_hidden_dim"],
    )
    p.add_argument("--context-len", type=int, default=DEFAULT_CONFIG["context_len"])
    p.add_argument("--sigreg-M", type=int, default=DEFAULT_CONFIG["sigreg_M"])
    p.add_argument(
        "--sigreg-lambda", type=float, default=DEFAULT_CONFIG["sigreg_lambda"]
    )
    p.add_argument("--val-frac", type=float, default=DEFAULT_CONFIG["val_frac"])
    p.add_argument("--val-every", type=int, default=DEFAULT_CONFIG["val_every"])
    p.add_argument("--val-batches", type=int, default=DEFAULT_CONFIG["val_batches"])
    p.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CONFIG["checkpoint_every"]
    )
    p.add_argument(
        "--diagnostic-every", type=int, default=DEFAULT_CONFIG["diagnostic_every"]
    )
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument(
        "--precision",
        type=str,
        default=DEFAULT_CONFIG["precision"],
        choices=("bf16", "fp32"),
    )
    p.add_argument("--log-every", type=int, default=100)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    # Flatten to names matching DEFAULT_CONFIG keys
    ns = argparse.Namespace(
        logdir=args.logdir,
        random_buffer=args.random_buffer,
        ppo_buffer=args.ppo_buffer,
        random_ratio=args.random_ratio,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        context_len=args.context_len,
        sigreg_M=args.sigreg_M,
        sigreg_lambda=args.sigreg_lambda,
        val_frac=args.val_frac,
        val_every=args.val_every,
        val_batches=args.val_batches,
        checkpoint_every=args.checkpoint_every,
        diagnostic_every=args.diagnostic_every,
        seed=args.seed,
        precision=args.precision,
        log_every=args.log_every,
    )
    train(ns)


if __name__ == "__main__":
    main()
