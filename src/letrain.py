"""
LeWM training script for Crafter.
Multi-GPU via torch DDP.
Logs: pred_loss, sigreg_loss, total_loss to wandb + jsonl.
"""

import os
import json
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from lemodel import LeWM
from ledata import CrafterDataset, collect_crafter_data, ACTION_DIM


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ──────────────────────────────────────────────────────────────
    if not Path(args.data_path).exists():
        print(f"Collecting {args.n_episodes} Crafter episodes...")
        collect_crafter_data(
            n_episodes=args.n_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            save_path=args.data_path,
        )

    dataset = CrafterDataset(args.data_path, context_len=args.context_len)
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = LeWM(
        img_size=64,
        patch_size=8,
        latent_dim=args.latent_dim,
        action_dim=ACTION_DIM,
        encoder_depth=12,
        encoder_heads=3,
        predictor_depth=6,
        predictor_heads=16,
        context_len=args.context_len,
        sigreg_M=args.sigreg_M,
        sigreg_lambda=args.sigreg_lambda,
        dropout=0.1,
    ).to(device)

    # Optional: resume / finetune from an existing checkpoint
    resume_path = getattr(args, 'resume', None)
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt_resume = torch.load(resume_path, map_location=device)
        # Strip DataParallel prefix if present
        sd = {k.replace('module.', '', 1): v
              for k, v in ckpt_resume['model'].items()}
        model.load_state_dict(sd, strict=False)

    # Optional: freeze encoder so only the predictor is trained
    if getattr(args, 'freeze_encoder', False):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("Encoder frozen — training predictor only")

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

    # ── Optimizer (only over trainable params) ─────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine LR schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()  # Mixed precision

    # ── Logging ───────────────────────────────────────────────────────────
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    log_file = open(logdir / 'metrics.jsonl', 'w')

    use_wandb = getattr(args, 'use_wandb', False)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print('Warning: use_wandb=True but wandb is not installed; install with `pip install wandb`')
        else:
            run_name = getattr(args, 'wandb_run_name', None) or logdir.name
            wandb_run = wandb.init(
                project=getattr(args, 'wandb_project', 'lewm-crafter'),
                name=run_name,
                config={
                    k: v for k, v in vars(args).items()
                    if v is not None and isinstance(v, (bool, int, float, str))
                },
                dir=str(logdir),
            )

    def log(metrics, step=None):
        print(' | '.join(f'{k}={v:.4f}' if isinstance(v, float)
                         else f'{k}={v}' for k, v in metrics.items()))
        log_file.write(json.dumps(metrics) + '\n')
        log_file.flush()
        if wandb_run is not None:
            import wandb
            wstep = step if step is not None else metrics.get('step', metrics.get('epoch'))
            payload = {}
            for k, v in metrics.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, float):
                    payload[k] = v
                elif isinstance(v, int):
                    payload[k] = float(v)
            if wstep is not None:
                wandb.log(payload, step=int(wstep))
            else:
                wandb.log(payload)

    # ── Training Loop ─────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float('inf')

    try:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()

            for batch_idx, (obs_seq, a_seq) in enumerate(train_loader):
                obs_seq = obs_seq.to(device, non_blocking=True)   # (B, T, C, H, W)
                a_seq = a_seq.to(device, non_blocking=True)       # (B, T, A)

                optimizer.zero_grad()

                rollout_steps = getattr(args, 'rollout_steps', 0)
                rollout_loss_weight = getattr(args, 'rollout_loss_weight', 0.1)
                fwd_kwargs = dict(
                    rollout_steps=rollout_steps,
                    rollout_loss_weight=rollout_loss_weight,
                )
                with autocast():
                    m = model if not isinstance(model, nn.DataParallel) else model.module
                    out = m(obs_seq, a_seq, **fwd_kwargs)
                    loss = out['loss']

                scaler.scale(loss).backward()

                # Gradient clipping (important for ViT stability)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1

                if global_step % args.log_every == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_entry = {
                        'epoch': epoch,
                        'step': global_step,
                        'loss': out['loss'].item(),
                        'pred_loss': out['pred_loss'],
                        'sigreg_loss': out['sigreg_loss'],
                        'lr': lr,
                        'time': time.time() - t0,
                    }
                    if 'rollout_loss' in out:
                        log_entry['rollout_loss'] = out['rollout_loss']
                    log(log_entry)
                    t0 = time.time()

            # ── Validation (teacher-forced loss only — no rollout loss) ───
            model.eval()
            val_losses = []
            with torch.no_grad():
                for obs_seq, a_seq in val_loader:
                    obs_seq = obs_seq.to(device)
                    a_seq = a_seq.to(device)
                    with autocast():
                        m = model if not isinstance(model, nn.DataParallel) else model.module
                        out = m(obs_seq, a_seq, rollout_steps=0)
                    val_losses.append(out['loss'].item())

            val_loss = sum(val_losses) / len(val_losses)
            log({'epoch': epoch, 'val_loss': val_loss}, step=global_step)

            # Save checkpoint
            ckpt = {
                'epoch': epoch,
                'step': global_step,
                'model': model.state_dict() if not isinstance(model, nn.DataParallel)
                         else model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }
            torch.save(ckpt, logdir / 'latest.pt')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, logdir / 'best.pt')
                print(f"  New best val_loss={val_loss:.4f}")

        print(f"Training complete. Best val_loss={best_val_loss:.4f}")
        print(f"Checkpoint saved to {logdir}/best.pt")
    finally:
        log_file.close()
        if wandb_run is not None:
            import wandb
            wandb.finish()


def _load_config(config_path: str) -> dict:
    """Flatten config.yaml into a single dict of key→value pairs."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    # Data
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--n_episodes', type=int)
    parser.add_argument('--max_steps_per_episode', type=int)
    parser.add_argument('--context_len', type=int)
    # Model
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--sigreg_M', type=int)
    parser.add_argument('--sigreg_lambda', type=float)
    # Training
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--total_steps', type=int)
    parser.add_argument('--log_every', type=int)
    parser.add_argument('--seed', type=int)
    # Rollout / finetune
    parser.add_argument('--rollout_steps', type=int,
                        help='Open-loop steps for multi-step rollout loss (0 = disabled)')
    parser.add_argument('--rollout_loss_weight', type=float,
                        help='Weight applied to the rollout loss term')
    parser.add_argument('--freeze_encoder', action='store_true', default=None,
                        help='Freeze encoder weights (predictor-only finetune)')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume/finetune from')
    # Infra
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--use_wandb', action='store_true', default=None,
                        help='Log metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    cli_args = parser.parse_args()

    # Load YAML defaults, then override with any CLI flags that were explicitly set
    defaults = {
        'data_path': 'data/crafter/ppo_rollouts/crafter_data.pkl',
        'n_episodes': 1000,
        'max_steps_per_episode': 10_000,
        'context_len': 16,
        'latent_dim': 256,
        'sigreg_M': 1024,
        'sigreg_lambda': 0.1,
        'epochs': 100,
        'batch_size': 128,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'warmup_steps': 1000,
        'total_steps': 200_000,
        'log_every': 50,
        'seed': 0,
        'logdir': 'data/crafter/world_model/lewm_crafter',
        'use_wandb': False,
        'wandb_project': 'lewm-crafter',
        'wandb_run_name': None,
        'rollout_steps': 0,
        'rollout_loss_weight': 0.1,
        'freeze_encoder': False,
        'resume': None,
    }

    if Path(cli_args.config).exists():
        defaults.update(_load_config(cli_args.config))

    # CLI flags win over YAML
    for key, val in vars(cli_args).items():
        if key != 'config' and val is not None:
            defaults[key] = val

    args = argparse.Namespace(**defaults)
    train(args)