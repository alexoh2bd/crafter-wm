"""Load LeWM checkpoints saved by letrain.py (handles DataParallel `module.` prefix)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ledata import ACTION_DIM
from lemodel import LeWM


def _as_dict(a: Any) -> dict:
    if isinstance(a, dict):
        return dict(a)
    return vars(a)


def merge_config_into_args(
    train_args: dict,
    config_path: str | None,
) -> dict:
    """Optional YAML config overrides / fills keys (same flatten as letrain)."""
    if not config_path:
        return train_args
    path = Path(config_path)
    if not path.is_file():
        return train_args
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    flat: dict = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    out = dict(train_args)
    out.update(flat)
    return out


def strip_dataparallel_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if not any(k.startswith("module.") for k in keys):
        return dict(state_dict)
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_lewm(
    checkpoint_path: str | Path,
    device: torch.device,
    config_path: str | None = None,
) -> LeWM:
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=device)
    raw_args = ckpt["args"]
    args = _as_dict(raw_args)
    args = merge_config_into_args(args, config_path)

    model = LeWM(
        img_size=64,
        patch_size=8,
        latent_dim=args.get("latent_dim", 256),
        action_dim=ACTION_DIM,
        encoder_depth=args.get("encoder_depth", 12),
        encoder_heads=args.get("encoder_heads", 3),
        predictor_depth=args.get("predictor_depth", 6),
        predictor_heads=args.get("predictor_heads", 16),
        context_len=args.get("context_len", 16),
        sigreg_M=args.get("sigreg_M", 1024),
        sigreg_lambda=args.get("sigreg_lambda", 0.1),
    ).to(device)

    sd = strip_dataparallel_prefix(ckpt["model"])
    model.load_state_dict(sd)
    model.eval()
    return model


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, C) uint8 -> (1, C, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(obs).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def obs_to_chw_float(obs: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 -> (C, H, W) float32 CPU in [0, 1]."""
    t = torch.from_numpy(obs).float() / 255.0
    return t.permute(2, 0, 1)
