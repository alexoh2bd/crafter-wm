# Dependencies (video): pip install imageio imageio-ffmpeg pillow
# Also requires: torch, crafter, numpy (same as training).

"""
Record a Crafter rollout with LeWM latent prediction metrics overlaid (MP4 or GIF).

Computes the same teacher-forcing objective as training (pred_loss / sigreg_loss from
LeWM.forward) over a sliding window of length ``context_len``.
Run from repo root:  PYTHONPATH=src python src/visualize_lewm_crafter.py --checkpoint ...
Or:  cd src && python visualize_lewm_crafter.py ...
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import crafter
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.cuda.amp import autocast

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ledata import ACTION_DIM  # noqa: E402
from lewm_checkpoint import load_lewm, obs_to_chw_float  # noqa: E402


def _pick_action(env: crafter.Env, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return int(np.random.choice([1, 2, 3, 4, 5, 5, 5]))


def _try_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        p = Path(name)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _overlay_frame(
    rgb_uint8: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    img = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(img)
    font = _try_font(13)
    y = 4
    x = 4
    for line in lines:
        draw.text(
            (x, y),
            line,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        y += 16
    return np.asarray(img)


def run_visualization(
    checkpoint: str,
    output: str,
    max_steps: int,
    max_episodes: int | None,
    fps: float,
    epsilon: float,
    device_str: str,
    config_path: str | None,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_lewm(checkpoint, device, config_path=config_path)
    context_len = int(model.predictor.context_len)
    print(f"Loaded LeWM from {checkpoint} (context_len={context_len})")

    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise SystemExit(
            "imageio is required for video output. Install with:\n"
            "  pip install imageio imageio-ffmpeg"
        ) from e

    env = crafter.Env()
    obs = env.reset()
    buf: deque[tuple[torch.Tensor, int]] = deque(maxlen=context_len)

    step_idx = 0
    ep_idx = 0
    completed_eps = 0
    ep_return = 0.0
    frames: list[np.ndarray] = []
    last_achievement_line = ""

    while step_idx < max_steps:
        action = _pick_action(env, epsilon)
        buf.append((obs_to_chw_float(obs), action))

        pred_loss = None
        sigreg_loss = None
        total_loss = None
        if len(buf) == context_len:
            obs_seq = torch.stack([x[0] for x in buf], dim=0).unsqueeze(0).to(device)
            a_seq = torch.zeros(1, context_len, ACTION_DIM, device=device)
            for t, (_, ai) in enumerate(buf):
                a_seq[0, t, ai] = 1.0
            with torch.no_grad():
                with autocast():
                    out = model(obs_seq, a_seq)
            pred_loss = float(out["pred_loss"])
            sigreg_loss = float(out["sigreg_loss"])
            total_loss = float(out["loss"].item())

        obs_next, reward, done, info = env.step(action)
        ep_return += float(reward)

        ach = info.get("achievements") or {}
        if ach:
            last_achievement_line = "ach: " + ", ".join(f"{k}:{v}" for k, v in sorted(ach.items()) if v)

        lines = [
            f"step {step_idx}  ep {ep_idx}  return {ep_return:.1f}",
        ]
        if pred_loss is not None:
            lines.append(f"pred_loss={pred_loss:.5f}  sigreg={sigreg_loss:.5f}  loss={total_loss:.5f}")
        else:
            lines.append(f"warmup {len(buf)}/{context_len} …")
        if last_achievement_line:
            lines.append(last_achievement_line[:120])

        frame = _overlay_frame(obs_next.astype(np.uint8), lines)
        frames.append(frame)

        obs = obs_next
        step_idx += 1

        if done:
            completed_eps += 1
            if max_episodes is not None and completed_eps >= max_episodes:
                break
            obs = env.reset()
            ep_idx += 1
            ep_return = 0.0

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    print(f"Writing {len(frames)} frames -> {out_path} ({ext or 'default'}, fps={fps})")
    if ext == ".gif":
        imageio.mimsave(str(out_path), frames, fps=fps)
    else:
        imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264", quality=8)
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LeWM checkpoint → Crafter simulation video with latent losses overlaid.",
        epilog="Install: pip install imageio imageio-ffmpeg pillow  (plus torch, crafter)",
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt / latest.pt")
    p.add_argument("--output", type=str, required=True, help="Output .mp4 or .gif")
    p.add_argument("--max_steps", type=int, default=5000, help="Cap simulation steps (default 5000)")
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Stop after this many completed episodes (optional)",
    )
    p.add_argument("--fps", type=float, default=12.0, help="Video frame rate (default 12)")
    p.add_argument("--epsilon", type=float, default=0.2, help="Epsilon-greedy scripted policy")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML to merge keys (e.g. context_len) with checkpoint args",
    )
    args = p.parse_args()
    run_visualization(
        checkpoint=args.checkpoint,
        output=args.output,
        max_steps=args.max_steps,
        max_episodes=args.max_episodes,
        fps=args.fps,
        epsilon=args.epsilon,
        device_str=args.device,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
